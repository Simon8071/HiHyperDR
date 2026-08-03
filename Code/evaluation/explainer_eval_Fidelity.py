import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from Code.training.DataHandler import DataHandler
from Code.training.DatasetConfig import get_default_checkpoint
from Code.training.Model_sparse import Model
from Code.training.Params import args

EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(EVALUATION_DIR)


HYPER_INCIDENCE_TEMP = 4.0


def get_subgraph_adj(edges, num_nodes, device):
    edge_tensor = torch.tensor(edges, dtype=torch.long, device=device).t()
    u, v = edge_tensor[0], edge_tensor[1]
    all_nodes = torch.arange(num_nodes, device=device)
    full_u = torch.cat([u, v, all_nodes])
    full_v = torch.cat([v, u, all_nodes])
    indices = torch.stack([full_u, full_v], dim=0)
    values = torch.ones(indices.shape[1], device=device)
    adj = torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes)
    ).coalesce()

    new_vals = (adj.values() > 0).float()
    idx = adj.indices()
    row_idx, col_idx = idx[0], idx[1]

    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row_idx, new_vals)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    norm_vals = new_vals * deg_inv_sqrt[row_idx] * deg_inv_sqrt[col_idx]

    return torch.sparse_coo_tensor(idx, norm_vals, (num_nodes, num_nodes))



def forward_with_mask(model, initial_embeds, adj, ddHyper, ccHyper, gcn_node_mask=None, hyper_node_mask=None):
    device = initial_embeds.device
    N = initial_embeds.shape[0]
    if gcn_node_mask is None:
        gcn_node_mask = torch.ones(N, device=device)
    if hyper_node_mask is None:
        hyper_node_mask = torch.ones(N, device=device)

    union_mask = ((gcn_node_mask + hyper_node_mask) > 0).float()
    curr_embed = initial_embeds * union_mask.unsqueeze(1)

    drug_h_mask = hyper_node_mask[:args.drug].unsqueeze(1)
    cell_h_mask = hyper_node_mask[args.drug:].unsqueeze(1)
    masked_ddHyper = ddHyper * drug_h_mask
    masked_ccHyper = ccHyper * cell_h_mask

    embedsLst = [curr_embed]

    for i in range(args.gnn_layer):
        gcn_input = embedsLst[-1] * gcn_node_mask.unsqueeze(1)
        gcnEmbeds = model.gcnLayer(adj, gcn_input)
        gcnEmbeds = gcnEmbeds * gcn_node_mask.unsqueeze(1)

        h_input = embedsLst[-1] * hyper_node_mask.unsqueeze(1)
        d_embeds = h_input[:args.drug]
        c_embeds = h_input[args.drug:]
        
        hyperDEmbeds = model.hgnnLayer(
            masked_ddHyper,
            d_embeds,
            node_hyperedge_weights=None,
        )
        hyperCEmbeds = model.hgnnLayer(
            masked_ccHyper,
            c_embeds,
            node_hyperedge_weights=None,
        )
        hyperEmbeds = torch.cat([hyperDEmbeds, hyperCEmbeds], axis=0)
        hyperEmbeds = hyperEmbeds * hyper_node_mask.unsqueeze(1)

        next_embed = gcnEmbeds + hyperEmbeds
        
        embedsLst.append(next_embed)

    embeds = sum(embedsLst)
    return embeds


def evaluate_fidelity_keep(
    top_k1=50,
    top_k2=50,
    top_he=64,
    top_he_nodes=5,
):
    """Evaluate predictions produced from only the explanation subgraph."""
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    print(f"Using device: {args.device}")

    print("Loading Data...")
    handler = DataHandler()
    handler.LoadData()
    print("Data Loaded.")

    drug_cfg = {
        'd_atom': handler.drug_batch.x.shape[1],
        'd_model': args.latdim,
        'dropout': 0.1,
        'num_total_atoms': handler.drug_batch.x.shape[0]
    }
    omics_cfg = {
        'omics_dims': [handler.omics_inputs_all[i].shape[1] for i in range(len(handler.omics_inputs_all))],
        'num_cells': handler.omics_inputs_all[0].shape[0],
        'hidden_dim': 128,
        'proj_dim': args.latdim
    }

    model = Model(drug_encoder_cfg=drug_cfg, omics_encoder_cfg=omics_cfg)
    model = model.to(args.device)

    model_path = get_default_checkpoint(
        CODE_ROOT, args.dataset, args.checkpoint_dir
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    print("Constructing Full Eval Graph (Train + Test)...")

    train_d_np = handler.train_d.cpu().numpy()
    train_c_np = handler.train_c.cpu().numpy()
    test_d_np = handler.test_d.cpu().numpy()
    test_c_np = handler.test_c.cpu().numpy()

    all_d = np.concatenate([train_d_np, test_d_np])
    all_c = np.concatenate([train_c_np, test_c_np])

    full_eval_adj = handler.getAdj(all_d, all_c).to(args.device)

    print("Step 1: Running Full Graph Forward Pass...")

    captured_data = {'att_logits': None}
    hyper_att_logs = []

    def hook_forward_fn(embeds, adj):
        N = embeds.size(0)
        idxs = adj._indices()
        row, col = idxs[0], idxs[1]
        edge_rep = torch.cat([embeds[row], embeds[col]], dim=-1)
        scores = model.attention.edge_score(edge_rep).squeeze(-1)
        att_matrix_logits = torch.full((N, N), float('-inf'), device=embeds.device)
        att_matrix_logits[row, col] = scores
        captured_data['att_logits'] = att_matrix_logits.detach()
        return torch.softmax(att_matrix_logits, dim=-1)

    def hook_forward_hypergraph(node_embeds, hypergraph_adj, chunk_size=512):
        hyperedge_embeds = hypergraph_adj.T @ node_embeds
        num_nodes, num_hyperedges = hypergraph_adj.shape
        node_hyperedge_weights = torch.zeros((num_nodes, num_hyperedges), device=node_embeds.device)
        node_hyperedge_logits = torch.zeros((num_nodes, num_hyperedges), device=node_embeds.device)
        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            node_chunk = node_embeds[start:end]
            scores = torch.matmul(node_chunk, hyperedge_embeds.T).reshape(-1, 1)
            att_logits = model.attention.score_mlp(scores).view(
                end - start,
                num_hyperedges,
                2,
            )
            node_hyperedge_logits[start:end] = att_logits[:, :, 1]
            node_hyperedge_weights[start:end] = F.softmax(
                att_logits,
                dim=-1,
            )[:, :, 1]
        hyper_att_logs.append(node_hyperedge_logits.detach())
        return node_hyperedge_weights

    def hook_build_hyper_incidence(node_embeds, hyper_param=None):
        if args.dense:
            scores = node_embeds @ hyper_param
        else:
            scores = node_embeds * args.mult
        incidence = F.softmax(
            scores / HYPER_INCIDENCE_TEMP,
            dim=1,
        )
        return incidence, scores

    original_attn_forward = model.attention.forward
    original_hyper_forward = model.attention.forward_hypergraph
    original_build_hyper_incidence = model.build_hyper_incidence

    model.attention.forward = hook_forward_fn
    model.attention.forward_hypergraph = hook_forward_hypergraph
    model.build_hyper_incidence = hook_build_hyper_incidence

    try:
        with torch.no_grad():
            preds_full_logits = model.predict(
                full_eval_adj,
                handler.test_d,
                handler.test_c,
                drug_batch=handler.drug_batch,
                omics_inputs=handler.omics_inputs_all
            )
            probs_full = torch.softmax(preds_full_logits, dim=1)
            labels_full_pred = torch.argmax(probs_full, dim=1)

            dEmbeds_tmp = model.encode_all_drugs(handler.drug_batch)
            cEmbeds_tmp = model.encode_all_cells(handler.omics_inputs_all)
            initial_embeds_full = torch.cat([dEmbeds_tmp, cEmbeds_tmp], dim=0)

            num_drugs = args.drug
            d_part = initial_embeds_full[:num_drugs]
            c_part = initial_embeds_full[num_drugs:]

            ddHyper_full = d_part * args.mult
            ccHyper_full = c_part * args.mult
            if args.dense:
                ddHyper_full = d_part @ model.dHyper
                ccHyper_full = c_part @ model.cHyper
    finally:
        model.attention.forward = original_attn_forward
        model.attention.forward_hypergraph = original_hyper_forward
        model.build_hyper_incidence = original_build_hyper_incidence

    print("Full graph predictions and initial embeddings cached.")

    print("Step 2: Mining Explanation Subgraphs (Graph & Hypergraph)...")

    att_matrix = captured_data['att_logits'].cpu().numpy()
    total_nodes = args.drug + args.cell
    node_subgraphs = {}

    for global_idx in tqdm(range(total_nodes), desc="Node Explanations"):
        edges = []
        row_att = att_matrix[global_idx, :].copy()
        row_att[global_idx] = -np.inf
        
        valid_indices = np.where(row_att > -1e10)[0]
        
        top_1st_hop = []
        if len(valid_indices) > 0:
            k_local = min(len(valid_indices), top_k1)
            
            valid_scores = row_att[valid_indices]
            sorted_indices_local = np.argsort(valid_scores)[::-1]
            top_indices = valid_indices[sorted_indices_local[:k_local]]
            top_1st_hop = top_indices.tolist()

            for neighbor in top_1st_hop:
                edges.append((int(global_idx), int(neighbor)))
        
        candidate_2nd_edges = []
        for neighbor in top_1st_hop:
            n_row = att_matrix[neighbor, :].copy()
            n_row[neighbor] = -np.inf
            
            n_valid_indices = np.where(n_row > -1e10)[0]
            
            for second_neighbor in n_valid_indices:
                score = n_row[second_neighbor]
                candidate_2nd_edges.append({
                    'u': int(neighbor),
                    'v': int(second_neighbor),
                    'score': score
                })
        
        if len(candidate_2nd_edges) > 0:
            candidate_2nd_edges.sort(key=lambda x: x['score'], reverse=True)
            
            limit_k2 = min(len(candidate_2nd_edges), top_k2)
            top_2nd_edges = candidate_2nd_edges[:limit_k2]

            for item in top_2nd_edges:
                edges.append((item['u'], item['v']))

        node_subgraphs[global_idx] = edges

    drug_hyper_w = hyper_att_logs[-2].cpu().numpy() 
    cell_hyper_w = hyper_att_logs[-1].cpu().numpy()

    hyper_subgraphs_nodes = {}

    for global_idx in tqdm(range(total_nodes), desc="Hypergraph Explanations"):
        extracted_nodes = set()

        if global_idx < args.drug:
            local_idx = global_idx
            weights = drug_hyper_w
            offset = 0
        else:
            local_idx = global_idx - args.drug
            weights = cell_hyper_w
            offset = args.drug

        node_row = weights[local_idx, :]
        top_he_indices = np.argsort(node_row)[-top_he:][::-1]

        for he_idx in top_he_indices:
            col_vals = weights[:, he_idx]
            top_node_indices = np.argsort(col_vals)[-top_he_nodes:][::-1]
            for n_idx in top_node_indices:
                real_global_idx = n_idx + offset
                extracted_nodes.add(int(real_global_idx))

        hyper_subgraphs_nodes[global_idx] = list(extracted_nodes)

    print("Step 3: Evaluating Fidelity Keep...")

    eval_drug_idxs = handler.test_d.cpu().numpy()
    eval_cell_idxs = handler.test_c.cpu().numpy()

    keep_prob_diffs = []

    keep_label_diffs = []

    # === 【新增初始化】用于记录三种图的预测标签 ===
    for i in tqdm(range(len(eval_drug_idxs)), desc="Eval Loop"):
        d_local = eval_drug_idxs[i]
        c_local = eval_cell_idxs[i]
        target_label = labels_full_pred[i].item()

        d_global = d_local
        c_global = num_drugs + c_local

        d_edges = node_subgraphs.get(d_global, [])
        c_edges = node_subgraphs.get(c_global, [])
        g_nodes = set([u for u, v in d_edges + c_edges] + [v for u, v in d_edges + c_edges] + [d_global, c_global])

        d_hyper_nodes = hyper_subgraphs_nodes.get(d_global, [])
        c_hyper_nodes = hyper_subgraphs_nodes.get(c_global, [])
        h_nodes = set(list(d_hyper_nodes) + list(c_hyper_nodes) + [d_global, c_global])

        gcn_keep_mask = torch.zeros(total_nodes, device=args.device)
        gcn_keep_mask[list(g_nodes)] = 1.0
        
        # >>> 在这里插入以下这 2 行，把目标节点在补图里复活！<<<
        gcn_keep_mask[int(d_global)] = 1.0
        gcn_keep_mask[int(c_global)] = 1.0

        hyper_keep_mask = torch.zeros(total_nodes, device=args.device)
        hyper_keep_mask[list(h_nodes)] = 1.0
        
        # >>> 同样在这里插入以下这 2 行，复活它们在超图里的根基！<<<
        hyper_keep_mask[int(d_global)] = 1.0
        hyper_keep_mask[int(c_global)] = 1.0

        adj_keep = get_subgraph_adj(list(d_edges + c_edges + [(d_global, c_global)]), total_nodes, args.device)
        
        with torch.no_grad():
            final_embeds_keep = forward_with_mask(
                model, initial_embeds_full, adj_keep, ddHyper_full, ccHyper_full,
                gcn_node_mask=gcn_keep_mask,
                hyper_node_mask=hyper_keep_mask
            )

            d_emb_k = final_embeds_keep[d_global].unsqueeze(0)
            c_emb_k = final_embeds_keep[c_global].unsqueeze(0)
            logits_k = model.classifierLayer(d_emb_k, c_emb_k)
            prob_k = torch.softmax(logits_k, dim=1)

            p_keep = prob_k[0, target_label].item()
            pred_label_k = torch.argmax(prob_k, dim=1).item()

        p_full = probs_full[i, target_label].item()

        keep_prob_diffs.append(abs(p_full - p_keep))

        keep_label_diffs.append(abs(target_label - pred_label_k))

    print("\n" + "=" * 50)
    print("      FIDELITY KEEP RESULTS (VS FULL MODEL PREDICTION)")
    print("=" * 50)
    print(f"Metrics calculated on {len(eval_drug_idxs)} test samples.")
    print(f"Top K1 (1st Hop): {top_k1}, Top K2 (2nd Hop): {top_k2}")
    print("-" * 50)
    print("Strategy: predict using only the explanation subgraph")
    print(f"  Avg Prob Diff (|Full - Keep|):    {np.mean(keep_prob_diffs):.6f}")
    print(f"  Label Flip Rate:                  {np.mean(keep_label_diffs):.6f}")
    print("=" * 50)
    metrics = {
        'top_k1': top_k1, 
        'top_k2': top_k2,
        'num_samples': len(eval_drug_idxs),
        'keep_prob_diff_mean': float(np.mean(keep_prob_diffs)),
        'keep_label_flip_rate': float(np.mean(keep_label_diffs))
    }

    return metrics


if __name__ == "__main__":
    k_combinations = [
        (5,5),
        (5,10),
        (5, 15),
        (5, 20),
        (5, 25),
    ]
    for k1, k2 in k_combinations:
        print(f"\nRunning grid: Top K1={k1}, Top K2={k2}")
        metrics = evaluate_fidelity_keep(
            top_k1=k1,
            top_k2=k2,
        )
