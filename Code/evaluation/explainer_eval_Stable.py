import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from Code.training.DataHandler import DataHandler
from Code.training.DatasetConfig import get_checkpoint_dir, get_default_checkpoint
from Code.training.Model_sparse import Model
from Code.training.Params import args

EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(EVALUATION_DIR)
MODEL_DIR = get_checkpoint_dir(CODE_ROOT, args.dataset, args.checkpoint_dir)
BASE_MODEL_PATH = get_default_checkpoint(CODE_ROOT, args.dataset, args.checkpoint_dir)
STABLE_MODEL_PATH = os.path.join(MODEL_DIR, 'stable_model_1.pkl')

# 双视角的解释超参数
TOP_K1 = 5
TOP_K2 = 25
TOP_HE = 64
TOP_HE_NODES =5


def get_full_graph_attention(model, handler, model_name="Model"):
    print(f"Running full graph forward pass for {model_name}...")
    model.eval()

    captured = {'logits': None, 'probs': None, 'hyper_weights': []}

    # 1. 图注意力卧底 Hook
    def hook_forward_fn(embeds, adj):
        N = embeds.size(0)
        idxs = adj._indices()
        row, col = idxs[0], idxs[1]
        edge_rep = torch.cat([embeds[row], embeds[col]], dim=-1)
        scores = model.attention.edge_score(edge_rep).squeeze(-1)
        att_logits = torch.full((N, N), float('-inf'), device=embeds.device)
        att_logits[row, col] = scores

        att_probs = torch.softmax(att_logits, dim=-1)

        captured['logits'] = att_logits.detach()
        captured['probs'] = att_probs.detach()

        return att_probs

    # 2. 超图注意力卧底 Hook
    def hook_forward_hypergraph(node_embeds, hypergraph_adj, chunk_size=512):
        hyperedge_embeds = hypergraph_adj.T @ node_embeds
        num_nodes, num_hyperedges = hypergraph_adj.shape
        node_hyperedge_weights = torch.zeros((num_nodes, num_hyperedges), device=node_embeds.device)

        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            node_chunk = node_embeds[start:end]
            scores = torch.matmul(node_chunk, hyperedge_embeds.T).view(-1, 1)
            att_logits = model.attention.score_mlp(scores).view(end - start, num_hyperedges, 2)
            node_hyperedge_weights[start:end] = F.softmax(att_logits, dim=-1)[:, :, 1]

        captured['hyper_weights'].append(node_hyperedge_weights.detach())
        return node_hyperedge_weights

    original_forward = model.attention.forward
    original_h_forward = model.attention.forward_hypergraph

    model.attention.forward = hook_forward_fn
    model.attention.forward_hypergraph = hook_forward_hypergraph

    try:
        with torch.no_grad():
            model.predict(
                handler.torchBiAdj,
                handler.test_d,
                handler.test_c,
                drug_batch=handler.drug_batch,
                omics_inputs=handler.omics_inputs_all
            )
    finally:
        model.attention.forward = original_forward
        model.attention.forward_hypergraph = original_h_forward

    # 提取最后一层超图注意力 (倒数第二个是drug，倒数第一个是cell)
    drug_hyper_w = captured['hyper_weights'][-2]
    cell_hyper_w = captured['hyper_weights'][-1]

    return captured['logits'], captured['probs'], drug_hyper_w, cell_hyper_w


def mine_node_subgraphs(att_logits_matrix, total_nodes, top_k1, top_k2):
    print("Mining Graph explanation subgraphs based on Base Model...")
    att_matrix_np = att_logits_matrix.cpu().numpy()
    node_subgraphs = {}

    for global_idx in tqdm(range(total_nodes), desc="Graph Explanations"):
        edges = []
        row_att = att_matrix_np[global_idx, :].copy()
        row_att[global_idx] = -np.inf # 排除自环

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
            n_row = att_matrix_np[neighbor, :].copy()
            n_row[neighbor] = -np.inf
            n_valid_indices = np.where(n_row > -1e10)[0]

            for second_neighbor in n_valid_indices:
                score = n_row[second_neighbor]
                candidate_2nd_edges.append({'u': int(neighbor), 'v': int(second_neighbor), 'score': score})

        if len(candidate_2nd_edges) > 0:
            candidate_2nd_edges.sort(key=lambda x: x['score'], reverse=True)
            limit_k2 = min(len(candidate_2nd_edges), top_k2)
            top_2nd_edges = candidate_2nd_edges[:limit_k2]
            for item in top_2nd_edges:
                edges.append((item['u'], item['v']))

        node_subgraphs[global_idx] = edges
    return node_subgraphs


def mine_hypergraph_subgraphs(drug_hyper_w, cell_hyper_w, total_nodes, top_he, top_he_nodes):
    print("Mining Hypergraph explanation subgraphs based on Base Model...")
    hyper_subgraphs = {}
    drug_hw_np = drug_hyper_w.cpu().numpy()
    cell_hw_np = cell_hyper_w.cpu().numpy()

    for global_idx in tqdm(range(total_nodes), desc="Hypergraph Explanations"):
        pairs = []
        if global_idx < args.drug:
            local_idx = global_idx
            weights = drug_hw_np
        else:
            local_idx = global_idx - args.drug
            weights = cell_hw_np

        node_row = weights[local_idx, :]
        top_he_indices = np.argsort(node_row)[-top_he:][::-1]

        for he_idx in top_he_indices:
            # 记录目标节点自身与这条关键超边的连接分量
            pairs.append((local_idx, he_idx))

            # 记录这条超边内起核心贡献的其他节点的连接分量
            col_vals = weights[:, he_idx]
            top_node_indices = np.argsort(col_vals)[-top_he_nodes:][::-1]
            for n_idx in top_node_indices:
                pairs.append((n_idx, he_idx))

        hyper_subgraphs[global_idx] = list(set(pairs))
    return hyper_subgraphs


def make_model(handler):
    drug_cfg = {'d_atom': handler.drug_batch.x.shape[1], 'd_model': args.latdim, 'dropout': 0.1,
                'num_total_atoms': handler.drug_batch.x.shape[0]}
    # 【这里已修正为下划线 omics_inputs_all】
    omics_cfg = {'omics_dims': [handler.omics_inputs_all[i].shape[1] for i in range(len(handler.omics_inputs_all))],
                 'num_cells': handler.omics_inputs_all[0].shape[0], 'hidden_dim': 128, 'proj_dim': args.latdim}
    return Model(drug_encoder_cfg=drug_cfg, omics_encoder_cfg=omics_cfg).to(args.device)

def main():
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print("Loading Data...")
    handler = DataHandler()
    handler.LoadData()

    # 传入 handler
    base_model = make_model(handler)
    stable_model = make_model(handler)

    print(f"Loading Base: {BASE_MODEL_PATH}")
    base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=args.device))
    print(f"Loading Stable: {STABLE_MODEL_PATH}")
    stable_model.load_state_dict(torch.load(STABLE_MODEL_PATH, map_location=args.device))

    # =============== 新增：构造全集图 (Train + Test) ===============
    print("Constructing Full Eval Graph (Train + Test)...")
    train_d_np = handler.train_d.cpu().numpy()
    train_c_np = handler.train_c.cpu().numpy()
    test_d_np = handler.test_d.cpu().numpy()
    test_c_np = handler.test_c.cpu().numpy()

    all_d_np = np.concatenate([train_d_np, test_d_np])
    all_c_np = np.concatenate([train_c_np, test_c_np])

    full_eval_adj = handler.getAdj(all_d_np, all_c_np).to(args.device)

    # 为了满足 get_full_graph_attention 里 predict 的参数要求 (不影响注意力分布，只需要合法输入即可)。
    # 我们这里传测试集的 d 和 c。实际上在这个函数里我们只关心前向传播中被 hook 截获的矩阵。
    handler.torchBiAdj = full_eval_adj

    # 提取 Base 和 Stable 模型的所有双通道权重
    base_logits, base_probs, base_dh_w, base_ch_w = get_full_graph_attention(base_model, handler, "Base Model")
    _, stable_probs, stable_dh_w, stable_ch_w = get_full_graph_attention(stable_model, handler, "Stable Model")

    total_nodes = base_logits.shape[0]

    # 仅使用 Base 模型来寻找图与超图“标准答案”解释结构
    node_subgraphs = mine_node_subgraphs(base_logits, total_nodes, TOP_K1, TOP_K2)
    hyper_subgraphs = mine_hypergraph_subgraphs(base_dh_w, base_ch_w, total_nodes, TOP_HE, TOP_HE_NODES)

    # =============== 修改：这里改为使用 test 数据 ===============
    eval_drugs = test_d_np
    eval_cells = test_c_np
    num_drugs = args.drug

    graph_cosine_sims = []
    hypergraph_cosine_sims = []

    print(f"\nComparing Explanation Scores (Graph + Hypergraph) on {len(eval_drugs)} TEST Samples...")

    for i in tqdm(range(len(eval_drugs)), desc="Sample Similarity"):
        d_local = eval_drugs[i]
        c_local = eval_cells[i]

        d_global = d_local
        c_global = num_drugs + c_local

        # =============== [A] 提取图相关注意力向量 ===============
        d_edges = node_subgraphs.get(d_global, [])
        c_edges = node_subgraphs.get(c_global, [])
        explanation_edges = list(set(d_edges + c_edges))
        explanation_edges.append((int(d_global), int(c_global)))

        if len(explanation_edges) > 0:
            u_t = torch.tensor([e[0] for e in explanation_edges], device=args.device)
            v_t = torch.tensor([e[1] for e in explanation_edges], device=args.device)
            vec_base_g = base_probs[u_t, v_t]
            vec_stable_g = stable_probs[u_t, v_t]
        else:
            vec_base_g = torch.empty(0, device=args.device)
            vec_stable_g = torch.empty(0, device=args.device)

        # =============== [B] 提取超图相关注意力向量 ===============
        d_hyper_pairs = hyper_subgraphs.get(d_global, [])
        c_hyper_pairs = hyper_subgraphs.get(c_global, [])

        # Drug 超图注意力分量
        if len(d_hyper_pairs) > 0:
            dn_idx = torch.tensor([p[0] for p in d_hyper_pairs], device=args.device)
            dhe_idx = torch.tensor([p[1] for p in d_hyper_pairs], device=args.device)
            vec_base_dh = base_dh_w[dn_idx, dhe_idx]
            vec_stable_dh = stable_dh_w[dn_idx, dhe_idx]
        else:
            vec_base_dh = torch.empty(0, device=args.device)
            vec_stable_dh = torch.empty(0, device=args.device)

        # Cell 超图注意力分量
        if len(c_hyper_pairs) > 0:
            cn_idx = torch.tensor([p[0] for p in c_hyper_pairs], device=args.device)
            che_idx = torch.tensor([p[1] for p in c_hyper_pairs], device=args.device)
            vec_base_ch = base_ch_w[cn_idx, che_idx]
            vec_stable_ch = stable_ch_w[cn_idx, che_idx]
        else:
            vec_base_ch = torch.empty(0, device=args.device)
            vec_stable_ch = torch.empty(0, device=args.device)

        vec_base_h = torch.cat([vec_base_dh, vec_base_ch])
        vec_stable_h = torch.cat([vec_stable_dh, vec_stable_ch])

        graph_sim = F.cosine_similarity(
            vec_base_g.unsqueeze(0), vec_stable_g.unsqueeze(0)
        ).item()
        hypergraph_sim = F.cosine_similarity(
            vec_base_h.unsqueeze(0), vec_stable_h.unsqueeze(0)
        ).item()
        graph_cosine_sims.append(graph_sim)
        hypergraph_cosine_sims.append(hypergraph_sim)

    graph_mean = np.mean(graph_cosine_sims)
    graph_std = np.std(graph_cosine_sims)
    hypergraph_mean = np.mean(hypergraph_cosine_sims)
    hypergraph_std = np.std(hypergraph_cosine_sims)

    print("\n" + "=" * 50)
    print("      EXPLANATION STABILITY (Graph + Hypergraph)")
    print("=" * 50)
    print(f"Base Model: {os.path.basename(BASE_MODEL_PATH)}")
    print(f"Stable Model: {os.path.basename(STABLE_MODEL_PATH)}")
    print(f"Subgraphs mined from Base Model (Full Graph)")
    print(f" - Graph: Top K1={TOP_K1}, Top K2={TOP_K2}")
    print(f" - Hypergraph: Top HE={TOP_HE}, Top HE_NODES={TOP_HE_NODES}")
    print(f"Total TEST Samples Compared: {len(graph_cosine_sims)}")
    print("-" * 50)
    print(f"Graph Similarity:      {graph_mean:.6f} (+/- {graph_std:.6f})")
    print(f"Hypergraph Similarity: {hypergraph_mean:.6f} (+/- {hypergraph_std:.6f})")
    print(f"Two-Channel Mean:      {(graph_mean + hypergraph_mean) / 2:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

