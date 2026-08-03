from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from Code.training.Params import args
from Code.training.FeatureInit import DrugEncoder, GraphCDROmicsEncoder
from Code.training.Utils.Utils import contrastLoss, l2_norm, ce

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class GATv2Score(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, in_dim, bias=True)
        self.a = nn.Linear(in_dim, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.W(x)
        h = self.act(h)
        return self.a(h)


class AttentionModule(nn.Module):

    def __init__(self, latdim):
        super().__init__()
        self.edge_score = GATv2Score(2 * latdim)
        self.score_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        self.last_att_matrix = None

    def forward(self, embeds, adj):
        N = embeds.size(0)
        idxs = adj._indices()
        row, col = idxs[0], idxs[1]
        edge_rep = torch.cat([embeds[row], embeds[col]], dim=-1)
        scores = self.edge_score(edge_rep).squeeze(-1)
        att_matrix = torch.full((N, N), float('-inf'), device=embeds.device)
        att_matrix[row, col] = scores
        att_matrix = torch.softmax(att_matrix, dim=-1)
        self.last_att_matrix = att_matrix.detach()
        return att_matrix

    def forward_hypergraph(self, node_embeds, hypergraph_adj, chunk_size=512):
        hyperedge_embeds = hypergraph_adj.T @ node_embeds
        num_nodes, num_hyperedges = hypergraph_adj.shape

        node_hyperedge_weights = torch.zeros((num_nodes, num_hyperedges),
                                             device=node_embeds.device,
                                             dtype=node_embeds.dtype)

        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            node_chunk = node_embeds[start:end]

            scores = torch.matmul(node_chunk, hyperedge_embeds.T)
            scores_flat = scores.reshape(-1, 1)
            att_logits = self.score_mlp(scores_flat)
            att_logits = att_logits.view(end - start, num_hyperedges, 2)
            att_probs = F.softmax(att_logits, dim=-1)
            weights_chunk = att_probs[:, :, 1]

            node_hyperedge_weights[start:end] = weights_chunk

        return node_hyperedge_weights


class GCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adj, embeds):
        try:
            return l2_norm(t.spmm(adj, embeds))
        except Exception:
            return l2_norm(adj @ embeds)


class HGNNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adj, embeds, node_hyperedge_weights=None):
        if node_hyperedge_weights is not None:
            adj_weighted = adj * node_hyperedge_weights
        else:
            adj_weighted = adj

        lat = adj_weighted.T @ embeds
        ret = adj_weighted @ lat
        return l2_norm(ret)


class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adj, keepRate, edge_weights=None):
        if keepRate == 1.0:
            if edge_weights is None:
                return adj
            return t.sparse_coo_tensor(adj._indices(), edge_weights, adj.shape)
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size(0)
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        if edge_weights is None:
            newVals = vals[mask] / keepRate
        else:
            newVals = edge_weights[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse_coo_tensor(newIdxs, newVals, adj.shape)


class ClassifierLayer(nn.Module):
    def __init__(self, dim=None):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(args.latdim * 2, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.cat((dEmbeds, gEmbeds), dim=1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        return self.lin2(embeds)


class Model(nn.Module):
    def __init__(self,
                 drug_encoder_cfg: Optional[dict] = None,
                 omics_encoder_cfg: Optional[dict] = None, ):
        super().__init__()

        self.drug_encoder = None
        self.drug_proj = None
        if drug_encoder_cfg is not None:
            self.drug_encoder = DrugEncoder(
                d_atom=drug_encoder_cfg['d_atom'],
                d_model=drug_encoder_cfg['d_model'],
                dropout=drug_encoder_cfg.get('dropout', 0.1),
                num_total_atoms=drug_encoder_cfg.get('num_total_atoms')
            )
            if drug_encoder_cfg['d_model'] != args.latdim:
                self.drug_proj = nn.Linear(drug_encoder_cfg['d_model'], args.latdim)

        self.omics_encoder = None
        self.omics_proj = None
        # DrugBank has no measured gene features.  Match the original DGCL
        # implementation by learning one unconstrained embedding per gene ID
        # instead of encoding the compatibility-only random feature matrix.
        self.use_cell_id_embedding = str(args.dataset).lower() == "drugbank"
        if self.use_cell_id_embedding:
            self.cell_embedding = nn.Parameter(
                init(torch.empty(args.cell, args.latdim))
            )
        elif omics_encoder_cfg is not None:
            omics_output_dim = omics_encoder_cfg['hidden_dim']
            self.omics_encoder = GraphCDROmicsEncoder(
                omics_dims=omics_encoder_cfg['omics_dims'],
                output_dim=omics_output_dim,
                num_cells=omics_encoder_cfg.get('num_cells')
            )
            if omics_output_dim != args.latdim:
                self.omics_proj = nn.Linear(omics_output_dim, args.latdim)

        self.gcnLayer = GCNLayer()
        self.hgnnLayer = HGNNLayer()
        self.classifierLayer = ClassifierLayer(args.latdim)

        self.attention = AttentionModule(args.latdim)

        if args.dense:
            self.dHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
            self.cHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
        self.edgeDropper = SpAdjDropEdge()

        self.pair_proj = nn.Sequential(
            nn.Linear(args.latdim * 2, args.latdim),
            nn.ReLU(),
            nn.Linear(args.latdim, args.latdim),
        )

        self.proto_dim = args.latdim
        self.proto_t = args.proto_t

        self.register_buffer("prototype", torch.empty(args.num_classes, self.proto_dim))
        nn.init.orthogonal_(self.prototype)
        self.prototype = F.normalize(self.prototype, dim=1)

    def encode_all_drugs(self, drug_batch):
        device = next(self.parameters()).device
        drug_batch = drug_batch.to(device)
        embeddings = self.drug_encoder(drug_batch)
        if self.drug_proj is not None:
            embeddings = self.drug_proj(embeddings)
        return embeddings

    def encode_all_cells(self, omics_inputs: List[torch.Tensor]):
        if self.use_cell_id_embedding:
            return self.cell_embedding
        out = self.omics_encoder(omics_inputs)
        if self.omics_proj is not None:
            out = self.omics_proj(out)
        return out

    def build_hyper_incidence(self, node_embeds, hyper_param=None):
        if args.dense:
            scores = node_embeds @ hyper_param
        else:
            scores = node_embeds * args.mult

        return scores, scores

    def forward(self, adj, keepRate, drug_batch: Optional[Dict] = None,
                omics_inputs: Optional[List[torch.Tensor]] = None):
        dEmbeds_all = self.encode_all_drugs(drug_batch)
        cEmbeds_all = self.encode_all_cells(omics_inputs)
        embeds = torch.cat([dEmbeds_all, cEmbeds_all], dim=0)
        embedsLst = [embeds]
        gcnEmbedsLst = [embeds]
        hyperEmbedsLst = [embeds]
        ddHyper, _ = self.build_hyper_incidence(
            dEmbeds_all,
            self.dHyper if args.dense else None
        )
        ccHyper, _ = self.build_hyper_incidence(
            cEmbeds_all,
            self.cHyper if args.dense else None
        )

        for i in range(args.gnn_layer):
            edge_w = self.attention(embedsLst[-1], adj)
            row, col = adj._indices()
            edge_vals = edge_w[row, col]

            adj_dropped = self.edgeDropper(adj, keepRate, edge_weights=edge_vals)

            gcnEmbeds = self.gcnLayer(adj_dropped, embedsLst[-1])

            d_node_hyper_w = self.attention.forward_hypergraph(
                embedsLst[-1][:args.drug],
                ddHyper
            )
            hyperDEmbeds = self.hgnnLayer(
                ddHyper,
                embedsLst[-1][:args.drug],
                node_hyperedge_weights=d_node_hyper_w
            )

            c_node_hyper_w = self.attention.forward_hypergraph(
                embedsLst[-1][args.drug:],
                ccHyper
            )
            hyperCEmbeds = self.hgnnLayer(
                ccHyper,
                embedsLst[-1][args.drug:],
                node_hyperedge_weights=c_node_hyper_w
            )

            hyperEmbeds = torch.cat([hyperDEmbeds, hyperCEmbeds], axis=0)
            gcnEmbedsLst.append(gcnEmbeds)
            hyperEmbedsLst.append(hyperEmbeds)
            embedsLst.append(gcnEmbeds + hyperEmbeds)
        embeds = sum(embedsLst)
        return embeds, gcnEmbedsLst, hyperEmbedsLst

    @torch.no_grad()
    def get_hyper_incidence_matrices(self, drug_batch: Optional[Dict] = None,
                                     omics_inputs: Optional[List[torch.Tensor]] = None,
                                     effective: bool = True):
        was_training = self.training
        self.eval()

        dEmbeds_all = self.encode_all_drugs(drug_batch)
        cEmbeds_all = self.encode_all_cells(omics_inputs)

        drug_h, drug_scores = self.build_hyper_incidence(
            dEmbeds_all,
            self.dHyper if args.dense else None
        )
        cell_h, cell_scores = self.build_hyper_incidence(
            cEmbeds_all,
            self.cHyper if args.dense else None
        )

        result = {
            "drug_raw": drug_h.detach().cpu(),
            "cell_raw": cell_h.detach().cpu(),
            "drug_score": drug_scores.detach().cpu(),
            "cell_score": cell_scores.detach().cpu(),
        }

        if effective:
            drug_w = self.attention.forward_hypergraph(dEmbeds_all, drug_h)
            cell_w = self.attention.forward_hypergraph(cEmbeds_all, cell_h)
            result["drug_effective"] = (drug_h * drug_w).detach().cpu()
            result["cell_effective"] = (cell_h * cell_w).detach().cpu()
            result["drug_attention"] = drug_w.detach().cpu()
            result["cell_attention"] = cell_w.detach().cpu()

        if was_training:
            self.train()
        return result

    def softmax_with_temperature(self, x, t=1.0, dim=-1):
        return torch.softmax(x / t, dim=dim)

    def class_split(self, labels: torch.Tensor, feats: torch.Tensor):
        class_feats = {}
        y = labels.view(-1).detach()

        for i in range(args.num_classes):
            idx = (y == i).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            class_feats[i] = feats[idx]
        return class_feats

    @torch.no_grad()
    def prototype_update(self, class_feats):

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for cls, feats in class_feats.items():
            feats = F.normalize(feats, dim=1)
            proto = self.prototype[cls].unsqueeze(0)
            cosine = cos(proto, feats)
            weights = torch.softmax(cosine / 5.0, dim=0).unsqueeze(0)
            new_proto = torch.mm(weights, feats).squeeze(0)
            new_proto = F.normalize(new_proto, dim=0)
            self.prototype[cls].copy_(new_proto)

    def global_ssl(self, class_feats, temp=None):
        logits_all = []
        for cls, feats in class_feats.items():
            prototype_ordered = torch.cat([
                self.prototype[cls:cls + 1].detach(),
                self.prototype[:cls].detach(),
                self.prototype[cls + 1:].detach()
            ], dim=0)

            logits_cls = torch.einsum(
                "nd,cd->nc",
                F.normalize(feats, dim=1),
                F.normalize(prototype_ordered, dim=1)
            ) / temp

            logits_all.append(logits_cls)

        logits = torch.cat(logits_all, dim=0)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def local_ssl(self, class_feats):
        losses = []
        max_class_size = max(v.size(0) for v in class_feats.values())
        margin = 0.2

        for cls, feats in class_feats.items():
            pos_proto = self.prototype[cls]
            neg_proto = self.prototype[1 - cls]

            pos_sim = F.cosine_similarity(feats, pos_proto.unsqueeze(0), dim=1)
            neg_sim = F.cosine_similarity(feats, neg_proto.unsqueeze(0), dim=1)

            loss = F.relu(margin + neg_sim - pos_sim)

            weight = max_class_size / feats.size(0)
            losses.append(weight * loss.mean())

        return sum(losses) / len(losses)

    def calcLosses(self, drugs, cells, labels, adj, keepRate,
                   drug_batch: Optional[Dict] = None,
                   omics_inputs: Optional[List[torch.Tensor]] = None):

        embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(
            adj, keepRate, drug_batch=drug_batch, omics_inputs=omics_inputs
        )

        num_drugs = args.drug
        dEmbeds_all = embeds[:num_drugs]
        cEmbeds_all = embeds[num_drugs:]

        dEmbeds = dEmbeds_all[drugs]
        cEmbeds = cEmbeds_all[cells]
        pre = self.classifierLayer(dEmbeds, cEmbeds)
        ceLoss = ce(pre, labels)

        sslLoss = 0.0
        for i in range(1, args.gnn_layer + 1):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:args.drug], embeds2[:args.drug],
                                    t.unique(drugs), args.temp)
            sslLoss += contrastLoss(embeds1[args.drug:], embeds2[args.drug:],
                                    t.unique(cells), args.temp)

        pair_features = torch.cat([dEmbeds, cEmbeds], dim=1)
        pair_z = self.pair_proj(pair_features)

        class_feats = self.class_split(labels, pair_z)
        self.prototype_update(class_feats)
        global_loss = self.global_ssl(class_feats, temp=self.proto_t)
        local_loss = self.local_ssl(class_feats)

        return ceLoss, sslLoss, global_loss, local_loss

    def predict(self, adj, drugs, cells, drug_batch: Optional[Dict] = None,
                omics_inputs: Optional[List[torch.Tensor]] = None
                ):
        embeds, _, _ = self.forward(adj, 1.0, drug_batch=drug_batch, omics_inputs=omics_inputs)
        num_drugs = args.drug
        dEmbeds_all = embeds[:num_drugs]
        cEmbeds_all = embeds[num_drugs:]
        dEmbeds = dEmbeds_all[drugs]
        cEmbeds = cEmbeds_all[cells]
        pre = self.classifierLayer(dEmbeds, cEmbeds)
        return pre
