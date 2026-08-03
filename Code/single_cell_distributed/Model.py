"""DDP-safe model wrapper for partitioned SingleCell training."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import softmax as sparse_softmax

from Code.training.Model_sparse import Model
from Code.training.Params import args
from Code.training.Utils.Utils import ce


class DistributedSingleCellModel(Model):
    """Run the complete distributed training computation inside DDP.forward."""

    def _sparse_edge_attention(self, embeddings, adjacency):
        indices = adjacency._indices()
        row, col = indices[0], indices[1]
        edge_features = torch.cat([embeddings[row], embeddings[col]], dim=-1)
        scores = self.attention.edge_score(edge_features).squeeze(-1)
        return sparse_softmax(scores, row, num_nodes=embeddings.shape[0])

    def _forward_local(self, handler, keep_rate):
        cell_embeddings = self.encode_all_cells(handler.local_omics)
        if handler.num_local_drugs:
            drug_embeddings = self.encode_all_drugs(handler.local_drug_batch)
        else:
            # Empty drug shards are expected for one-drug cancer/tissue
            # subsets. Avoid sending an empty PyG Batch through GIN/BatchNorm.
            drug_embeddings = cell_embeddings.new_empty((0, args.latdim))
        num_local_drugs = drug_embeddings.shape[0]

        embeddings = torch.cat([drug_embeddings, cell_embeddings], dim=0)
        embeddings_list = [embeddings]
        gcn_embeddings_list = [embeddings]
        hyper_embeddings_list = [embeddings]

        drug_hyper, _ = self.build_hyper_incidence(
            drug_embeddings, self.dHyper if args.dense else None
        )
        cell_hyper, _ = self.build_hyper_incidence(
            cell_embeddings, self.cHyper if args.dense else None
        )

        for _ in range(args.gnn_layer):
            edge_values = self._sparse_edge_attention(
                embeddings_list[-1], handler.local_adj
            )
            dropped_adj = self.edgeDropper(
                handler.local_adj, keep_rate, edge_weights=edge_values
            )
            gcn_embeddings = self.gcnLayer(dropped_adj, embeddings_list[-1])

            if num_local_drugs:
                drug_node_weights = self.attention.forward_hypergraph(
                    embeddings_list[-1][:num_local_drugs], drug_hyper
                )
                hyper_drug_embeddings = self.hgnnLayer(
                    drug_hyper,
                    embeddings_list[-1][:num_local_drugs],
                    node_hyperedge_weights=drug_node_weights,
                )
            else:
                hyper_drug_embeddings = drug_embeddings
            cell_node_weights = self.attention.forward_hypergraph(
                embeddings_list[-1][num_local_drugs:], cell_hyper
            )
            hyper_cell_embeddings = self.hgnnLayer(
                cell_hyper,
                embeddings_list[-1][num_local_drugs:],
                node_hyperedge_weights=cell_node_weights,
            )
            hyper_embeddings = torch.cat(
                [hyper_drug_embeddings, hyper_cell_embeddings], dim=0
            )
            gcn_embeddings_list.append(gcn_embeddings)
            hyper_embeddings_list.append(hyper_embeddings)
            embeddings_list.append(gcn_embeddings + hyper_embeddings)

        final_embeddings = sum(embeddings_list)
        return (
            final_embeddings[:num_local_drugs],
            final_embeddings[num_local_drugs:],
            gcn_embeddings_list,
            hyper_embeddings_list,
        )

    @staticmethod
    def _gather_embeddings_by_global_id(local_embeddings, local_global_ids, total):
        """Gather variable-sized shards and restore the original global ID order."""
        if not dist.is_available() or not dist.is_initialized():
            if local_global_ids.numel() != total:
                raise RuntimeError(
                    f"Single-process shard has {local_global_ids.numel()} of {total} nodes."
                )
            global_embeddings = torch.zeros(
                total,
                local_embeddings.shape[1],
                dtype=local_embeddings.dtype,
                device=local_embeddings.device,
            )
            global_embeddings.index_copy_(
                0, local_global_ids, local_embeddings.detach()
            )
            return global_embeddings

        world_size = dist.get_world_size()
        count = torch.tensor(
            [local_embeddings.shape[0]],
            dtype=torch.long,
            device=local_embeddings.device,
        )
        gathered_counts = [torch.zeros_like(count) for _ in range(world_size)]
        dist.all_gather(gathered_counts, count)
        counts = [int(item.item()) for item in gathered_counts]
        max_count = max(counts)

        padded_embeddings = torch.zeros(
            max_count,
            local_embeddings.shape[1],
            dtype=local_embeddings.dtype,
            device=local_embeddings.device,
        )
        padded_ids = torch.full(
            (max_count,), -1, dtype=torch.long, device=local_embeddings.device
        )
        padded_embeddings[: count.item()] = local_embeddings.detach()
        padded_ids[: count.item()] = local_global_ids

        gathered_embeddings = [
            torch.zeros_like(padded_embeddings) for _ in range(world_size)
        ]
        gathered_ids = [torch.zeros_like(padded_ids) for _ in range(world_size)]
        dist.all_gather(gathered_embeddings, padded_embeddings)
        dist.all_gather(gathered_ids, padded_ids)

        valid_embeddings = []
        valid_ids = []
        for rank, valid_count in enumerate(counts):
            valid_embeddings.append(gathered_embeddings[rank][:valid_count])
            valid_ids.append(gathered_ids[rank][:valid_count])
        all_embeddings = torch.cat(valid_embeddings, dim=0)
        all_ids = torch.cat(valid_ids, dim=0)

        if all_ids.numel() != total or torch.unique(all_ids).numel() != total:
            raise RuntimeError(
                "Distributed node partitions are incomplete or contain duplicate global IDs."
            )
        if int(all_ids.min().item()) != 0 or int(all_ids.max().item()) != total - 1:
            raise RuntimeError("Gathered global node IDs are outside the expected range.")

        global_embeddings = torch.zeros(
            total,
            local_embeddings.shape[1],
            dtype=local_embeddings.dtype,
            device=local_embeddings.device,
        )
        global_embeddings.index_copy_(0, all_ids, all_embeddings)
        return global_embeddings

    def _global_embeddings(self, handler, local_drugs, local_cells):
        global_drugs = self._gather_embeddings_by_global_id(
            local_drugs, handler.local_drug_gids, handler.total_drugs
        )
        global_cells = self._gather_embeddings_by_global_id(
            local_cells, handler.local_cell_gids, handler.total_cells
        )
        return global_drugs, global_cells

    @staticmethod
    def _chunked_contrast_loss(embeddings_1, embeddings_2, temperature, chunk=2048):
        node_count = embeddings_1.shape[0]
        if node_count == 0:
            return embeddings_2.sum() * 0.0
        embeddings_1 = F.normalize(embeddings_1, p=2, dim=1)
        embeddings_2 = F.normalize(embeddings_2, p=2, dim=1)
        loss = embeddings_2.sum() * 0.0
        for start in range(0, node_count, chunk):
            end = min(start + chunk, node_count)
            logits = embeddings_1[start:end] @ embeddings_2.T / temperature
            labels = torch.arange(start, end, device=embeddings_1.device)
            loss = loss + F.cross_entropy(logits, labels, reduction="sum")
        return loss / node_count

    def _distributed_losses(
        self,
        handler,
        local_drugs,
        local_cells,
        gcn_embeddings_list,
        hyper_embeddings_list,
    ):
        zero = local_drugs.sum() * 0.0 + local_cells.sum() * 0.0

        if handler.local_train_y.numel() > 0:
            train_drugs = local_drugs[handler.local_train_d]
            train_cells = local_cells[handler.local_train_c]
            local_logits = self.classifierLayer(train_drugs, train_cells)
            local_ce = ce(local_logits, handler.local_train_y)
        else:
            train_drugs = local_drugs[:0]
            train_cells = local_cells[:0]
            local_ce = zero

        ssl_loss = zero
        if args.ssl_reg != 0:
            for layer in range(1, args.gnn_layer + 1):
                graph_view = gcn_embeddings_list[layer].detach()
                hyper_view = hyper_embeddings_list[layer]
                split = handler.num_local_drugs
                ssl_loss = ssl_loss + self._chunked_contrast_loss(
                    graph_view[:split], hyper_view[:split], args.temp
                )
                ssl_loss = ssl_loss + self._chunked_contrast_loss(
                    graph_view[split:], hyper_view[split:], args.temp
                )

        global_loss = zero
        local_loss = zero
        if (
            handler.local_train_y.numel() > 0
            and (args.global_cl_reg != 0 or args.local_cl_reg != 0)
        ):
            pair_features = torch.cat([train_drugs, train_cells], dim=1)
            pair_projection = self.pair_proj(pair_features)
            class_features = self.class_split(
                handler.local_train_y, pair_projection
            )
            if class_features:
                self.prototype_update(class_features)
                if args.global_cl_reg != 0:
                    global_loss = self.global_ssl(
                        class_features, temp=self.proto_t
                    )
                if args.local_cl_reg != 0:
                    local_loss = self.local_ssl(class_features)

        global_drugs, global_cells = self._global_embeddings(
            handler, local_drugs, local_cells
        )
        cross_ce = zero
        cross_drug = handler.cross_edges["d_local"]
        if cross_drug["y"].numel() > 0:
            logits = self.classifierLayer(
                local_drugs[cross_drug["u"]],
                global_cells[cross_drug["v_global"]],
            )
            cross_ce = cross_ce + ce(logits, cross_drug["y"])

        cross_cell = handler.cross_edges["c_local"]
        if cross_cell["y"].numel() > 0:
            logits = self.classifierLayer(
                global_drugs[cross_cell["u_global"]],
                local_cells[cross_cell["v"]],
            )
            cross_ce = cross_ce + ce(logits, cross_cell["y"])

        total_ce = local_ce + 0.5 * cross_ce
        weighted_ssl = args.ssl_reg * ssl_loss
        weighted_global = args.global_cl_reg * global_loss
        weighted_local = args.local_cl_reg * local_loss
        total_loss = total_ce + weighted_ssl + weighted_global + weighted_local
        return (
            total_loss,
            total_ce.detach(),
            weighted_ssl.detach(),
            weighted_global.detach(),
            weighted_local.detach(),
        )

    def forward(self, handler, keep_rate=1.0, mode="train"):
        """DDP entry point; never call ``ddp_model.module.forward`` for training."""
        local_drugs, local_cells, graph_views, hyper_views = self._forward_local(
            handler, keep_rate
        )
        if mode == "encode_global":
            return self._global_embeddings(handler, local_drugs, local_cells)
        if mode != "train":
            raise ValueError(f"Unsupported distributed forward mode: {mode!r}")
        return self._distributed_losses(
            handler,
            local_drugs,
            local_cells,
            graph_views,
            hyper_views,
        )
