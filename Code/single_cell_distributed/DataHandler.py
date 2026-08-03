"""Partitioned data loading for the large SingleCell cohort.

The regular :mod:`DataHandler` keeps complete datasets on one device.  This
adapter instead gives every rank a disjoint drug/cell subset, retains
cross-partition interactions as supervised edges, and reads only the H5AD rows
owned by the current rank.
"""

import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse

from Code.training.DatasetConfig import get_dataset_config, resolve_data_dir
from Code.training.Params import args


class DistributedSingleCellDataHandler:
    def __init__(self, rank, world_size, device):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = torch.device(device)

        canonical, config = get_dataset_config(args.dataset)
        if canonical != "SingleCell":
            raise ValueError(
                "DistributedSingleCellDataHandler only supports --dataset SingleCell."
            )
        self.dataset_name = canonical
        self.config = config
        self.data_dir = resolve_data_dir(args.data_path, canonical)

        (
            self.single_cell_group,
            self.single_cell_subset,
            self.triplet_path,
            self.expression_path,
        ) = self._resolve_subset_paths()
        interactions = pd.read_csv(self.triplet_path)
        interactions = self._normalize_interactions(interactions)

        self.all_drug_names = sorted(interactions["drug_id"].unique().tolist())
        self.all_cell_names = sorted(interactions["cell_id"].unique().tolist())
        self.total_drugs = len(self.all_drug_names)
        self.total_cells = len(self.all_cell_names)
        if self.world_size > self.total_cells:
            raise ValueError(
                f"Requested {self.world_size} GPUs for only {self.total_cells} cells."
            )

        self.drug_name2id = {
            name: idx for idx, name in enumerate(self.all_drug_names)
        }
        self.cell_name2id = {
            name: idx for idx, name in enumerate(self.all_cell_names)
        }
        args.drug = self.total_drugs
        args.cell = self.total_cells

        # One deterministic global partition is independently reconstructed by
        # every process. np.array_split guarantees a non-empty, unique drug
        # subset whenever world_size <= total_drugs.
        rng = np.random.default_rng(args.seed)
        drug_permutation = rng.permutation(self.total_drugs)
        self.drug_partitions = [
            np.sort(chunk.astype(np.int64, copy=False))
            for chunk in np.array_split(drug_permutation, self.world_size)
        ]
        self.cell_partitions = [
            chunk.astype(np.int64, copy=False)
            for chunk in np.array_split(np.arange(self.total_cells), self.world_size)
        ]
        self.my_drug_gids = self.drug_partitions[self.rank]
        self.my_cell_gids = self.cell_partitions[self.rank]

        self.drug_owners = np.empty(self.total_drugs, dtype=np.int64)
        self.cell_owners = np.empty(self.total_cells, dtype=np.int64)
        for owner, gids in enumerate(self.drug_partitions):
            self.drug_owners[gids] = owner
        for owner, gids in enumerate(self.cell_partitions):
            self.cell_owners[gids] = owner

        self.global_to_local_d = np.full(self.total_drugs, -1, dtype=np.int64)
        self.global_to_local_c = np.full(self.total_cells, -1, dtype=np.int64)
        self.global_to_local_d[self.my_drug_gids] = np.arange(
            len(self.my_drug_gids), dtype=np.int64
        )
        self.global_to_local_c[self.my_cell_gids] = np.arange(
            len(self.my_cell_gids), dtype=np.int64
        )

        stratify = interactions["label"]
        self.df_train, self.df_test = train_test_split(
            interactions,
            test_size=args.test_size,
            random_state=args.seed,
            shuffle=True,
            stratify=stratify,
        )
        self.df_train = self.df_train.reset_index(drop=True)
        self.df_test = self.df_test.reset_index(drop=True)
        self.eval_split_name = (
            f"generated stratified {1 - args.test_size:.0%}/{args.test_size:.0%} split"
        )

    def _resolve_subset_paths(self):
        """Resolve an exact paired CSV/H5AD cohort without hidden path rules."""
        group = str(args.single_cell_group).strip()
        requested = args.single_cell_subset
        if group == "ALL":
            if requested and str(requested).strip().lower() != "all":
                raise ValueError(
                    "--single_cell_group ALL only accepts --single_cell_subset ALL."
                )
            subset = "ALL"
        else:
            folder = Path(self.data_dir) / group
            suffix = "_merged_triplets.csv"
            available = sorted(
                path.name[: -len(suffix)] for path in folder.glob(f"*{suffix}")
            )
            if not requested:
                raise ValueError(
                    f"--single_cell_group {group} requires --single_cell_subset. "
                    f"Available values: {available}"
                )
            lookup = {name.casefold(): name for name in available}
            key = str(requested).strip().casefold()
            if key not in lookup:
                raise ValueError(
                    f"Unknown {group} subset {requested!r}. Available values: {available}"
                )
            subset = lookup[key]

        folder = Path(self.data_dir) / group
        triplet_path = folder / f"{subset}_merged_triplets.csv"
        expression_path = folder / f"{subset}_normalized_expression.h5ad"
        missing = [str(path) for path in (triplet_path, expression_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"SingleCell subset files are missing: {missing}")
        return group, subset, str(triplet_path), str(expression_path)

    @staticmethod
    def _normalize_interactions(frame):
        rename = {}
        if "drug_id" not in frame.columns:
            for candidate in ("drug_name", "Drug", "drug"):
                if candidate in frame.columns:
                    rename[candidate] = "drug_id"
                    break
        if "cell_id" not in frame.columns:
            for candidate in ("cell", "Cell"):
                if candidate in frame.columns:
                    rename[candidate] = "cell_id"
                    break
        frame = frame.rename(columns=rename)
        required = {"drug_id", "cell_id", "label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"SingleCell triplets are missing columns: {sorted(missing)}")

        frame = frame[["drug_id", "cell_id", "label"]].copy()
        frame["drug_id"] = frame["drug_id"].astype(str).str.strip()
        frame["cell_id"] = frame["cell_id"].astype(str).str.strip()
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
        frame = frame.dropna(subset=["drug_id", "cell_id", "label"])
        frame["label"] = frame["label"].astype(np.int64)
        unexpected = sorted(set(frame["label"]) - {0, 1})
        if unexpected:
            raise ValueError(f"SingleCell labels must be binary; found {unexpected}.")
        return frame

    @staticmethod
    def _graph_arrays(graph, drug_name):
        if isinstance(graph, dict):
            try:
                features = graph["node_features"]
                adjacency = graph["adjacency_matrix"]
            except KeyError as error:
                raise ValueError(
                    f"Drug graph {drug_name!r} lacks node_features/adjacency_matrix."
                ) from error
        elif isinstance(graph, (tuple, list)) and len(graph) >= 2:
            features, adjacency = graph[:2]
        else:
            raise ValueError(f"Unsupported drug graph format for {drug_name!r}.")
        features = np.asarray(features, dtype=np.float32)
        adjacency = np.asarray(adjacency, dtype=np.float32)
        if features.ndim != 2 or adjacency.shape != (
            features.shape[0], features.shape[0]
        ):
            raise ValueError(
                f"Invalid drug graph {drug_name!r}: x={features.shape}, "
                f"adj={adjacency.shape}."
            )
        return features, adjacency

    def _load_local_drugs(self):
        graph_path = os.path.join(self.data_dir, "drug_graph_data.pkl")
        with open(graph_path, "rb") as handle:
            raw_graphs = pickle.load(handle)
        graphs = {str(key).strip(): value for key, value in raw_graphs.items()}

        # Infer the shared input width from a real cohort drug even when this
        # rank owns no drugs. This keeps DDP model parameter shapes identical.
        first_name = self.all_drug_names[0]
        if first_name not in graphs:
            raise KeyError(f"No molecular graph was found for drug {first_name!r}.")
        first_features, _ = self._graph_arrays(graphs[first_name], first_name)
        feature_dim = first_features.shape[1]
        data_list = []
        for gid in self.my_drug_gids:
            drug_name = self.all_drug_names[int(gid)]
            if drug_name not in graphs:
                raise KeyError(f"No molecular graph was found for drug {drug_name!r}.")
            features, adjacency = self._graph_arrays(graphs[drug_name], drug_name)
            if features.shape[1] != feature_dim:
                raise ValueError(
                    f"Drug feature dimensions differ: expected {feature_dim}, "
                    f"got {features.shape[1]} for {drug_name!r}."
                )
            edge_index, _ = dense_to_sparse(torch.from_numpy(adjacency))
            data_list.append(
                Data(x=torch.from_numpy(features), edge_index=edge_index)
            )

        self.local_drug_batch = (
            Batch.from_data_list(data_list).to(self.device) if data_list else None
        )
        self.num_local_drugs = len(data_list)
        self.drug_feature_dim = int(feature_dim)

    def _load_local_expression(self):
        try:
            import anndata as ad
        except ImportError as error:
            raise ImportError(
                "SingleCell training requires anndata and h5py. "
                "Install the packages from requirements.txt."
            ) from error

        adata = ad.read_h5ad(self.expression_path, backed="r")
        try:
            obs_names = np.asarray(adata.obs_names.astype(str))
            if len(set(obs_names)) != len(obs_names):
                raise ValueError("SingleCell H5AD contains duplicate obs_names.")
            obs_lookup = {name: idx for idx, name in enumerate(obs_names)}
            # Three cancer-specific H5AD files add a dataset prefix such as
            # Data18_ to the CSV cell ID. Keep exact names authoritative, and
            # expose a stripped alias only when that alias is unambiguous.
            stripped_lookup = {}
            ambiguous_aliases = set()
            for idx, name in enumerate(obs_names):
                alias = re.sub(r"^Data\d+_", "", name)
                if alias == name:
                    continue
                if alias in stripped_lookup:
                    ambiguous_aliases.add(alias)
                else:
                    stripped_lookup[alias] = idx
            for alias in ambiguous_aliases:
                stripped_lookup.pop(alias, None)

            local_names = [self.all_cell_names[int(gid)] for gid in self.my_cell_gids]
            source_rows = []
            missing = []
            for name in local_names:
                if name in obs_lookup:
                    source_rows.append(obs_lookup[name])
                elif name in stripped_lookup:
                    source_rows.append(stripped_lookup[name])
                else:
                    missing.append(name)
            if missing:
                raise ValueError(
                    f"H5AD is missing {len(missing)} assigned cells; examples: {missing[:5]}"
                )

            source_rows = np.asarray(source_rows, dtype=np.int64)
            sort_order = np.argsort(source_rows)
            sorted_rows = source_rows[sort_order]
            matrix = adata.X[sorted_rows, :]
            if sp.issparse(matrix):
                matrix = matrix.toarray()
            matrix = np.asarray(matrix, dtype=np.float32)
            inverse_order = np.empty_like(sort_order)
            inverse_order[sort_order] = np.arange(len(sort_order))
            matrix = matrix[inverse_order]
            self.expression_dim = int(matrix.shape[1])
            self.local_omics = [torch.from_numpy(matrix).to(self.device)]
            self.num_local_cells = int(matrix.shape[0])
        finally:
            adata.file.close()

    @staticmethod
    def _normalize_adj(matrix):
        rowsum = np.asarray(matrix.sum(1)).reshape(-1)
        inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
        inv_sqrt[rowsum == 0] = 0
        degree = sp.diags(inv_sqrt)
        return degree.dot(matrix).dot(degree).tocoo()

    def _build_local_adj(self, local_drugs, local_cells):
        values = np.ones(len(local_drugs), dtype=np.float32)
        bipartite = sp.coo_matrix(
            (values, (local_drugs, local_cells)),
            shape=(self.num_local_drugs, self.num_local_cells),
        )
        drug_zero = sp.csr_matrix((self.num_local_drugs, self.num_local_drugs))
        cell_zero = sp.csr_matrix((self.num_local_cells, self.num_local_cells))
        matrix = sp.vstack(
            [
                sp.hstack([drug_zero, bipartite]),
                sp.hstack([bipartite.transpose(), cell_zero]),
            ]
        ).tocsr()
        matrix = (matrix != 0).astype(np.float32)
        matrix = matrix + sp.eye(matrix.shape[0], dtype=np.float32)
        matrix = self._normalize_adj(matrix)
        indices = torch.from_numpy(
            np.vstack([matrix.row, matrix.col]).astype(np.int64)
        )
        values = torch.from_numpy(matrix.data.astype(np.float32))
        return torch.sparse_coo_tensor(
            indices, values, torch.Size(matrix.shape)
        ).coalesce()

    def _map_frame(self, frame):
        drug_gids = frame["drug_id"].map(self.drug_name2id).to_numpy(np.int64)
        cell_gids = frame["cell_id"].map(self.cell_name2id).to_numpy(np.int64)
        labels = frame["label"].to_numpy(np.int64)
        return drug_gids, cell_gids, labels

    def _build_training_edges(self):
        drug_gids, cell_gids, labels = self._map_frame(self.df_train)
        drug_owner = self.drug_owners[drug_gids]
        cell_owner = self.cell_owners[cell_gids]

        inner = (drug_owner == self.rank) & (cell_owner == self.rank)
        cross_d = (drug_owner == self.rank) & (cell_owner != self.rank)
        cross_c = (cell_owner == self.rank) & (drug_owner != self.rank)

        local_d = self.global_to_local_d[drug_gids[inner]]
        local_c = self.global_to_local_c[cell_gids[inner]]
        self.local_train_d = torch.from_numpy(local_d).long().to(self.device)
        self.local_train_c = torch.from_numpy(local_c).long().to(self.device)
        self.local_train_y = torch.from_numpy(labels[inner]).long().to(self.device)

        self.cross_edges = {
            "d_local": {
                "u": torch.from_numpy(
                    self.global_to_local_d[drug_gids[cross_d]]
                ).long().to(self.device),
                "v_global": torch.from_numpy(cell_gids[cross_d]).long().to(
                    self.device
                ),
                "y": torch.from_numpy(labels[cross_d]).long().to(self.device),
            },
            "c_local": {
                "v": torch.from_numpy(
                    self.global_to_local_c[cell_gids[cross_c]]
                ).long().to(self.device),
                "u_global": torch.from_numpy(drug_gids[cross_c]).long().to(
                    self.device
                ),
                "y": torch.from_numpy(labels[cross_c]).long().to(self.device),
            },
        }
        self.local_adj = self._build_local_adj(local_d, local_c).to(self.device)

    def _build_test_edges(self):
        drug_gids, cell_gids, labels = self._map_frame(self.df_test)
        self.test_d_gid = torch.from_numpy(drug_gids).long().to(self.device)
        self.test_c_gid = torch.from_numpy(cell_gids).long().to(self.device)
        self.test_y = torch.from_numpy(labels).long().to(self.device)

    def LoadData(self):
        self._load_local_drugs()
        self._load_local_expression()
        self._build_training_edges()
        self._build_test_edges()
        self.local_drug_gids = torch.from_numpy(self.my_drug_gids).long().to(
            self.device
        )
        self.local_cell_gids = torch.from_numpy(self.my_cell_gids).long().to(
            self.device
        )

        print(
            f"[Rank {self.rank}] SingleCell ready: "
            f"subset={self.single_cell_group}/{self.single_cell_subset}, "
            f"drugs={self.num_local_drugs}, cells={self.num_local_cells}, "
            f"inner={len(self.local_train_y)}, "
            f"cross-d={len(self.cross_edges['d_local']['y'])}, "
            f"cross-c={len(self.cross_edges['c_local']['y'])}, "
            f"expression_dim={self.expression_dim}"
        )
        if self.rank == 0:
            print(f"  triplets:  {self.triplet_path}")
            print(f"  expression: {self.expression_path}")
