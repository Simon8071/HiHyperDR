import os
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
import torch
import torch as t
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from Code.training.DatasetConfig import get_dataset_config, resolve_data_dir
from Code.training.Params import args


class DataHandler:
    def __init__(self):
        self.drug_batch = None 
        self.omics_inputs_all = None
        self.torchBiAdj = None
        
        self.train_d = None
        self.train_c = None
        self.train_y = None
        self.test_d = None
        self.test_c = None
        self.test_y = None

    def normalizeAdj(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def getAdj(self, drug_ids, cell_ids):
        mat = sp.coo_matrix((np.ones(len(drug_ids)), (drug_ids, cell_ids)), shape=(args.drug, args.cell))
        a = sp.csr_matrix((args.drug, args.drug))
        b = sp.csr_matrix((args.cell, args.cell))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse_coo_tensor(idxs, vals, shape).to(args.device)


    @staticmethod
    def _clean_ids(values):
        return (
            values.astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    @staticmethod
    def _normalize_labels(df):
        df = df.copy()
        labels = pd.to_numeric(df["label"], errors="coerce").replace(-1, 0)
        valid = labels.notna()
        df = df.loc[valid].copy()
        df["label"] = labels.loc[valid].astype(np.int64)
        unexpected = sorted(set(df["label"]) - {0, 1})
        if unexpected:
            raise ValueError(f"Only binary labels are supported; found {unexpected}.")
        return df

    def _load_gdsc(self, data_dir, eval_filename):
        mapping = pd.read_csv(
            os.path.join(data_dir, "drug_name_pubchem_id.csv"), dtype=str
        )
        mapping["cid"] = self._clean_ids(mapping["cid"])
        cid_to_name = dict(zip(mapping["cid"], mapping["drug_name"].astype(str)))

        def read_interactions(filename):
            frame = pd.read_csv(os.path.join(data_dir, filename), dtype=str)
            frame["CID"] = self._clean_ids(frame["CID"])
            frame["COSMIC_ID"] = self._clean_ids(frame["COSMIC_ID"])
            frame["drug_id"] = frame["CID"].map(cid_to_name)
            frame["entity_id"] = frame["COSMIC_ID"]
            return self._normalize_labels(frame[["drug_id", "entity_id", "label"]])

        eval_filename = eval_filename or "val_balanced.csv"
        train_df = read_interactions("train.csv")
        eval_df = read_interactions(eval_filename)

        omics_frames = []
        for filename in ("cell_expression.csv", "cell_mutation.csv", "cell_copy.csv"):
            frame = pd.read_csv(os.path.join(data_dir, filename), index_col=0)
            frame.index = self._clean_ids(pd.Series(frame.index)).to_numpy()
            omics_frames.append(frame)
        return train_df, eval_df, omics_frames, "cell lines", eval_filename

    def _load_drugbank(self, data_dir, eval_filename):
        columns = ["drug_id", "entity_id", "label"]

        def read_interactions(filename):
            frame = pd.read_csv(
                os.path.join(data_dir, filename),
                header=None,
                names=columns,
                dtype=str,
            )
            frame["drug_id"] = self._clean_ids(frame["drug_id"])
            frame["entity_id"] = self._clean_ids(frame["entity_id"])
            return self._normalize_labels(frame)

        eval_filename = eval_filename or "test.csv"
        train_df = read_interactions("train.csv")
        eval_df = read_interactions(eval_filename)
        return train_df, eval_df, None, "genes", eval_filename

    def _load_expression_dataset(self, data_dir, canonical, config):
        expression = pd.read_csv(
            os.path.join(data_dir, config["expression_file"]), index_col=0
        )
        expression.index = self._clean_ids(pd.Series(expression.index)).to_numpy()

        labels = pd.read_csv(os.path.join(data_dir, config["label_file"]), dtype=str)
        if canonical == "PDTC":
            labels = labels.rename(
                columns={"drug_name": "drug_id", "cell": "entity_id", "AUC": "label"}
            )
        else:
            labels = labels.rename(
                columns={"Drug": "drug_id", "Cell": "entity_id", "Label": "label"}
            )
        labels = labels[["drug_id", "entity_id", "label"]].copy()
        labels["drug_id"] = self._clean_ids(labels["drug_id"])
        labels["entity_id"] = self._clean_ids(labels["entity_id"])
        labels = self._normalize_labels(labels)

        stratify = labels["label"] if labels["label"].value_counts().min() >= 2 else None
        train_df, eval_df = train_test_split(
            labels,
            test_size=args.test_size,
            random_state=args.seed,
            shuffle=True,
            stratify=stratify,
        )
        split_name = f"generated stratified {1 - args.test_size:.0%}/{args.test_size:.0%} split"
        return (
            train_df.reset_index(drop=True),
            eval_df.reset_index(drop=True),
            [expression],
            "samples",
            split_name,
        )

    @staticmethod
    def _as_dense_float32(matrix):
        if sp.issparse(matrix):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    def _load_drug_graphs(self, data_dir):
        graph_path = os.path.join(data_dir, "drug_graph_data.pkl")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Drug graph file not found: {graph_path}")
        with open(graph_path, "rb") as handle:
            raw_graphs = pickle.load(handle)
        graphs = {str(key).strip(): value for key, value in raw_graphs.items()}
        for key, value in graphs.items():
            if not isinstance(value, (tuple, list)) or len(value) < 2:
                raise ValueError(
                    f"Drug graph {key!r} must contain at least feature and adjacency matrices."
                )
        return graphs

    def _build_drug_batch(self, final_drugs, graphs):
        drug_data_list = []
        for drug_id in final_drugs:
            graph = graphs[drug_id]
            x_np = self._as_dense_float32(graph[0])
            adj_np = self._as_dense_float32(graph[1])
            if x_np.ndim != 2 or adj_np.shape != (x_np.shape[0], x_np.shape[0]):
                raise ValueError(
                    f"Invalid graph shapes for {drug_id!r}: x={x_np.shape}, adj={adj_np.shape}."
                )
            x = torch.from_numpy(x_np)
            adj = torch.from_numpy(adj_np)
            edge_index, _ = dense_to_sparse(adj)
            drug_data_list.append(Data(x=x, edge_index=edge_index))
        return Batch.from_data_list(drug_data_list)

    def LoadData(self, eval_filename=None):
        if not hasattr(args, "device"):
            args.device = t.device("cuda" if t.cuda.is_available() else "cpu")

        canonical, config = get_dataset_config(args.dataset)
        data_dir = resolve_data_dir(args.data_path, canonical)
        if config["mode"] == "single_cell_distributed":
            raise ValueError(
                "SingleCell is partitioned across GPUs. Run "
                "`python -m Code.single_cell_distributed.Main --dataset SingleCell --gpus N` "
                "instead of Code.training.Main."
            )
        graphs = self._load_drug_graphs(data_dir)

        if config["mode"] == "gdsc":
            train_df, eval_df, feature_frames, entity_type, split_name = self._load_gdsc(
                data_dir, eval_filename
            )
        elif config["mode"] == "drugbank":
            train_df, eval_df, feature_frames, entity_type, split_name = self._load_drugbank(
                data_dir, eval_filename
            )
        else:
            train_df, eval_df, feature_frames, entity_type, split_name = (
                self._load_expression_dataset(data_dir, canonical, config)
            )

        available_drugs = set(graphs)
        train_df = train_df[train_df["drug_id"].isin(available_drugs)].copy()
        eval_df = eval_df[eval_df["drug_id"].isin(available_drugs)].copy()

        if feature_frames is not None:
            available_entities = set(feature_frames[0].index.astype(str))
            train_df = train_df[train_df["entity_id"].isin(available_entities)].copy()
            eval_df = eval_df[eval_df["entity_id"].isin(available_entities)].copy()

        all_interactions = pd.concat([train_df, eval_df], ignore_index=True)
        final_drugs = sorted(all_interactions["drug_id"].unique().tolist())
        final_cells = sorted(all_interactions["entity_id"].unique().tolist())
        if not final_drugs or not final_cells:
            raise ValueError(f"No usable {canonical} interactions remained after ID matching.")

        drug_map = {drug_id: idx for idx, drug_id in enumerate(final_drugs)}
        cell_map = {entity_id: idx for idx, entity_id in enumerate(final_cells)}
        self.dataset_name = canonical
        self.data_dir = data_dir
        self.entity_type = entity_type
        self.eval_split_name = split_name
        self.final_drugs = final_drugs
        self.drug_map = drug_map
        self.final_cells = final_cells
        self.cell_map = cell_map
        args.drug = len(final_drugs)
        args.cell = len(final_cells)

        self.drug_batch = self._build_drug_batch(final_drugs, graphs)
        self.omics_inputs_all = []
        if canonical == "DrugBank":
            rng = np.random.default_rng(args.seed)
            random_features = rng.standard_normal(
                (len(final_cells), args.random_feature_dim), dtype=np.float32
            )
            self.omics_inputs_all.append(torch.from_numpy(random_features).to(args.device))
        else:
            for frame in feature_frames:
                sorted_frame = frame.reindex(final_cells).fillna(0)
                values = sorted_frame.to_numpy(dtype=np.float32, copy=True)
                self.omics_inputs_all.append(torch.from_numpy(values).to(args.device))

        def map_interactions(frame):
            drug_ids = frame["drug_id"].map(drug_map).to_numpy(dtype=np.int64)
            cell_ids = frame["entity_id"].map(cell_map).to_numpy(dtype=np.int64)
            labels = frame["label"].to_numpy(dtype=np.int64)
            return drug_ids, cell_ids, labels

        train_d, train_c, train_y = map_interactions(train_df)
        test_d, test_c, test_y = map_interactions(eval_df)
        self.train_d = torch.from_numpy(train_d).long().to(args.device)
        self.train_c = torch.from_numpy(train_c).long().to(args.device)
        self.train_y = torch.from_numpy(train_y).long().to(args.device)
        self.test_d = torch.from_numpy(test_d).long().to(args.device)
        self.test_c = torch.from_numpy(test_c).long().to(args.device)
        self.test_y = torch.from_numpy(test_y).long().to(args.device)
        self.torchBiAdj = self.getAdj(train_d, train_c).to(args.device)

        print(
            f"Dataset: {canonical} ({data_dir})\n"
            f"Data loaded: {len(final_drugs)} drugs, {len(final_cells)} {entity_type}.\n"
            f"Train samples: {len(self.train_y)}, Test samples: {len(self.test_y)} "
            f"[{split_name}]"
        )
