"""Single-node, multi-GPU training entry point for the SingleCell cohort.

Run from the repository root, for example::

    python -m Code.single_cell_distributed.Main --dataset SingleCell --gpus 4
    python -m Code.single_cell_distributed.Main --dataset SingleCell --gpus 4 \
        --single_cell_group cancer --single_cell_subset "Breast cancer"
"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.parallel import DistributedDataParallel as DDP

import Code.training.Utils.TimeLogger as logger
from Code.training.DatasetConfig import canonical_dataset_name, get_checkpoint_dir
from Code.single_cell_distributed.DataHandler import DistributedSingleCellDataHandler
from Code.single_cell_distributed.Model import DistributedSingleCellModel
from Code.training.Params import CODE_ROOT, args
from Code.training.Utils.TimeLogger import log


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_process(rank, world_size):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    args.device = torch.device(f"cuda:{rank}")


class DistributedCoach:
    def __init__(self, handler, rank):
        self.handler = handler
        self.rank = rank

        # Do not create per-atom or per-cell mask parameters here: their shapes
        # depend on the local shard and DDP requires identical parameter shapes
        # on every rank.
        drug_config = {
            "d_atom": handler.drug_feature_dim,
            "d_model": args.latdim,
            "dropout": 0.1,
        }
        omics_config = {
            "omics_dims": [handler.expression_dim],
            "hidden_dim": 128,
            "proj_dim": args.latdim,
        }
        base_model = DistributedSingleCellModel(
            drug_encoder_cfg=drug_config,
            omics_encoder_cfg=omics_config,
        ).to(args.device)
        self.model = DDP(
            base_model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.reg
        )
        self.best_auc = float("-inf")

    @staticmethod
    def _format_metrics(prefix, epoch, metrics):
        values = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        return f"Epoch {epoch}/{args.epoch}, {prefix}: {values}"

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, ce_loss, ssl_loss, global_loss, local_loss = self.model(
            self.handler, keep_rate=args.keepRate, mode="train"
        )
        total.backward()
        self.optimizer.step()
        return {
            "Loss": float(total.detach().item()),
            "CE": float(ce_loss.item()),
            "SSL": float(ssl_loss.item()),
            "Global": float(global_loss.item()),
            "Local": float(local_loss.item()),
        }

    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()
        global_drugs, global_cells = self.model(
            self.handler, keep_rate=1.0, mode="encode_global"
        )

        metrics = None
        if self.rank == 0:
            logits = self.model.module.classifierLayer(
                global_drugs[self.handler.test_d_gid],
                global_cells[self.handler.test_c_gid],
            )
            probabilities = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = self.handler.test_y.cpu().numpy()
            predictions = (probabilities > args.threshold).astype(np.int64)
            metrics = {
                "Acc": accuracy_score(labels, predictions),
                "precision": precision_score(
                    labels, predictions, zero_division=0
                ),
                "recall": recall_score(labels, predictions, zero_division=0),
                "AUPR": average_precision_score(labels, probabilities),
                "AUC": roc_auc_score(labels, probabilities),
                "F1": f1_score(labels, predictions, zero_division=0),
            }
            print("Confusion matrix (TN FP / FN TP):")
            print(confusion_matrix(labels, predictions))

        dist.barrier()
        return metrics

    def save_model(self, label):
        if self.rank != 0:
            return
        checkpoint_dir = get_checkpoint_dir(
            CODE_ROOT, "SingleCell", args.checkpoint_dir
        )
        if not args.checkpoint_dir:
            if self.handler.single_cell_group == "ALL":
                checkpoint_dir = os.path.join(checkpoint_dir, "ALL")
            else:
                checkpoint_dir = os.path.join(
                    checkpoint_dir,
                    self.handler.single_cell_group,
                    self.handler.single_cell_subset,
                )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{label}_model.pkl")
        torch.save(self.model.module.state_dict(), checkpoint_path)

        metadata = {
            "dataset": "SingleCell",
            "single_cell_group": self.handler.single_cell_group,
            "single_cell_subset": self.handler.single_cell_subset,
            "triplet_file": self.handler.triplet_path,
            "expression_file": self.handler.expression_path,
            "world_size": dist.get_world_size(),
            "seed": args.seed,
            "test_size": args.test_size,
            "lr": args.lr,
            "epoch": args.epoch,
            "latdim": args.latdim,
            "hyperNum": args.hyperNum,
            "gnn_layer": args.gnn_layer,
            "keepRate": args.keepRate,
            "temp": args.temp,
            "ssl_reg": args.ssl_reg,
            "global_cl_reg": args.global_cl_reg,
            "local_cl_reg": args.local_cl_reg,
            "proto_t": args.proto_t,
            "reg": args.reg,
            "drugs": self.handler.total_drugs,
            "cells": self.handler.total_cells,
            "expression_dim": self.handler.expression_dim,
        }
        with open(
            os.path.join(checkpoint_dir, f"{label}_params.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        log(f"Saved distributed checkpoint: {checkpoint_path}")

    def run(self):
        start = time.time()
        for epoch in range(args.epoch):
            train_metrics = self.train_epoch()
            if self.rank == 0:
                log(self._format_metrics("Train", epoch, train_metrics))

            if epoch % args.tstEpoch == 0:
                test_metrics = self.test_epoch()
                if self.rank == 0:
                    log(self._format_metrics("Test", epoch, test_metrics))
                    if test_metrics["AUC"] > self.best_auc:
                        self.best_auc = test_metrics["AUC"]
                        self.save_model("best_auc")
                        log(
                            f"New best SingleCell AUC {self.best_auc:.4f} "
                            f"at epoch {epoch}."
                        )

        final_metrics = self.test_epoch()
        if self.rank == 0:
            if final_metrics["AUC"] > self.best_auc:
                self.best_auc = final_metrics["AUC"]
                self.save_model("best_auc")
            self.save_model(time.strftime("%Y%m%d-%H%M%S"))
            log(
                f"Distributed SingleCell training finished in "
                f"{time.time() - start:.2f}s; best AUC={self.best_auc:.4f}."
            )


def run_process(rank, world_size):
    setup_process(rank, world_size)
    try:
        set_seed(args.seed)
        handler = DistributedSingleCellDataHandler(
            rank=rank, world_size=world_size, device=args.device
        )
        handler.LoadData()
        coach = DistributedCoach(handler, rank)
        coach.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    args.dataset = canonical_dataset_name(args.dataset)
    if args.dataset != "SingleCell":
        raise ValueError(
            "Use Code.training.Main for GDSC/DrugBank/PDTC/TCGA; "
            "Code.single_cell_distributed.Main requires --dataset SingleCell."
        )
    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1.")
    available_gpus = torch.cuda.device_count()
    if available_gpus < args.gpus:
        raise RuntimeError(
            f"Requested {args.gpus} GPUs, but PyTorch sees {available_gpus}."
        )

    logger.saveDefault = True
    print(
        f"Starting distributed SingleCell training on {args.gpus} GPUs "
        f"(master={args.master_addr}:{args.master_port})."
    )
    mp.spawn(run_process, args=(args.gpus,), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
