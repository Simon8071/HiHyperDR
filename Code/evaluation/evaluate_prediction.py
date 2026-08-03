import os

import numpy as np
import torch
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

from Code.training.DataHandler import DataHandler
from Code.training.DatasetConfig import get_default_checkpoint
from Code.training.Model_sparse import Model
from Code.training.Params import args


EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(EVALUATION_DIR)
def build_model(handler):
    drug_cfg = {
        "d_atom": handler.drug_batch.x.shape[1],
        "d_model": args.latdim,
        "dropout": 0.1,
        "num_total_atoms": handler.drug_batch.x.shape[0],
    }
    omics_cfg = {
        "omics_dims": [tensor.shape[1] for tensor in handler.omics_inputs_all],
        "num_cells": handler.omics_inputs_all[0].shape[0],
        "hidden_dim": 128,
        "proj_dim": args.latdim,
    }
    return Model(drug_encoder_cfg=drug_cfg, omics_encoder_cfg=omics_cfg).to(args.device)


def main():
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    args.device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    handler = DataHandler()
    handler.LoadData(eval_filename=args.split)
    model = build_model(handler)
    checkpoint = args.checkpoint or get_default_checkpoint(
        CODE_ROOT, args.dataset, args.checkpoint_dir
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train {handler.dataset_name} first "
            "or pass --checkpoint explicitly."
        )
    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        logits = model.predict(
            handler.torchBiAdj,
            handler.test_d,
            handler.test_c,
            drug_batch=handler.drug_batch,
            omics_inputs=handler.omics_inputs_all,
        )
        probabilities = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    labels = handler.test_y.cpu().numpy()
    predictions = (probabilities > args.threshold).astype(np.int64)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "aupr": average_precision_score(labels, probabilities),
        "auc": roc_auc_score(labels, probabilities),
        "f1": f1_score(labels, predictions, zero_division=0),
    }

    print(f"Device: {args.device}")
    print(f"Dataset: {handler.dataset_name}")
    print(f"Split: {handler.eval_split_name}")
    print(f"Model: {args.model if args.checkpoint is None else 'custom'}")
    print(f"Checkpoint: {checkpoint}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("Confusion matrix (TN FP / FN TP):")
    print(confusion_matrix(labels, predictions))


if __name__ == "__main__":
    main()
