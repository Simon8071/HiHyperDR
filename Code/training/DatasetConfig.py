import os


DATASET_CONFIGS = {
    "GDSC": {
        "directory": "GDSC",
        "mode": "gdsc",
        "default_eval_file": "val_balanced.csv",
    },
    "DrugBank": {
        "directory": "Drugbank",
        "mode": "drugbank",
        "default_eval_file": "test.csv",
    },
    "PDTC": {
        "directory": "PDTC",
        "mode": "expression",
        "expression_file": "merged_pdtc_expression.csv",
        "label_file": "merged_pdtc_labels.csv",
    },
    "TCGA": {
        "directory": "TCGA",
        "mode": "expression",
        "expression_file": "total_tcgadata.csv",
        "label_file": "total_tcgalabel.csv",
    },
    "SingleCell": {
        "directory": "single cell",
        "mode": "single_cell_distributed",
        "default_group": "ALL",
        "default_subset": "ALL",
    },
}

_DATASET_ALIASES = {
    "gdsc": "GDSC",
    "drugbank": "DrugBank",
    "drug_bank": "DrugBank",
    "pdtc": "PDTC",
    "tcga": "TCGA",
    "singlecell": "SingleCell",
    "single_cell": "SingleCell",
    "single cell": "SingleCell",
}


def canonical_dataset_name(name):
    key = str(name).strip().lower()
    if key not in _DATASET_ALIASES:
        choices = ", ".join(DATASET_CONFIGS)
        raise ValueError(f"Unsupported dataset {name!r}. Choose one of: {choices}.")
    return _DATASET_ALIASES[key]


def get_dataset_config(name):
    canonical = canonical_dataset_name(name)
    return canonical, DATASET_CONFIGS[canonical]


def resolve_data_dir(base_path, dataset):
    """Resolve either a common data root or a dataset-specific directory."""
    canonical, config = get_dataset_config(dataset)
    base_path = os.path.abspath(base_path)
    nested = os.path.join(base_path, config["directory"])

    if os.path.isdir(nested):
        return nested
    if os.path.isdir(base_path):
        basename = os.path.basename(os.path.normpath(base_path)).lower()
        accepted = {canonical.lower(), config["directory"].lower()}
        if basename in accepted:
            return base_path

    raise FileNotFoundError(
        f"Could not locate the {canonical} data directory. Looked for "
        f"{nested!r} and dataset directory {base_path!r}."
    )


def get_checkpoint_dir(code_root, dataset, override=None):
    if override:
        return os.path.abspath(override)
    canonical = canonical_dataset_name(dataset)
    root = os.path.join(code_root, "Models", "ckl")
    # Preserve the path used by the released GDSC checkpoint.
    return root if canonical == "GDSC" else os.path.join(root, canonical)


def get_default_checkpoint(code_root, dataset, override_dir=None):
    checkpoint_dir = get_checkpoint_dir(code_root, dataset, override_dir)
    canonical = canonical_dataset_name(dataset)
    candidates = (
        ["best_model.pkl", "best_auc_model.pkl"]
        if canonical == "GDSC"
        else ["best_auc_model.pkl", "best_model.pkl"]
    )
    for filename in candidates:
        path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(path):
            return path
    return os.path.join(checkpoint_dir, candidates[0])
