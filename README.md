# HiHyperDR

HiHyperDR is a Hierarchical self-explainable hypergraph learning model
for drug response prediction. Unlike post-hoc methods that apply an external
explainer after training, HiHyperDR embeds explanation into model optimization.
Information Bottleneck constraints jointly optimize prediction and explanation,
allowing the model to produce explanations at the feature, local-graph, and
global-hypergraph levels that remain aligned with its internal decision logic.

## Main contributions

1. **Graph–hypergraph co-modeling.** Cell–drug bipartite graphs are combined
   with dynamic hypergraphs to capture both fine-grained pairwise interactions
   and integral higher-order biological dependencies.
2. **Information Bottleneck-driven endogenous self-explanation.**
   Interpretability is formulated as an intrinsic optimization objective, which
   reduces the fidelity gap between post-hoc explanations and model decisions.
3. **Multi-level explanation.** The framework identifies biological drivers at
   the feature, local-graph, and global-hypergraph levels, including important
   genes, drug substructures, interaction subgraphs, and sub-hypergraphs.
4. **Distributed single-cell extension.** A multi-GPU framework improves
   throughput and computational efficiency for large single-cell drug-response
   data while preserving global semantic integrity and predictive performance.

## Installation

Run all commands from the repository root. Install PyTorch for the CUDA version
on your machine, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Running the model

### Reproduce prediction results directly

Pretrained checkpoints are provided for GDSC, DrugBank, PDTC, and TCGA. The
following commands report accuracy, precision, recall, AUPR, AUC, F1, and the
confusion matrix:

```bash
python -m Code.evaluation.evaluate_prediction \
  --dataset GDSC \
  --checkpoint Code/Models/ckl/GDSC/best_model.pkl \
  --gpu 0

python -m Code.evaluation.evaluate_prediction \
  --dataset DrugBank \
  --config Code/Models/ckl/DrugBank/best_params.json \
  --checkpoint Code/Models/ckl/DrugBank/best_model.pkl \
  --gpu 0

python -m Code.evaluation.evaluate_prediction \
  --dataset PDTC \
  --config Code/Models/ckl/PDTC/best_params.json \
  --checkpoint Code/Models/ckl/PDTC/best_model.pkl \
  --gpu 0

python -m Code.evaluation.evaluate_prediction \
  --dataset TCGA \
  --config Code/Models/ckl/TCGA/best_params.json \
  --checkpoint Code/Models/ckl/TCGA/best_model.pkl \
  --gpu 0
```

### Standard datasets

GDSC, DrugBank, PDTC, and TCGA use the standard training entry point:

```bash
python -m Code.training.Main --dataset GDSC --gpu 0 --epoch 3000
python -m Code.training.Main --dataset DrugBank --gpu 0 --epoch 3000
python -m Code.training.Main --dataset PDTC --gpu 0 --epoch 3000
python -m Code.training.Main --dataset TCGA --gpu 0 --epoch 3000
```

PDTC and TCGA are stratified into training and test sets at an 8:2 ratio. The
best checkpoint is selected by AUC and saved under `Code/Models/ckl/`.

### Distributed single-cell training

The single-cell implementation has a separate multi-GPU entry point. The
default cohort is `ALL`:

```bash
python -m Code.single_cell_distributed.Main \
  --dataset SingleCell \
  --gpus 4 \
  --single_cell_group ALL
```

To run a cancer, drug-type, or tissue subset:

```bash
python -m Code.single_cell_distributed.Main \
  --dataset SingleCell \
  --gpus 4 \
  --single_cell_group cancer \
  --single_cell_subset "Breast cancer"
```

Valid groups are `ALL`, `cancer`, `drug`, and `tissue`. Subset checkpoints are
saved separately under `Code/Models/ckl/SingleCell/`.

### Reproduce explanation results

Explanation fidelity can be evaluated directly with the provided GDSC
checkpoint:

```bash
python -m Code.evaluation.explainer_eval_Fidelity \
  --dataset GDSC \
  --checkpoint_dir Code/Models/ckl/GDSC
```

Stability evaluation compares the base model with
`Code/Models/ckl/GDSC/stable_model_1.pkl`. This perturbed model must be generated
before running the stability evaluator:

```bash
# Step 1: search for and save stable_model_1.pkl
python -m Code.training.perturbation_model \
  --dataset GDSC \
  --checkpoint_dir Code/Models/ckl/GDSC

# Step 2: evaluate graph and hypergraph explanation stability
python -m Code.evaluation.explainer_eval_Stable \
  --dataset GDSC \
  --checkpoint_dir Code/Models/ckl/GDSC
```

## Code structure

| Path | Function |
| --- | --- |
| `Code/training/Main.py` | Standard training entry point |
| `Code/training/DataHandler.py` | Loads GDSC, DrugBank, PDTC, and TCGA and constructs graphs |
| `Code/training/DatasetConfig.py` | Dataset names, paths, and split configuration |
| `Code/training/FeatureInit.py` | Drug-graph and omics feature encoders |
| `Code/training/Model_sparse.py` | Main HiHyperDR graph–hypergraph model |
| `Code/training/Params.py` | Shared command-line and model parameters |
| `Code/training/perturbation_model.py` | Searches for a prediction-stable perturbed model |
| `Code/training/Utils/` | Losses, metrics, normalization, and logging utilities |
| `Code/single_cell_distributed/Main.py` | Multi-GPU single-cell training entry point |
| `Code/single_cell_distributed/DataHandler.py` | Partitions single-cell data and reads rank-local H5AD rows |
| `Code/single_cell_distributed/Model.py` | DDP-compatible single-cell HiHyperDR model |
| `Code/evaluation/evaluate_prediction.py` | Prediction performance evaluation |
| `Code/evaluation/explainer_eval_Fidelity.py` | Explanation fidelity evaluation |
| `Code/evaluation/explainer_eval_Stable.py` | Explanation stability evaluation |
| `Code/data/` | Dataset files and drug molecular graphs |
| `Code/Models/ckl/` | Trained model checkpoints and parameter files |
