# Water Quality Spatial‑Temporal Imputation Using Diffusion Graph Convolutional Networks

This repository contains the core code accompanying the paper:

> **Water Quality Spatial‑Temporal Imputation Using Diffusion Graph Convolutional Networks:  
> A Case Study in Georgia, USA**

The framework reconstructs daily pH time series at unobserved river monitoring locations by:

- modeling spatio‑temporal correlations via **diffusion graph convolutional networks (DGCN)**,
- integrating multiple auxiliary water‑quality indicators through a **node‑level enhancement module**,
- training with an **inductive subgraph sampling and masking strategy** to remain robust to noisy or dynamic sensor networks,
- and comparing the proposed model against several baseline methods.

The public release here includes only the files needed to reproduce the main numerical experiments reported in the manuscript.

## Repository Contents

Tracked (to be uploaded to GitHub):

- `README.md` – This file.
- `config.py` – Helper for loading YAML configuration files (OmegaConf).
- `config_AUX_DGCN.yaml` – Configuration for the proposed AUX_DGCN model.
- `config_DGCN.yaml` – Configuration for the DGCN / IGNNK baseline.
- `config_LSTM.yaml` – Configuration for the LSTM baseline with auxiliary variables.
- `config_DGCN_LSTM.yaml` – Configuration for the hybrid DGCN_LSTM model.
- `config_KNN.yaml` – Configuration for the K‑nearest‑neighbor imputation baseline.
- `config_Mean.yaml` – Configuration for the global‑mean baseline (KNN with `k=0`).
- `data/preprocess.py` – Data partitioning, subgraph sampling, masking, and adjacency‑matrix utilities.
- `models/__init__.py` – Implementations of:
  - `IGNNK` (DGCN baseline),
  - `Aux_DGCN` (proposed auxiliary‑enhanced DGCN),
  - `LSTMmodel` (LSTM with auxiliary features),
  - `DGCN_LSTM` (hybrid DGCN + LSTM),
  - `KNN` (graph‑based KNN / global‑mean imputer),
  - and the shared `MODEL_REGISTRY`.
- `trainer.py` – Generic training and evaluation loop:
  - subgraph construction at each epoch,
  - early stopping on validation MAE,
  - final test‑set evaluation (MAE, MAPE, RMSE).
- `utils.py` – Helpers to build models and loss functions from the config.
- `main.py` – Script to train **AUX_DGCN** once (single run, single config).
- `run_with_baseline.py` – Script to train **all models** defined by `config_*.yaml` for multiple runs and summarize their test performance.
- `datasets/water_dataset.mat` – Daily water‑quality time series used in the case study (pH target and auxiliary indicators).
- `datasets/adjacency_matrix.npy` – Spatial adjacency matrix between monitoring locations (river‑network graph).

Untracked files and folders (e.g., raw data, logs, figures, analysis utilities) are *not* required for this code release and are therefore not documented here.

## Data Description

The experiments in the paper use a daily water‑quality dataset from **37 monitoring locations** in Georgia, USA.  
For each station and day, multiple indicators are available; the **median pH** is the imputation target, and the remaining indicators (e.g., dissolved oxygen, conductivity, temperature) are treated as auxiliary variables.

In the code, we assume:

- `datasets/water_dataset.mat` contains:
  - `Y_tr`, `Y_te`: target pH series for training and test locations,
  - `X_tr`, `X_te`: auxiliary indicators aligned with `Y_tr` and `Y_te`.
- `datasets/adjacency_matrix.npy` stores the spatial adjacency matrix between monitoring locations, constructed from the river network (e.g., thresholded Gaussian kernel on distances or group information).  
  A helper `generate_adjacency_matrix` is provided in `data/preprocess.py` for users who have grouping variables instead of a pre‑computed matrix.

Both `datasets/water_dataset.mat` and `datasets/adjacency_matrix.npy` are included in this repository to support direct reproduction of the experiments.  
If you wish to run the framework on your own data, you can replace these two files with datasets that follow the same structure.

## Models in This Release

All models operate on subgraphs sampled from the global monitoring network:

- **AUX_DGCN (proposed)** – Diffusion GCN that:
  - propagates information over the river network using random‑walk diffusion,
  - augments each node with an adaptive enhancement signal derived from auxiliary variables (NodeAdaptiveAugmentor),
  - is trained with subgraph sampling and masking to support inductive imputation at unseen locations.
- **DGCN (IGNNK)** – Diffusion GCN without auxiliary enhancement (graph‑only baseline).
- **LSTM** – Sequence model that flattens node and feature dimensions at each time step and applies a standard LSTM.
- **DGCN_LSTM** – Hybrid model combining a graph block and an LSTM block.
- **KNN / Mean** – Non‑parametric baselines that impute missing nodes using neighbors in the adjacency graph; `k=0` reduces to per‑time‑step global mean.

All training and evaluation logic is shared via `Trainer` in `trainer.py`.

## Running the Code

### 1. Environment

The code has been tested with Python 3.8+ and PyTorch 1.10+. A minimal environment can be created as:

```bash
conda create -n water-quality-dgcn python=3.10
conda activate water-quality-dgcn
pip install torch numpy scipy matplotlib omegaconf einops
```

### 2. Data (already included)

The repository ships with a `datasets/` folder containing:

- `water_dataset.mat`
- `adjacency_matrix.npy`

If you replace them with your own data, make sure the internal variable names and dimensions follow the conventions described in the paper and in `main.py` / `run_with_baseline.py`.

### 3. Train the Proposed Model (Single Run)

To train the proposed **AUX_DGCN** model once and save its best checkpoint:

```bash
python main.py
```

The configuration used is `config_AUX_DGCN.yaml`, and logs/checkpoints are written to the paths defined in this file.

### 4. Reproduce Baseline Comparison

To train all models (AUX_DGCN, DGCN, LSTM, DGCN_LSTM, KNN, Mean) for multiple runs and aggregate their test‑set metrics:

```bash
python run_with_baseline.py
```

This script:

- iterates over all `config_*.yaml` files,
- trains each model `num_runs` times,
- saves run‑wise histories under each model’s `log_dir`,
- writes a summary JSON file in `logs/` with the mean and standard deviation of MAE, MAPE, and RMSE across runs.

These summary statistics correspond to the model comparison results reported in the manuscript.

## Reproducibility Notes

- Each `config_*.yaml` specifies a global random seed (`seed`) and data‑sampling parameters (`unobserved_proportion`, `h`, `N_sub`, `u`, etc.).  
  Keeping these values unchanged is important for matching the protocol described in the paper.
- Training uses an early‑stopping strategy based on validation MAE and restores the best checkpoint before final test evaluation.
- Subgraph sampling and masking are implemented in `data/preprocess.py` and used consistently in both training and testing through `Trainer.fit`.
