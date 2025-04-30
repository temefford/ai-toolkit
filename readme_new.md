# fluxft: LoRA Fine-Tuning Toolkit for FLUX.1 [schnell]

## Omnicom POC Overview

**FLUX.1 [schnell]** is a 12B-parameter rectified flow transformer released under Apache-2.0 by Black Forest Labs.  We apply **Low-Rank Adaptation (LoRA)** to the UNet in `FluxPipeline` via the Axolotl framework, enabling efficient multi-GPU fine-tuning with drastically reduced trainable parameters and VRAM footprint.  Experiments run on RunPod H100 clouds but are fully reproducible on AWS (G5/G4dn) or Replit.

This repo delivers:
- A **production-grade Python package** (`toolkit/`) with modules for data loading, LoRA injection, training loops, hyperparameter search, and metric computation.
- **CLI scripts** (`run.py`, `optimize_run.py`) for batch runs in CI or terminal.
- **Jupyter notebooks** (`notebooks/`) demonstrating end-to-end fine-tuning, inference, hyperopt, and style evaluation.
- **Modular components**, **unit tests**, and **GitHub Actions** workflows for linting, testing, and docs.

---

## Objectives

1. **Reusable Fine-Tuning Library**  
   - Core implementation in `toolkit/`:
     - `patch_lora.py`: injects LoRA adapters into UNet layers.
     - `data/loader.py`: dataset & dataloader abstractions.
     - `training/train.py`: training loop with EMA, gradient accumulation, checkpointing.
   - Main entrypoint: `run.py` reads `config/*.yaml` or JSON to launch jobs via `jobs/GenerateJob.py`.

2. **Hyperparameter Optimization**  
   - `jobs/OptimizeJob.py`: randomized grid search over:
     - **rank** ∈ [4,8,16,32]
     - **lr** ∈ [5e-5,1e-4,2e-4]
     - **dropout** ∈ [0.05,0.1,0.2]
     - **batch_size** ∈ [1,2,4]
   - Config ranges in `config/optimize_params.yaml`.
   - Launcher script: `optimize_run.py` combines base config `config/schnell_config.yaml` and optimize ranges, runs `OptimizeJob`, saves `flux_schnell_optimize_hyperopt_results.png`.

3. **Performance Metric Definition**  
   Implemented in `toolkit/metrics.py`:
   - **Validation MSE**: pixel-wise MSE on held-out images.
   - **CLIP Score**: semantic alignment via `openai/clip-vit-base-patch32`.
   - **Inception Score (IS)** and **Fréchet Inception Distance (FID)** via `torchmetrics`.
   - Utilities convert PIL images to normalized tensors and resize to 299×299.

4. **Cost Benchmarking Across Clouds**  
   - `OptimizeJob` logs **GPU-seconds** per trial (`cost_sec`) and normalizes metrics.
   - Extendable to convert `cost_sec` → USD using per-second H100/AWS/Replit rates.
   - Future job module (`jobs/CostBenchmarkJob.py`) can automate across providers.

5. **Art-Style Evaluation**  
   - Curated datasets under `Baroque/metadata.json` (20–200 images/style).
   - Notebooks in `notebooks/art_style_evaluation.ipynb` load style metadata, run inference, compute metric trends via `toolkit/metrics.py`, and plot style comparisons.

---

## Scope & Deliverables

- **fluxft/** Python package with submodules:
  - `data/`, `patch_lora.py`, `training/`, `metrics.py`, `accelerator.py`, `job.py`.
- **CLI tools**:
  - `fluxft finetune`: wraps `GenerateJob` (alias via `run.py`).
  - `fluxft optimize`: wraps `OptimizeJob` (alias via `optimize_run.py`).
- **Jupyter notebooks**:
  - `notebooks/HF_Upload_Inference_Demo.ipynb` & fixed version.
  - `notebooks/Hyperopt_Demo.ipynb`.
  - `notebooks/Art_Style_Evaluation.ipynb`.
- **Tests & CI**:
  - Unit tests in `tests/` covering data loaders, LoRA injection, training loop, metric functions.
  - GitHub Actions: `lint.yml`, `test.yml`, `docs.yml`.
- **Documentation**:
  - API reference via Sphinx or MkDocs in `docs/`.
  - `readme_new.md` with full overview and instructions.
- **Experimental Reports**:
  - Hyperopt CSV/JSON outputs in `experiments/`.
  - Cost vs. performance tables and Matplotlib plots.

---

## Installation

```bash
git clone https://github.com/TheoMefff/flux_schnell_baroque.git
cd ai-toolkit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # includes diffusers, accelerate, torch, transformers, torchmetrics, matplotlib, seaborn
```

## Quickstart

### 1. Fine-Tune with LoRA
```bash
python run.py config/schnell_config.yaml
```
This launches `GenerateJob` (see `jobs/GenerateJob.py`), applies LoRA, trains on your dataset, and saves checkpoints & samples.

### 2. Hyperparameter Search
```bash
python optimize_run.py
```
Uses `jobs/OptimizeJob.py` + `config/optimize_params.yaml` to run randomized trials, produce a hyperopt plot in the root directory.

### 3. Performance Evaluation
Open `notebooks/Hyperopt_Demo.ipynb` or run:
```python
from toolkit.metrics import compute_clip_score, compute_fid  
# supply lists of PIL images and prompts
```

### 4. Art-Style Comparison
Launch `notebooks/Art_Style_Evaluation.ipynb` to generate side-by-side metrics & sample grids for each style in `Baroque/metadata.json`.

---

## Configuration

- **`config/schnell_config.yaml`**: model & training hyperparameters, paths, push_to_hub settings.
- **`config/optimize_params.yaml`**: hyperparameter ranges and trial count.

Customize any range or param then rerun the corresponding script.

---

## Development & Testing

- Run unit tests:
  ```bash
  pytest tests/
  ```
- Linting & formatting via `flake8` and `black`.
- CI workflows in `.github/workflows/`.

---

## Contributing

1. Fork repo & create feature branch.
2. Write tests for new functionality.
3. Submit PR with clear description & target branch `main`.

---

## License

Apache-2.0 © Black Forest Labs
