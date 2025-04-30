# AI Toolkit: Diffusion Model Training and Analysis

## Overview

AI Toolkit is a modular, production-grade framework for training and evaluating diffusion-based neural models (e.g., Stable Diffusion, FLUX). It provides:

- A reusable Python library (`toolkit/`) with abstractions for configuration, data loading, model patching (LoRA), training loops, and metrics computation.
- An agent-style job system (`jobs/`) supporting fine-tuning, hyperparameter optimization, cost benchmarking, and custom evaluations.
- CLI entrypoints (`run.py`, `optimize_run.py`) for batch execution via YAML/JSON configs.
- Jupyter notebooks (`notebooks/`) showcasing end-to-end workflows: fine-tuning, hyperopt, style analysis, and reporting.
- Automated reporting: generates Markdown and PDF summaries with plots and tables.
- Comprehensive tests (`tests/`) and CI workflows for linting, testing, and documentation.

---

## Repo Structure

```
ai-toolkit/
├── toolkit/                # Core library modules
│   ├── config.py           # Config loader and parser
│   ├── data_loader.py      # Dataset and DataLoader abstractions
│   ├── patch_lora.py       # LoRA adapter injection into UNet
│   ├── metrics.py          # Loss and evaluation metrics (MSE, CLIP, FID, IS)
│   ├── image_utils.py      # Utilities for image handling
│   ├── optimizer.py        # Optimizer factory (AdamW8bit, etc.)
│   ├── paths.py            # Standardized project paths
│   └── training/           # Subpackage for training routines
│       └── train.py        # Training loop (EMA, checkpointing, sampling)
├── jobs/                   # Job definitions using BaseJob abstraction
│   ├── BaseJob.py          # Core job orchestration
│   ├── BaseSDTrainProcess.py  # Base for SD training processes
│   ├── TrainJob.py         # Single-run fine-tuning job
│   ├── OptimizeJob.py      # Randomized hyperparameter search
│   ├── CostBenchmarkJob.py # Cost and performance benchmarking across providers
│   ├── BenchmarkJob.py     # Generic benchmark runner
│   └── ExtensionJob.py     # Adapter for custom job types
├── config/                 # Configuration files
│   ├── schnell_config.yaml     # Base training configuration
│   ├── optimize_params.yaml    # Hyperopt search spaces and trials
│   └── examples/            # Example YAML/JSON configs
├── notebooks/              # Demonstration notebooks
│   ├── Hyperopt_Demo.ipynb
│   ├── Art_Style_Evaluation.ipynb
│   └── HF_Upload_Inference_Demo.ipynb
├── tests/                  # Unit tests for toolkit and jobs
│   └── test_*.py
├── run.py                  # CLI for fine-tuning (alias for TrainJob)
├── optimize_run.py         # CLI for hyperparameter optimization
├── experiments/            # Output CSV/JSON/plots from runs
└── README.md               # This detailed overview
```

---

## Installation

```bash
git clone https://github.com/temefford/ai-toolkit.git
cd ai-toolkit
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Required packages include: torch, diffusers, accelerate, transformers, torchmetrics, matplotlib, seaborn, reportlab, pillow
```

### System Libraries

On Ubuntu/Debian:
```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0
```

---

## Configuration

### Base Training Config (`config/schnell_config.yaml`)

Defines model & training parameters:
- **process**: list of stages, e.g., `sd_trainer`, with nested keys:
  - `training_folder`, `device`, LoRA `network` settings,
  - `save` (checkpoint settings),
  - `datasets` with paths, augmentations, resolution buckets,
  - `train` hyperparameters: batch size, steps, optimizer, learning rate,
  - optional `performance_log_every`, `trigger_word`, and more.

### Hyperopt Config (`config/optimize_params.yaml`)

Specifies search space and trial count:
```yaml
optimize:
  trials: 10
  rank: [4, 8, 16, 32]
  lr: [5e-5, 1e-4, 2e-4]
  dropout: [0.05, 0.1, 0.2]
  batch_size: [1, 2, 4]
``` 

---

## CLI Usage

### 1. Fine-Tuning (TrainJob)

```bash
python run.py config/schnell_config.yaml
# or with accelerate for distributed:
accelerate launch run.py config/schnell_config.yaml
```

Outputs:
- Checkpoints and samples under `output/<timestamp>/`
- Optional HF Hub push if configured.

### 2. Hyperparameter Optimization (OptimizeJob)

```bash
python optimize_run.py
# or accelerate:
accelerate launch optimize_run.py
```

Generates:
- CSV of trial results
- Markdown & PDF reports in `outputs/Optimize_Reports/`
- Plots: cost vs. metric, hyperopt summary.

### 3. Cost Benchmarking (CostBenchmarkJob)

Configure `config/cost_benchmark.yaml`:
```yaml
cost_benchmark:
  base_config_file: config/schnell_config.yaml
  providers:
    - name: runpod, cost_per_sec: 0.05, throughput_ratio: 1.0
    - name: aws_g5, cost_per_sec: 0.03, throughput_ratio: 0.8
  datasets:
    - name: Baroque, metadata_file: Baroque/metadata.json, gt_dir: Baroque/gt, prompts_file: Baroque/prompts.txt
  output_dir: outputs/Cost_Benchmarks
```

Run:
```bash
python run.py config/cost_benchmark.yaml
```

Generate report in Python:
```python
from jobs.CostBenchmarkJob import CostBenchmarkJob
job = CostBenchmarkJob(config)
job.run()
job.create_extensive_report()
```

Reports saved to `outputs/Cost_Benchmarks/`.

### 4. Custom Benchmarks (BenchmarkJob)

Use `jobs/BenchmarkJob.py` for generic performance tests across datasets/providers. Refer to docstring for parameters.

---

## Toolkit Module Reference

### `toolkit/config.py`

- `get_config(path)`: loads YAML/JSON into nested `OrderedDict`.

### `toolkit/data_loader.py`

- `get_dataloader_from_datasets(...)`: returns PyTorch DataLoader with optional caching.
- Supports resolution bucketing, latent caching, dropout.

### `toolkit/patch_lora.py`

- Injects LoRA adapters into UNet layers via `lora_special` or `lycoris_special`.

### `toolkit/metrics.py`

- `compute_mse`, `compute_clip_score`, `compute_fid`, `compute_is`.
- Utilizes `torchmetrics` and `openai/clip`.

### `toolkit/training/train.py`

- `TrainLoop` class: handles step loop, EMA, logging, checkpointing, sampling.

### `toolkit/optimizer.py`

- Factory for optimizers: supports `adamw8bit`, `adamw`, etc.

---

## Jupyter Notebooks

- **Hyperopt_Demo.ipynb**: runs a hyperparameter search and visualizes results.
- **Art_Style_Evaluation.ipynb**: loads style metadata, computes metrics, plots style comparisons.
- **HF_Upload_Inference_Demo.ipynb**: demonstrates pushing to HF and inference.

---

## Testing & CI

- Run all tests:
```bash
pytest tests/
```
- Lint & format:
```bash
flake8 .
black .
```
- CI workflows in `.github/workflows/` for linting, testing, and docs.

---

## Contributing

1. Fork this repo and create a feature branch.
2. Add unit tests for new functionality.
3. Open a PR against `main` with a clear description.

---

## License

Apache-2.0  Black Forest Labs