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
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub
# Required packages include: torch, diffusers, accelerate, transformers, torchmetrics, matplotlib, seaborn, reportlab, pillow
```

### System Libraries

On Ubuntu/Debian:
```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
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

This job runs identical fine-tuning and evaluation across multiple hardware providers and datasets and aggregates cost (USD) vs. performance metrics.

Sample config (`config/cost_benchmark.yaml`):
```yaml
job: "cost_benchmark"
config:
  name: "cost_benchmark"
  hardware: "runpod"
  base_config_file: "config/schnell_config.yaml"
  providers:
    - name: "runpod"
      cost_per_sec: 0.08333
      throughput_ratio: 1.0
    - name: "aws_g5"
      cost_per_sec: 0.03
      throughput_ratio: 0.8
  datasets:
    - name: "Baroque"
      metadata_file: "Baroque/metadata.json"
      gt_dir: "Baroque/gt"
      prompts_file: "Baroque/prompts.txt"
  output_dir: "outputs/Cost_Benchmarks"
```

**Run benchmark:**
```bash
python run.py config/cost_benchmark.yaml
# or distributed
accelerate launch run.py config/cost_benchmark.yaml
```

For each provider–dataset pair, a subfolder is created under the configured `output_dir` with the pattern:
```
<output_dir>/<hardware>_<provider>_<dataset>_<timestamp>/
```
Inside each subfolder you'll find:
- Model checkpoints (`*.safetensors`)
- Sample images and training logs

After training and evaluation complete, generate an aggregated report:
```bash
python - <<EOF
from toolkit.config import get_config
from jobs.CostBenchmarkJob import CostBenchmarkJob
cfg = get_config("config/cost_benchmark.yaml")
job = CostBenchmarkJob(cfg)
job.run()
job.create_extensive_report()
EOF
```

Report artifacts are saved in `outputs/Cost_Benchmarks/`:
- Markdown: `cost_benchmark_report_<timestamp>.md`
- PDF: `cost_benchmark_report_<timestamp>.pdf`
- Plots: `cost_comparison_<timestamp>.png`, `cost_vs_<metric>_<timestamp>.png`

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

---

## Running on Rackspace PVC

This section provides step-by-step instructions for running AI Toolkit on Rackspace Private Cloud (PVC) with H100 GPUs. It covers environment setup, driver installation, CUDA, PyTorch, and troubleshooting common issues such as code hanging at image generation.

### 1. Provision an H100 Instance
- Deploy an Ubuntu 24.04 (or similar) VM with an H100 GPU attached.

### 2. Install NVIDIA Drivers (Required for H100)
**H100 requires NVIDIA driver version 525.x or newer.**

```bash
sudo apt update
sudo apt install nvidia-driver-550
sudo reboot
```

After reboot, verify the driver and GPU are recognized:

```bash
nvidia-smi
```
You should see your H100 GPU listed. If you get `Command 'nvidia-smi' not found` or no GPU is shown, the driver is not installed or loaded.

### 3. Install CUDA Toolkit (Optional)
- CUDA toolkit is typically only needed for compiling custom ops. H100 requires CUDA 12.x+.
- Your Python environment already includes CUDA runtime libraries via pip packages (e.g., `nvidia-cudnn-cu12`).

### 4. Set Up Python Environment
```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 5. Verify PyTorch Sees the GPU
Run the following in Python:
```python
import torch
print(torch.cuda.is_available())          # Should print: True
print(torch.cuda.get_device_name(0))      # Should print: NVIDIA H100 ...
```
If these fail, double-check your driver and CUDA installation.

### 6. Run Your Training Script
Proceed as normal:
```bash
python run.py --config config/schnell_config.yaml
```

### 7. Troubleshooting: Code Hangs at "Generating Images"
If your code hangs at the point of generating images or just before training:
- **Root Cause:** The NVIDIA driver is missing or not loaded, so PyTorch cannot see the GPU.
- **Fix:** Install the correct NVIDIA driver (see above), then reboot.
- **Check:** `nvidia-smi` must work and show your GPU.

#### Additional Checks
- Ensure your PyTorch and CUDA versions are compatible with H100 (PyTorch 2.0+, CUDA 12.x+).
- Avoid using custom CUDA extensions that may not support H100.
- Set DataLoader `num_workers=0` if you suspect multiprocessing issues.
- Compare your environment (`pip freeze`, `nvidia-smi`, `nvcc --version`) with a working setup (e.g., Runpod).

### 8. Example Environment (Working)
- **PyTorch:** 2.6.0
- **CUDA Toolkit:** 12.0
- **nvidia-cudnn-cu12:** 9.1.0.70
- **Driver:** 550+

### 9. Support
If you encounter issues, collect the output of:
- `nvidia-smi`
- `nvcc --version`
- `pip freeze`
- Error logs from your training script

and share with your support channel or open an issue on GitHub.