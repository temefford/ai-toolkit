from collections import OrderedDict
from jobs import BaseJob
from toolkit.extension import get_all_extensions_process_dict
from collections import OrderedDict
from datetime import datetime
import time
from pathlib import Path
from PIL import Image
from toolkit.metrics import compute_validation_mse, compute_clip_score, compute_inception_score, compute_fid
import pandas as pd
import matplotlib.pyplot as plt
import os
from huggingface_hub import HfApi, login

class BenchmarkJob(BaseJob):
    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.early_stop_on_plateau = self.get_conf('early_stop_on_plateau', False)
        self.training_folder = self.get_conf('training_folder', required=True)
        self.is_v2 = self.get_conf('is_v2', False)
        self.log_dir = self.get_conf('log_dir', None)
        self.device = self.get_conf('device', 'cpu')
        # GPU cost per second (from config, default fallback)
        self.gpu_cost_per_second = self.config.get('cost_per_second', 0.0008333)
        self.process_dict = get_all_extensions_process_dict()
        self.load_processes(self.process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")
        # start timer for fine-tuning run
        start_time = time.time()
        last_losses = []
        for process in self.process:
            process.run(early_stop_on_plateau=self.early_stop_on_plateau)
            # Try to get loss from process if available
            loss = getattr(process, 'last_loss', None)
            if loss is not None:
                last_losses.append(loss)
                if len(last_losses) > 3:
                    last_losses.pop(0)
                if self.early_stop_on_plateau and len(last_losses) == 3:
                    if max(last_losses) - min(last_losses) < 0.001:
                        print('Early stopping: Loss plateaued for 3 consecutive runs.')
                        early_stop_triggered = True
                        break
        # end timer and compute cost
        end_time = time.time()
        duration = end_time - start_time
        self.finetune_duration = duration
        cost = duration * self.gpu_cost_per_second
        print(f"Fine-tuning run time: {duration:.2f}s, cost: ${cost:.4f}")
        # run evaluation metrics if configured
        eval_conf = self.get_conf('evaluation', None)
        if eval_conf:
            gt_folder = eval_conf.get('ground_truth_folder')
            gen_folder = eval_conf.get('generated_folder')
            prompts = eval_conf.get('prompts', [])
            if not gt_folder or not gen_folder:
                raise ValueError('evaluation config requires ground_truth_folder and generated_folder')
            gt_paths = sorted(Path(gt_folder).glob('*'))
            gen_paths = sorted(Path(gen_folder).glob('*'))
            gt_images = [Image.open(str(p)) for p in gt_paths]
            gen_images = [Image.open(str(p)) for p in gen_paths]
            mse = compute_validation_mse(gt_images, gen_images)
            clip = compute_clip_score(prompts, gen_images, device=self.device)
            is_mean, is_std = compute_inception_score(gen_images, device=self.device)
            fid_score = compute_fid(gt_images, gen_images, device=self.device)
            print('Evaluation metrics:')
            print(f'  MSE: {mse:.4f}')
            print(f'  CLIP Score: {clip:.4f}')
            print(f'  Inception Score: {is_mean:.4f} +/- {is_std:.4f}')
            print(f'  FID: {fid_score:.4f}')
            # save metrics and generate analysis
            metrics_dict = {'duration_s': duration, 'finetune_duration_s': getattr(self, 'finetune_duration', duration), 'cost_$': cost, 'MSE': mse, 'CLIP': clip, 'IS_mean': is_mean, 'IS_std': is_std, 'FID': fid_score}
            # Add hardware info from config if present
            hardware = self.config.get('hardware', None)
            if hardware:
                metrics_dict['hardware'] = hardware
            # include hyperparameters from config
            hyperparams = {}
            for section in ['train', 'sample', 'model']:
                section_conf = self.config.get(section, {})
                for key, val in section_conf.items():
                    hyperparams[f'{section}_{key}'] = val
            metrics_dict.update(hyperparams)
            # explicitly include selected train hyperparameters
            train_conf = self.config.get('train', {})
            for key in ['batch_size', 'steps', 'gradient_accumulation_steps', 'train_unet', 'noise_scheduler', 'optimizer', 'lr']:
                if key in train_conf:
                    metrics_dict[key] = train_conf[key]
            df = pd.DataFrame([metrics_dict])
            csv_path = Path(self.training_folder) / f"{self.name}_benchmark_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved results CSV to {csv_path}")

            # --- Compile Markdown Report ---
            results_dir = Path('results/benchmarks')
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            hardware_name = hardware if hardware else 'unknown_hardware'
            md_filename = f"{hardware_name}_{timestamp}.md"
            md_path = results_dir / md_filename
            # Markdown content
            md_content = f"""
# Benchmark Results ({hardware_name}) - {timestamp}

## Metrics

| Metric | Value |
|--------|-------|
"""
            for k, v in metrics_dict.items():
                md_content += f"| {k} | {v} |\n"
            # summary statistics
            stats = df.describe().T
            print("Metrics summary:")
            print(stats.to_string())
            # plot metrics
            fig, ax = plt.subplots(figsize=(8, 4))
            df.T.plot(kind='bar', legend=False, ax=ax)
            ax.set_title("Benchmark Metrics")
            ax.set_ylabel("Value")
            plt.xticks(rotation=45, ha='right')
            fig.tight_layout()
            plot_path = results_dir / f"{self.name}_benchmark_metrics.png"
            fig.savefig(plot_path)
            print(f"Saved metrics plot to {plot_path}")
            plt.close(fig)
            md_content += f"\n## Plot\n\n![]({plot_path.name})\n"
            md_content += f"\n## CSV\n\nResults CSV: `{csv_path}`\n"
            # Find last safetensor
            safetensors = list(Path(self.training_folder).rglob("*.safetensors"))
            if safetensors:
                last_safetensor = max(safetensors, key=lambda p: p.stat().st_mtime)
                md_content += f"\n## Model\n\nLast .safetensors file: `{last_safetensor}`\n"
            else:
                last_safetensor = None
                md_content += "\nNo .safetensors file found in training folder.\n"
            with open(md_path, 'w') as f:
                f.write(md_content)
            print(f"Saved markdown report to {md_path}")

            # --- Push to Hugging Face Hub ---
            hf_token = os.getenv("HF_TOKEN")
            hf_repo_id = os.getenv("HF_REPO_ID") or self.config.get('save', {}).get('hf_repo_id')
            if hf_token and hf_repo_id:
                try:
                    login(token=hf_token)
                    api = HfApi()
                    upload_files = [(md_path, md_path.name), (csv_path, csv_path.name), (plot_path, plot_path.name)]
                    if last_safetensor:
                        upload_files.append((last_safetensor, last_safetensor.name))
                    for local_path, repo_path in upload_files:
                        print(f"Uploading {local_path} to HuggingFace repo {hf_repo_id}...")
                        api.upload_file(
                            path_or_fileobj=str(local_path),
                            path_in_repo=repo_path,
                            repo_id=hf_repo_id,
                            repo_type="model",
                            token=hf_token,
                            commit_message=f"Add benchmark result: {repo_path} ({timestamp})"
                        )
                    print("Upload to HuggingFace complete!")
                except Exception as e:
                    print(f"Error uploading to HuggingFace: {e}")
            else:
                print("HF_TOKEN or HF_REPO_ID not set, skipping HuggingFace upload.")
        # Print last .safetensors file saved
        safetensors = list(Path(self.training_folder).rglob("*.safetensors"))
        if safetensors:
            last_safetensor = max(safetensors, key=lambda p: p.stat().st_mtime)
            print(f"Last .safetensors file saved: {last_safetensor}")
        else:
            print("No .safetensors file found in training folder.")
