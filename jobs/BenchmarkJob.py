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

class BenchmarkJob(BaseJob):
    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.training_folder = self.get_conf('training_folder', required=True)
        self.is_v2 = self.get_conf('is_v2', False)
        self.log_dir = self.get_conf('log_dir', None)
        self.device = self.get_conf('device', 'cpu')
        # GPU cost per second (e.g. runpod rate)
        self.gpu_cost_per_second = self.get_conf('gpu_cost_per_second', 0.0083)
        self.process_dict = get_all_extensions_process_dict()
        self.load_processes(self.process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")
        # start timer for fine-tuning run
        start_time = time.time()
        for process in self.process:
            process.run()
        # end timer and compute cost
        end_time = time.time()
        duration = end_time - start_time
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
            metrics_dict = {'duration_s': duration, 'cost_$': cost, 'MSE': mse, 'CLIP': clip, 'IS_mean': is_mean, 'IS_std': is_std, 'FID': fid_score}
            # include hyperparameters from config
            hyperparams = {}
            for section in ['train', 'sample', 'model']:
                section_conf = self.config.get(section, {})
                for key, val in section_conf.items():
                    hyperparams[f'{section}_{key}'] = val
            metrics_dict.update(hyperparams)
            df = pd.DataFrame([metrics_dict])
            csv_path = Path(self.training_folder) / f"{self.name}_benchmark_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved results CSV to {csv_path}")
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
            plot_path = Path(self.training_folder) / f"{self.name}_benchmark_metrics.png"
            fig.savefig(plot_path)
            print(f"Saved metrics plot to {plot_path}")
            plt.close(fig)
