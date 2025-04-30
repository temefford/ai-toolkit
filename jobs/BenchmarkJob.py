import os
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

from jobs import BaseJob
from jobs.TrainJob import TrainJob
from jobs.EvaluationJob import EvaluationJob
from toolkit.config import get_config


class BenchmarkJob(BaseJob):
    """
    Orchestrates fine-tuning across multiple style datasets using optimized hyperparameters,
    then evaluates cost and performance and creates visualizations.
    """

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        bench_cfg = self.get_conf('benchmark', required=True)
        # Base training config path
        self.base_config_file = bench_cfg.get('base_config_file')
        # List of style datasets: each with 'name' and 'metadata_file'
        self.style_datasets = bench_cfg.get('style_datasets', [])
        # Best hyperparameters from optimization
        self.best_params = bench_cfg.get('best_params', {})
        # Device and output directory
        self.device = self.get_conf('device', 'cpu')
        self.output_dir = bench_cfg.get('output_dir', 'benchmarks')
        os.makedirs(self.output_dir, exist_ok=True)
        self.records = []
        self.hardware = self.get_conf('hardware', required=True)

    def run(self):
        super().run()
        # Load base config
        base_full = get_config(self.base_config_file)
        base_cfg = base_full['config']

        # Fine-tune models for each style
        for item in self.style_datasets:
            name = item.get('name')
            meta_file = item.get('metadata_file')
            print(f"Starting fine-tuning for style: {name}")

            # Prepare training config
            cfg = copy.deepcopy(base_cfg)
            # Override dataset metadata file
            if 'datasets' in cfg and cfg['datasets']:
                cfg['datasets'][0]['metadata_file'] = meta_file
            # Set output folder
            style_out = os.path.join(self.output_dir, name)
            cfg['training_folder'] = style_out

            # Run training
            start = time.time()
            train_conf = {'job': 'train', 'config': cfg}
            job = TrainJob(train_conf)
            job.run()
            job.cleanup()
            duration = time.time() - start

            # Locate latest checkpoint
            ckpts = []
            if os.path.isdir(style_out):
                ckpts = [os.path.join(style_out, f) for f in os.listdir(style_out) if f.endswith('.safetensors')]
                ckpts.sort()
            best_ckpt = ckpts[-1] if ckpts else None
            self.records.append({'style': name, 'checkpoint': best_ckpt, 'train_time': duration})

        # Plot training times
        df_train = pd.DataFrame(self.records)
        csv_train = os.path.join(self.output_dir, f"{self.name}_train_times.csv")
        df_train.to_csv(csv_train, index=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df_train, x='style', y='train_time')
        plt.title('Training Time per Style')
        plt.ylabel('Time (s)')
        plt.xlabel('Style')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.name}_train_times.png"))
        print(f"Saved training times to {csv_train} and plot.")

        # Run evaluation and style comparison
        eval_conf = {
            'job': 'evaluation',
            'config': {
                'name': f"{self.name}_eval",
                'evaluation': {
                    'ckpts': [r['checkpoint'] for r in self.records if r['checkpoint']],
                    'validation': {},
                    'art_styles': [item['metadata_file'] for item in self.style_datasets],
                }
            }
        }
        eval_job = EvaluationJob(eval_conf)
        eval_job.run()
        eval_job.cleanup()
