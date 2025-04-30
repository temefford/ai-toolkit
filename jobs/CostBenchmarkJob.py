import os
import time
import copy
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from jobs import BaseJob
from jobs.TrainJob import TrainJob
from jobs.EvaluationJob import EvaluationJob
from toolkit.config import get_config


class CostBenchmarkJob(BaseJob):
    """
    Run identical fine-tuning experiments across multiple cloud/hardware providers,
    track cost (USD) and performance, and aggregate results.

    Config structure under config.cost_benchmark:
      base_config_file: path to base training YAML
      providers:
        - name: str
          cost_per_sec: float
          throughput_ratio: float  # relative speed (1.0 = baseline)
      datasets:
        - name: str
          metadata_file: path
          gt_dir: path  # ground truth images
          prompts_file: path
      performance_target:
        metric_name: threshold_value  # e.g., clip: 0.3
      output_dir: path for results
    """

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        cb = self.get_conf('cost_benchmark', required=True)
        self.base_config_file = cb.get('base_config_file')
        self.providers = cb.get('providers', [])
        self.datasets = cb.get('datasets', [])
        self.performance_target = cb.get('performance_target', {})
        self.output_dir = cb.get('output_dir', 'cost_benchmark')
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = self.get_conf('device', 'cpu')
        self.records = []

    def run(self):
        super().run()
        # load base config
        base_full = get_config(self.base_config_file)
        base_cfg = base_full['config']

        metric_key = list(self.performance_target.keys())[0] if self.performance_target else None
        for prov in self.providers:
            pname = prov['name']
            cost_rate = prov.get('cost_per_sec', 0.0)
            throughput = prov.get('throughput_ratio', 1.0)
            for ds in self.datasets:
                ds_name = ds['name']
                meta_file = ds.get('metadata_file')
                gt_dir = ds.get('gt_dir')
                prompts_file = ds.get('prompts_file')

                # prepare training config
                cfg = copy.deepcopy(base_cfg)
                if 'datasets' in cfg and cfg['datasets']:
                    cfg['datasets'][0]['metadata_file'] = meta_file
                # set output folder for this run
                exp_dir = os.path.join(self.output_dir, f"{pname}_{ds_name}")
                cfg['training_folder'] = exp_dir

                # TODO: early stopping when performance_target is reached
                start = time.time()
                job = TrainJob({'job': 'train', 'config': cfg, 'meta': OrderedDict({'provider': pname, 'dataset': ds_name})})
                job.run()
                job.cleanup()
                elapsed = time.time() - start

                # compute cost USD
                cost_usd = (elapsed / throughput) * cost_rate

                # run evaluation on validation set
                eval_conf = {
                    'job': 'evaluation',
                    'config': {
                        'name': f"{pname}_{ds_name}_eval",
                        'evaluation': {
                            'ckpts': [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith('.safetensors')],
                            'validation': {'gt_dir': gt_dir, 'prompts_file': prompts_file},
                            'art_styles': []
                        }
                    }
                }
                eval_job = EvaluationJob(eval_conf)
                eval_job.run()
                eval_job.cleanup()

                # extract performance metric
                perf_value = None
                for e in eval_job.results:
                    if e['evaluation_type'] == 'validation':
                        perf_value = e.get(metric_key)
                        break

                self.records.append({
                    'provider': pname,
                    'dataset': ds_name,
                    'elapsed_sec': elapsed,
                    'cost_usd': cost_usd,
                    metric_key: perf_value
                })

        # aggregate results
        df = pd.DataFrame(self.records)
        csv_path = os.path.join(self.output_dir, f"{self.name}_cost_analysis.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved cost analysis CSV: {csv_path}")

        sns.set(style='whitegrid')
        # cost per dataset per provider
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x='dataset', y='cost_usd', hue='provider')
        plt.title('Cost per Dataset by Provider')
        plt.savefig(os.path.join(self.output_dir, f"{self.name}_cost_comparison.png"))

        # cost vs performance
        if metric_key:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x='cost_usd', y=metric_key, hue='provider', style='dataset', s=100)
            plt.title(f"Cost vs {metric_key}" )
            plt.savefig(os.path.join(self.output_dir, f"{self.name}_cost_vs_{metric_key}.png"))
            print(f"Saved cost vs performance plot.")
