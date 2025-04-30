import os
import random
import time
import copy
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd

from jobs import BaseJob
from toolkit.job import get_job


class OptimizeJob(BaseJob):
    """
    Randomized hyperparameter search over LoRA rank, learning rate, dropout, and batch size.

    Expects an 'optimize' section in config under 'config':
      optimize:
        trials: 10
        rank: [4, 8, 16, 32]
        lr: [5e-5, 1e-4, 2e-4]
        dropout: [0.05, 0.1, 0.2]
        batch_size: [1, 2, 4]
    And a 'base_config' section under 'config' containing the training setup.
    """
    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.device = self.get_conf('device', 'cpu')
        self.optimize_cfg = self.get_conf('optimize', required=True)
        self.base_cfg = self.get_conf('base_config', required=True)

    def run(self):
        super().run()
        trials = int(self.optimize_cfg.get('trials', len(self.optimize_cfg.get('rank', []))))
        results = []

        for i in range(trials):
            # sample hyperparameters
            rank = random.choice(self.optimize_cfg.get('rank', [self.base_cfg['network']['linear']]))
            lr = random.choice(self.optimize_cfg.get('lr', [self.base_cfg['train']['lr']]))
            dropout = random.choice(self.optimize_cfg.get('dropout', [self.base_cfg['datasets'][0].get('caption_dropout_rate', 0)]))
            batch_size = random.choice(self.optimize_cfg.get('batch_size', [self.base_cfg['train']['batch_size']]))

            # prepare trial config
            trial_cfg = copy.deepcopy(self.base_cfg)
            trial_cfg['network']['linear'] = rank
            trial_cfg['train']['lr'] = lr
            trial_cfg['train']['batch_size'] = batch_size
            trial_cfg['datasets'][0]['caption_dropout_rate'] = dropout

            full_conf = {'job': 'extension', 'config': trial_cfg}

            # run training and measure cost
            start_ts = time.time()
            job = get_job(full_conf, None)
            job.run()
            job.cleanup()
            duration = time.time() - start_ts

            # placeholder for metrics - processes may define attributes
            proc = job.process[-1]
            val_loss = getattr(proc, 'val_loss', None)
            clip_score = getattr(proc, 'clip_score', None)

            results.append({
                'trial': i,
                'rank': rank,
                'lr': lr,
                'dropout': dropout,
                'batch_size': batch_size,
                'cost_sec': duration,
                'val_loss': val_loss,
                'clip_score': clip_score,
            })

        # normalize metrics by cost
        for r in results:
            if r['val_loss'] is not None:
                r['norm_loss'] = r['val_loss'] * r['cost_sec']
            if r['clip_score'] is not None:
                r['norm_clip'] = r['clip_score'] / r['cost_sec']

        # create DataFrame and plot results
        df = pd.DataFrame(results)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(df['lr'], df['norm_clip'], c=df['rank'], cmap='viridis', s=100)
        plt.colorbar(sc, label='LoRA rank')
        plt.xlabel('Learning Rate')
        plt.ylabel('Normalized CLIP Score')
        plt.title('Hyperparameter Search Results')
        plot_path = os.path.join(os.getcwd(), f"{self.name}_hyperopt_results.png")
        plt.savefig(plot_path)
        print(f"Saved hyperparameter search plot to {plot_path}")
