import os
import json
import tempfile
from collections import OrderedDict

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import requests

from jobs import BaseJob
from toolkit.config import get_config
from toolkit.metrics import compute_clip_score


class OptimizeJob2(BaseJob):
    """
    Advanced HPO using Optuna with TPE sampler, MedianPruner, and k-fold CV.

    Config under config.optuna:
      base_config_file: path to base training YAML
      dataset_metadata: path to JSON metadata list
      n_trials: int (e.g., 50)
      timeout: seconds (e.g., 10800)
      n_folds: int (e.g., 5)
      random_seed: int
      n_startup_trials: int
    """

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        opt_cfg = self.get_conf('optuna', required=True)
        self.base_cfg_file = opt_cfg.get('base_config_file')
        self.metadata_file = opt_cfg.get('dataset_metadata')
        self.n_trials = int(opt_cfg.get('n_trials', 50))
        self.timeout = int(opt_cfg.get('timeout', 3 * 3600))
        self.n_folds = int(opt_cfg.get('n_folds', 5))
        self.random_seed = int(opt_cfg.get('random_seed', 42))
        self.n_startup = int(opt_cfg.get('n_startup_trials', 5))
        # load metadata list
        with open(self.metadata_file) as f:
            self.metadata = json.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        super().run()
        sampler = TPESampler(seed=self.random_seed)
        pruner = MedianPruner(n_startup_trials=self.n_startup, n_warmup_steps=1)
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        best = study.best_params
        print('Best hyperparameters:', best)
        out = f"{self.name}_best_params.json"
        with open(out, 'w') as f:
            json.dump(best, f, indent=4)
        print(f"Saved best params to {out}")

    def objective(self, trial):
        # suggest hyperparams
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.1)
        lora_rank = trial.suggest_categorical('lora_rank', [4, 8, 16, 32])
        projection = trial.suggest_categorical('projection', ['conv1x1', 'linear'])

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.metadata)):
            # split metadata
            train_meta = [self.metadata[i] for i in train_idx]
            val_meta = [self.metadata[i] for i in val_idx]
            train_file = self._write_temp(train_meta)
            val_file = self._write_temp(val_meta)

            # load base config
            base_full = get_config(self.base_cfg_file)
            cfg = base_full['config']
            # override dataset metadata
            if cfg.get('datasets'):
                cfg['datasets'][0]['metadata_file'] = train_file
            # inject hyperparams
            cfg['train']['lr'] = lr
            cfg['train']['batch_size'] = batch_size
            cfg['train']['weight_decay'] = weight_decay
            cfg['network']['linear'] = lora_rank
            cfg['network']['projection'] = projection
            # run training
            from jobs.TrainJob import TrainJob
            job = TrainJob({'job': 'train', 'config': cfg})
            job.run()
            job.cleanup()
            # pick latest checkpoint
            out_folder = cfg.get('training_folder')
            cpts = sorted([os.path.join(out_folder, f) for f in os.listdir(out_folder) if f.endswith('.safetensors')])
            ckpt = cpts[-1]
            # load pipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                ckpt,
                torch_dtype=torch.float16 if self.device.type=='cuda' else torch.float32
            ).to(self.device)
            # eval on val
            prompts = [item['caption'] for item in val_meta]
            imgs = []
            for it in val_meta:
                link = it['Link']; ext = os.path.splitext(link)[1]
                r = requests.get(link)
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(r.content); tmp.close()
                imgs.append(Image.open(tmp.name))
            gen = [pipe(p).images[0] for p in prompts]
            clip = compute_clip_score(prompts, gen, device=self.device)
            scores.append(clip)
            trial.report(sum(scores)/len(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return sum(scores)/len(scores)

    def _write_temp(self, meta_list):
        tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        with open(tmp.name, 'w') as f:
            json.dump(meta_list, f)
        return tmp.name
