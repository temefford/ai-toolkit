import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

from collections import OrderedDict
from jobs import BaseJob
from toolkit.metrics import (
    compute_validation_mse,
    compute_clip_score,
    compute_inception_score,
    compute_fid,
)

class EvaluationJob(BaseJob):
    """
    Performance evaluation and art style comparison job.
    Evaluates model checkpoints on a validation set and multiple art style datasets.
    """
    def __init__(self, config: OrderedDict):
        super().__init__(config)
        eval_cfg = self.get_conf('evaluation', required=True)
        self.ckpts = eval_cfg.get('ckpts', [])
        val_cfg = eval_cfg.get('validation', {})
        self.gt_dir = val_cfg.get('gt_dir')
        self.prompts_file = val_cfg.get('prompts_file')
        self.style_files = eval_cfg.get('art_styles', [])
        self.device = self.get_conf('device', 'cpu')
        self.results = []

    def run(self):
        super().run()
        # ----- Validation evaluation -----
        for ckpt in self.ckpts:
            ckpt_name = os.path.basename(ckpt)
            print(f'Evaluating checkpoint: {ckpt_name}')
            t0 = time.time()
            pipe = StableDiffusionPipeline.from_pretrained(
                ckpt,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)
            load_time = time.time() - t0

            mse, clip = None, None
            if self.gt_dir and self.prompts_file:
                with open(self.prompts_file) as f:
                    prompts = [l.strip() for l in f if l.strip()]
                gt_images = [Image.open(os.path.join(self.gt_dir, f'{i}.png')) for i in range(len(prompts))]
                gen_images = [pipe(p).images[0] for p in prompts]
                mse = compute_validation_mse(gt_images, gen_images)
                clip = compute_clip_score(prompts, gen_images, device=self.device)

            self.results.append({
                'checkpoint': ckpt_name,
                'evaluation_type': 'validation',
                'mse': mse,
                'clip': clip,
                'time_sec': load_time,
            })

        # ----- Art style comparison -----
        for style_file in self.style_files:
            style_name = os.path.splitext(os.path.basename(style_file))[0]
            with open(style_file) as f:
                meta = json.load(f)
            cache_dir = os.path.join('art_style_data', style_name)
            os.makedirs(cache_dir, exist_ok=True)
            prompts, gt_images = [], []
            for item in meta:
                caption = item['caption']
                url = item['Link']
                ext = os.path.splitext(url)[1]
                local_path = os.path.join(cache_dir, f'{item["hash"]}{ext}')
                if not os.path.exists(local_path):
                    resp = requests.get(url)
                    with open(local_path, 'wb') as fh:
                        fh.write(resp.content)
                prompts.append(caption)
                gt_images.append(Image.open(local_path))

            for ckpt in self.ckpts:
                ckpt_name = os.path.basename(ckpt)
                print(f'Style eval: {style_name} on {ckpt_name}')
                pipe = StableDiffusionPipeline.from_pretrained(
                    ckpt,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                ).to(self.device)
                gen_images = [pipe(c).images[0] for c in prompts]
                clip_s = compute_clip_score(prompts, gen_images, device=self.device)
                is_mean, is_std = compute_inception_score(gen_images, device=self.device)
                fid_s = compute_fid(gt_images, gen_images, device=self.device)

                self.results.append({
                    'checkpoint': ckpt_name,
                    'evaluation_type': f'style_{style_name}',
                    'clip': clip_s,
                    'is_mean': is_mean,
                    'is_std': is_std,
                    'fid': fid_s,
                })

        # ----- Save and visualize -----
        df = pd.DataFrame(self.results)
        out_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f'{self.name}_results.csv')
        df.to_csv(csv_path, index=False)
        print(f'Saved results CSV: {csv_path}')

        sns.set(style='whitegrid')
        val_df = df[df['evaluation_type']=='validation']
        if not val_df.empty:
            plt.figure(figsize=(8,6))
            sns.scatterplot(data=val_df, x='time_sec', y='clip', hue='checkpoint', s=100)
            plt.title('Validation CLIP vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('CLIP Score')
            plt.savefig(os.path.join(out_dir, f'{self.name}_validation_clip_time.png'))

        style_df = df[df['evaluation_type'].str.startswith('style_')]
        if not style_df.empty:
            style_df['style'] = style_df['evaluation_type'].str.replace('style_', '')
            plt.figure(figsize=(10,6))
            sns.barplot(data=style_df, x='style', y='clip', hue='checkpoint')
            plt.title('Art Style CLIP Scores by Checkpoint')
            plt.savefig(os.path.join(out_dir, f'{self.name}_style_clip.png'))

            plt.figure(figsize=(10,6))
            sns.barplot(data=style_df, x='style', y='fid', hue='checkpoint')
            plt.title('Art Style FID by Checkpoint')
            plt.savefig(os.path.join(out_dir, f'{self.name}_style_fid.png'))

        print(f'Saved visualizations to {out_dir}')
