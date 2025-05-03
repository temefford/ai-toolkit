import os
import random
import time
import copy
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import HfApi, login

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
        import gc
        import torch
        trials = int(self.optimize_cfg.get('trials', len(self.optimize_cfg.get('rank', []))))
        results = []

        for i in range(trials):
            # Memory cleanup before starting a new training job
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # sample hyperparameters from nested process[0]
            base_proc = self.base_cfg['process'][0]
            rank = random.choice(self.optimize_cfg.get('rank', [base_proc['network']['linear']]))
            lr = random.choice(self.optimize_cfg.get('lr', [base_proc['train']['lr']]))
            dropout = random.choice(self.optimize_cfg.get('dropout', [base_proc['datasets'][0].get('caption_dropout_rate', 0)]))
            batch_size = random.choice(self.optimize_cfg.get('batch_size', [base_proc['train']['batch_size']]))

            # prepare trial config (deepcopy and update process[0])
            trial_cfg = copy.deepcopy(self.base_cfg)
            proc = trial_cfg['process'][0]
            proc['network']['linear'] = rank
            proc['train']['lr'] = lr
            proc['train']['batch_size'] = batch_size
            proc['datasets'][0]['caption_dropout_rate'] = dropout

            full_conf = {'job': 'benchmark', 'config': trial_cfg}

            # run training and measure cost
            start_ts = time.time()
            job = get_job(full_conf, None)
            job.run()
            job.cleanup()
            # Memory cleanup after finishing a training job
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

        # Ensure outputs directory exists
        outputs_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        timestamp = "20250430-101446"
        plot_path = os.path.join(outputs_dir, f"{self.name}_hyperopt_results_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"Saved hyperparameter search plot to {plot_path}")

        # Save results table as markdown
        table_md = df.to_markdown(index=False)

        # Compose markdown report
        md_report = f"""
# Hyperparameter Optimization Results ({timestamp})

![Results Plot](./{os.path.basename(plot_path)})

## Results Table

{table_md}
"""
        md_path = os.path.join(outputs_dir, f"{self.name}_hyperopt_results_{timestamp}.md")
        with open(md_path, 'w') as f:
            f.write(md_report)
        print(f"Saved markdown report to {md_path}")

        # Try to export PDF (requires markdown2 and reportlab or pypandoc)
        try:
            import markdown2
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            import io
            from PIL import Image
            pdf_path = os.path.join(outputs_dir, f"{self.name}_hyperopt_results_{timestamp}.pdf")
            # Convert markdown to HTML
            html = markdown2.markdown(md_report)
            # Render PDF (simple: just text and image)
            c = canvas.Canvas(pdf_path, pagesize=letter)
            width, height = letter
            textobject = c.beginText(40, height-40)
            for line in md_report.split('\n'):
                textobject.textLine(line)
            c.drawText(textobject)
            # Insert plot image
            img = Image.open(plot_path)
            img_width, img_height = img.size
            aspect = img_height / img_width
            pdf_img_width = width - 80
            pdf_img_height = pdf_img_width * aspect
            c.drawInlineImage(plot_path, 40, height-40-pdf_img_height-20, width=pdf_img_width, height=pdf_img_height)
            c.showPage()
            c.save()
            print(f"Saved PDF report to {pdf_path}")
        except Exception as e:
            print(f"Could not generate PDF: {e}\nInstall markdown2, reportlab, and pillow for PDF export.")

        # Save detailed results to CSV
        csv_path = os.path.join(outputs_dir, f"{self.name}_optimize_results_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed results CSV to {csv_path}")

        # Save aggregated markdown report
        agg_md_path = os.path.join(outputs_dir, "optimize_results.md")
        agg_md_content = f"# Hyperparameter Optimization Results ({timestamp})\n\n"
        agg_md_content += "## Trials Results\n\n"
        agg_md_content += df.to_markdown(index=False)
        with open(agg_md_path, 'w') as f:
            f.write(agg_md_content)
        print(f"Saved aggregated markdown report to {agg_md_path}")

        # --- Upload optimize results CSV to Hugging Face ---
        hf_token = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID") or self.config.get('save', {}).get('hf_repo_id')
        if hf_token and hf_repo_id:
            try:
                login(token=hf_token)
                api = HfApi()
                print(f"Uploading CSV {csv_path} to {hf_repo_id}...")
                api.upload_file(
                    path_or_fileobj=str(csv_path),
                    path_in_repo=os.path.basename(csv_path),
                    repo_id=hf_repo_id,
                    repo_type="model",
                    token=hf_token,
                    commit_message=f"Add optimize results CSV ({timestamp})"
                )
                print("CSV upload complete!")
            except Exception as e:
                print(f"Error uploading CSV to Hugging Face: {e}")
        else:
            print("HF_TOKEN or HF_REPO_ID not set, skipping CSV upload.")
