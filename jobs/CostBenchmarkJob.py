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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

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

    def create_extensive_report(self):
        import datetime
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from PIL import Image
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        # Ensure output directory is outputs/Cost_Benchmarks
        base_output_dir = os.path.join('outputs', 'Cost_Benchmarks')
        os.makedirs(base_output_dir, exist_ok=True)
        self.output_dir = base_output_dir

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        df = pd.DataFrame(self.records)
        md_lines = []
        md_lines.append(f"# Cost Benchmark Report ({timestamp})\n")
        md_lines.append("## Providers\n")
        for prov in self.providers:
            md_lines.append(f"- **{prov['name']}**: ${prov.get('cost_per_sec', 0):.5f}/sec, throughput ratio: {prov.get('throughput_ratio', 1.0)}")
        md_lines.append("\n## Datasets\n")
        for ds in self.datasets:
            md_lines.append(f"- **{ds['name']}**: metadata: {ds.get('metadata_file','')}, GT: {ds.get('gt_dir','')}, prompts: {ds.get('prompts_file','')}")
        md_lines.append("\n## Results Table\n")
        md_lines.append(df.to_markdown(index=False))

        # Plots
        plot_paths = []
        sns.set(style='whitegrid')
        # Cost per dataset per provider
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x='dataset', y='cost_usd', hue='provider')
        plt.title('Cost per Dataset by Provider')
        plot1 = os.path.join(self.output_dir, f"cost_comparison_{timestamp}.png")
        plt.savefig(plot1)
        plt.close()
        plot_paths.append(plot1)
        md_lines.append(f"\n![Cost per Dataset by Provider]({os.path.basename(plot1)})\n")

        # Cost vs performance (if available)
        metric_key = None
        for k in ['clip', 'psnr', 'ssim', 'fid', 'accuracy']:
            if k in df.columns:
                metric_key = k
                break
        if metric_key:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x='cost_usd', y=metric_key, hue='provider', style='dataset', s=100)
            plt.title(f"Cost vs {metric_key}")
            plot2 = os.path.join(self.output_dir, f"cost_vs_{metric_key}_{timestamp}.png")
            plt.savefig(plot2)
            plt.close()
            plot_paths.append(plot2)
            md_lines.append(f"\n![Cost vs {metric_key}]({os.path.basename(plot2)})\n")

        # Save markdown
        md_report = '\n'.join(md_lines)
        md_path = os.path.join(self.output_dir, f"cost_benchmark_report_{timestamp}.md")
        with open(md_path, 'w') as f:
            f.write(md_report)
        print(f"Saved markdown report to {md_path}")

        # Try to export PDF
        try:
            pdf_path = os.path.join(self.output_dir, f"cost_benchmark_report_{timestamp}.pdf")
            c = canvas.Canvas(pdf_path, pagesize=letter)
            width, height = letter
            textobject = c.beginText(40, height-40)
            for line in md_report.split('\n'):
                textobject.textLine(line)
            c.drawText(textobject)
            # Insert each plot image
            y_cursor = height-400
            for plot in plot_paths:
                if os.path.exists(plot):
                    img = Image.open(plot)
                    img_width, img_height = img.size
                    aspect = img_height / img_width
                    pdf_img_width = width - 80
                    pdf_img_height = pdf_img_width * aspect
                    if y_cursor - pdf_img_height < 40:
                        c.showPage()
                        y_cursor = height-40
                    c.drawInlineImage(plot, 40, y_cursor-pdf_img_height, width=pdf_img_width, height=pdf_img_height)
                    y_cursor -= (pdf_img_height + 40)
            c.showPage()
            c.save()
            print(f"Saved PDF report to {pdf_path}")
        except Exception as e:
            print(f"Could not generate PDF: {e}\nInstall reportlab and pillow for PDF export.")

    def run(self):
        super().run()
        # initialize config attributes
        cfg = self.config
        self.base_config_file = cfg.get('base_config_file')
        self.providers = cfg.get('providers', [])
        self.datasets = cfg.get('datasets', [])
        self.performance_target = cfg.get('performance_target', {})
        self.output_dir = cfg.get('output_dir', 'outputs/Cost_Benchmarks')
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
                os.makedirs(exp_dir, exist_ok=True)
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
