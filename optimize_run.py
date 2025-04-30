#!/usr/bin/env python3
"""
Script to run hyperparameter optimization using the OptimizeJob.
"""
import yaml
from jobs.OptimizeJob import OptimizeJob

# Paths to configs
OPTIMIZE_CFG_PATH = "config/optimize_params.yaml"
BASE_CFG_PATH = "config/schnell_config.yaml"

# Load optimization ranges
with open(OPTIMIZE_CFG_PATH, 'r') as f:
    data = yaml.safe_load(f)
opt_cfg = data.get('optimize', {})

# Load base training config
from toolkit.config import get_config
base_full = get_config(BASE_CFG_PATH)
base_cfg = base_full['config']

# Construct full config for OptimizeJob
full_cfg = {
    'job': 'optimize',
    'config': {
        'name': 'flux_schnell_optimize',
        'optimize': opt_cfg,
        'base_config': base_cfg
    }
}

# Run optimization
job = OptimizeJob(full_cfg)
job.run()
job.cleanup()  
