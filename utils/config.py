import yaml
from pathlib import Path

def load_config(config_path='configs/default.yaml'):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(base_config, override_config):
    """Merge two configs, override takes precedence"""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged