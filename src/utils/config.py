from pathlib import Path
import yaml

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True); return Path(path)
