"""Config loading and output paths shared by all pipeline stages."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=ROOT / "config.json"):
    with open(path) as f:
        cfg = json.load(f)
    cfg["_config_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def out_dir(cfg, sub):
    d = ROOT / cfg["output_dir"] / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def dates(start, end):
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
