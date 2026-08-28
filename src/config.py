from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    p = Path(path); p = p if p.is_absolute() else ROOT / p
    if not p.is_file(): raise FileNotFoundError(f"Configuration file not found: {p}")
    with p.open(encoding="utf-8") as f: cfg = yaml.safe_load(f) or {}
    for key in ("input", "dataset", "classes", "models", "training", "evaluation"):
        if key not in cfg: raise ValueError(f"Missing configuration section: {key}")
    cfg["classes"] = {int(k): str(v) for k, v in cfg["classes"].items()}
    cfg["_config_path"], cfg["_project_root"] = str(p), str(ROOT)
    return cfg

def public_config(cfg): return {k: v for k, v in deepcopy(cfg).items() if not k.startswith("_")}
def project_path(cfg, value):
    p = Path(value); return p if p.is_absolute() else Path(cfg["_project_root"]) / p
