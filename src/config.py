import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

_DATA_SUBDIRS = ("checkpoints", "datasets", "evals", "plots")


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_data_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["data_root"])


def ensure_data_dirs(data_root: Path) -> None:
    for subdir in _DATA_SUBDIRS:
        (data_root / subdir).mkdir(parents=True, exist_ok=True)


def resolve_data_path(cfg: dict[str, Any], relative: str) -> Path:
    return get_data_root(cfg) / relative
