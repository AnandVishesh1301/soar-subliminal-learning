import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru for CLI scripts; route stdlib logging through loguru."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            level_name: str | int
            try:
                level_name = logger.level(record.levelname).name
            except ValueError:
                level_name = record.levelno

            frame = logging.currentframe()
            depth = 0
            while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level_name, record.getMessage()
            )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("transformers", "datasets", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)

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


def get_hf_cache_dir(cfg: dict[str, Any]) -> Path:
    cache_dir = cfg["paths"].get("hf_cache_dir")
    if cache_dir:
        return Path(cache_dir).expanduser()
    return get_data_root(cfg) / "hf_cache"


def resolve_data_path(cfg: dict[str, Any], relative: str) -> Path:
    return get_data_root(cfg) / relative


def get_teacher_model_id(cfg: dict[str, Any]) -> str:
    return cfg["teacher"].get("model_id", cfg["models"]["teacher_base"])


def get_teacher_checkpoint(cfg: dict[str, Any]) -> Path:
    return resolve_data_path(cfg, cfg["teacher"]["output_dir"])


def get_control_model_id(cfg: dict[str, Any]) -> str:
    return cfg["models"].get("control_base", cfg["models"]["student_base"])
