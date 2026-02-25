"""
Load dataset definitions from datasets.yaml and resolve paths for the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

logger = __import__("logging").getLogger(__name__)


def load_datasets_config(config_path: Path) -> dict[str, Any]:
    """Load datasets.yaml; return dict with 'datasets' key (name -> config)."""
    if not config_path.is_file():
        return {"datasets": {}}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {"datasets": {}}


def get_dataset_config(
    config_path: Path,
    name: str,
    project_root: Path | None = None,
) -> dict[str, Any] | None:
    """
    Return resolved config for a named dataset.
    Paths in config are used as-is (must be absolute or relative to cwd).
    mlf_root null -> use root for MLF (UASpeech layout).
    """
    data = load_datasets_config(config_path)
    datasets = data.get("datasets") or {}
    cfg = datasets.get(name)
    if not cfg:
        return None
    root = Path(cfg.get("root", "")).expanduser().resolve()
    mlf_root = cfg.get("mlf_root")
    if mlf_root is None or mlf_root == "":
        mlf_root = root
    else:
        mlf_root = Path(mlf_root).expanduser().resolve()
    layout = cfg.get("layout", "uaspeech")
    output_manifest = cfg.get("output_manifest", "metadata.csv")
    return {
        "root": root,
        "mlf_root": mlf_root,
        "layout": layout,
        "output_manifest": output_manifest,
    }


def list_dataset_names(config_path: Path) -> list[str]:
    """Return list of defined dataset names."""
    data = load_datasets_config(config_path)
    datasets = data.get("datasets") or {}
    return list(datasets.keys())
