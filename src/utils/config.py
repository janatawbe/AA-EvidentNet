"""Configuration loading and hashing.

Configs are plain YAML files under configs/. hash_config produces a stable
SHA-256 digest of a config's content so that results/logs can be traced back
to the exact configuration that produced them.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file into a dict.

    Raises:
        FileNotFoundError: if the config file does not exist.
        yaml.YAMLError: if the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def _canonicalize(obj: Any) -> Any:
    """Recursively sort dict keys so hashing is independent of key order."""
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canonicalize(v) for v in obj]
    return obj


def hash_config(config: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hex digest of a config dict.

    Key order does not affect the resulting hash; value order within lists
    does (lists are treated as ordered data).
    """
    canonical = _canonicalize(config)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
