"""Per-run structured logging: results/logs/<run_id>/ with run.log,
metrics.jsonl, config.yaml, environment.txt, dataset_hash.txt,
git_commit.txt.

Named `logging.py` to mirror the task's suggested layout; Python 3's
absolute-import semantics mean `import logging` elsewhere always resolves
to the standard library, never to this module, so there is no shadowing
risk. This module does not use the stdlib `logging` package itself - it
implements a small dedicated RunLogger instead, since training runs need a
combined human-readable log + a machine-readable metrics stream tied to a
single run directory.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

RUN_LOG_FILENAME = "run.log"
METRICS_FILENAME = "metrics.jsonl"
CONFIG_FILENAME = "config.yaml"
ENVIRONMENT_FILENAME = "environment.txt"
DATASET_HASH_FILENAME = "dataset_hash.txt"
GIT_COMMIT_FILENAME = "git_commit.txt"


def generate_run_id(model_name: str, seed: int, smoke_test: bool = False) -> str:
    """YYYYMMDD_HHMMSS_model_seed<seed>[_smoke]_<6-hex> - the trailing
    random suffix guards against collisions when two runs start within the
    same second (e.g. scripted multi-seed launches), since the timestamp
    alone is not guaranteed unique."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    smoke_tag = "_smoke" if smoke_test else ""
    return f"{timestamp}_{model_name}_seed{seed}{smoke_tag}_{suffix}"


class RunLogger:
    """Owns one results/logs/<run_id>/ directory. Never overwrites an
    existing run directory - each run_id must be unique (see
    generate_run_id)."""

    def __init__(self, run_dir: Union[str, Path]):
        self.run_dir = Path(run_dir)
        if self.run_dir.exists():
            raise FileExistsError(f"Run directory already exists, refusing to overwrite: {self.run_dir}")
        self.run_dir.mkdir(parents=True)
        self._log_path = self.run_dir / RUN_LOG_FILENAME
        self._metrics_path = self.run_dir / METRICS_FILENAME
        self._log_file = open(self._log_path, "a", encoding="utf-8")
        self._metrics_file = open(self._metrics_path, "a", encoding="utf-8")

    def log(self, message: str, print_to_stdout: bool = True) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}"
        if print_to_stdout:
            print(line)
        self._log_file.write(line + "\n")
        self._log_file.flush()

    def log_metrics(self, record: Dict[str, Any]) -> None:
        self._metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
        self._metrics_file.flush()

    def write_config(self, config: Dict[str, Any]) -> None:
        with open(self.run_dir / CONFIG_FILENAME, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    def write_environment(self, env_info: Dict[str, Any]) -> None:
        with open(self.run_dir / ENVIRONMENT_FILENAME, "w", encoding="utf-8") as f:
            for key in sorted(env_info.keys()):
                f.write(f"{key}: {env_info[key]}\n")

    def write_dataset_hash(self, dataset_hash: str) -> None:
        (self.run_dir / DATASET_HASH_FILENAME).write_text(dataset_hash + "\n", encoding="utf-8")

    def write_git_commit(self, git_commit: Optional[str]) -> None:
        (self.run_dir / GIT_COMMIT_FILENAME).write_text(f"{git_commit}\n", encoding="utf-8")

    def close(self) -> None:
        self._log_file.close()
        self._metrics_file.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
