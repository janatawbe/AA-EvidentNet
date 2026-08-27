"""Experiment registry: experiments/registry.csv.

Append-only from the caller's point of view: registering a run adds one
new row; updating a run's status/checkpoint/notes rewrites the file with
that one row changed, every other run's row untouched. A run's
`test_result` column is only ever populated by a future, explicitly
test-evaluation task - Task 6's training runs always leave it empty.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

REGISTRY_COLUMNS = ["experiment_id", "model", "seed", "config", "checkpoint", "test_result", "status", "notes"]

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
VALID_STATUSES = {STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED}

DEFAULT_REGISTRY_PATH = "experiments/registry.csv"


class RegistryError(Exception):
    """Raised for a duplicate experiment_id, an unknown status value, or
    an update targeting a nonexistent run."""


def load_registry(registry_path: Union[str, Path] = DEFAULT_REGISTRY_PATH) -> List[Dict[str, str]]:
    path = Path(registry_path)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_registry(rows: List[Dict[str, Any]], registry_path: Union[str, Path]) -> None:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in REGISTRY_COLUMNS})


def register_run(
    experiment_id: str,
    model: str,
    seed: int,
    config: str,
    checkpoint: str = "",
    test_result: str = "",
    status: str = STATUS_RUNNING,
    notes: str = "",
    registry_path: Union[str, Path] = DEFAULT_REGISTRY_PATH,
) -> None:
    if status not in VALID_STATUSES:
        raise RegistryError(f"Unknown status '{status}' (must be one of {sorted(VALID_STATUSES)})")

    rows = load_registry(registry_path)
    if any(r["experiment_id"] == experiment_id for r in rows):
        raise RegistryError(f"experiment_id '{experiment_id}' is already registered - run IDs must be unique")

    rows.append(
        {
            "experiment_id": experiment_id,
            "model": model,
            "seed": seed,
            "config": config,
            "checkpoint": checkpoint,
            "test_result": test_result,
            "status": status,
            "notes": notes,
        }
    )
    _write_registry(rows, registry_path)


def update_run(
    experiment_id: str,
    registry_path: Union[str, Path] = DEFAULT_REGISTRY_PATH,
    status: Optional[str] = None,
    checkpoint: Optional[str] = None,
    test_result: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    if status is not None and status not in VALID_STATUSES:
        raise RegistryError(f"Unknown status '{status}' (must be one of {sorted(VALID_STATUSES)})")

    rows = load_registry(registry_path)
    for row in rows:
        if row["experiment_id"] == experiment_id:
            if status is not None:
                row["status"] = status
            if checkpoint is not None:
                row["checkpoint"] = checkpoint
            if test_result is not None:
                row["test_result"] = test_result
            if notes is not None:
                row["notes"] = notes
            _write_registry(rows, registry_path)
            return

    raise RegistryError(f"experiment_id '{experiment_id}' not found in {registry_path} - cannot update")
