"""PyTorch dataset layer for AA-EvidentNet, built directly on top of the
manifests produced by Tasks 3-4 (src/data/build_split.py,
src/data/generate_balanced_dataset.py). This module never modifies a
manifest or a raw/processed image — it only loads, validates, and serves
them.

Manifest usage (see README.md for the full rationale):
  - data/manifests/train_original.csv  - original training images only
  - data/manifests/train_balanced.csv  - DEFAULT for model training
                                          (original + augmented, exactly
                                          target_samples_per_class each)
  - data/manifests/val_original.csv    - ALWAYS validation; original only
  - data/manifests/test_original.csv   - ALWAYS test; original only

Path resolution: a row's `path` is relative to raw_dir when
is_original=="true", and relative to processed_train_dir when
is_original=="false" (see resolve_image_path()) - this mirrors the
convention established when train_balanced.csv was written.

Class index mapping: build_class_to_idx() sorts the canonical class names
alphabetically and assigns indices 0..9 in that order. This function is
the single source of truth for the mapping - every dataset, every split,
every model, every experiment must call it (or construct a Dataset, which
calls it internally) rather than deriving an ordering from filesystem
iteration order, config file order, or anything else non-deterministic.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from PIL import Image
from torch.utils.data import Dataset

from src.data.build_split import MANIFEST_COLUMNS

VALID_SPLITS = ("train", "val", "test")
IS_ORIGINAL_VALUES = ("true", "false")


class DatasetManifestError(Exception):
    """Raised when a manifest is missing required columns, contains
    invalid values, or is otherwise unsafe to load. Never silently
    repaired - the caller must fix the manifest (or regenerate it via
    run_pipeline.py prepare_dataset) and re-run."""


def build_class_to_idx(canonical_classes: Iterable[str]) -> Dict[str, int]:
    """The single deterministic canonical_class -> integer label mapping,
    used by every dataset/split/model/experiment. Alphabetical order,
    never filesystem order."""
    return {name: idx for idx, name in enumerate(sorted(canonical_classes))}


def build_idx_to_class(canonical_classes: Iterable[str]) -> Dict[int, str]:
    return {idx: name for name, idx in build_class_to_idx(canonical_classes).items()}


def load_manifest_rows(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load a manifest CSV, failing loudly if required columns are missing.
    Does not validate row VALUES - see validate_manifest_rows() for that.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetManifestError(f"Manifest not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in MANIFEST_COLUMNS if c not in fieldnames]
        if missing:
            raise DatasetManifestError(
                f"{path}: missing required column(s) {missing}. "
                f"Expected columns: {list(MANIFEST_COLUMNS)}, found: {fieldnames}"
            )
        rows = list(reader)

    if not rows:
        raise DatasetManifestError(f"{path}: manifest has a valid header but zero rows")

    return rows


def validate_manifest_rows(
    rows: Sequence[Dict[str, str]],
    canonical_classes: Iterable[str],
    manifest_name: str = "manifest",
    expected_split: Optional[str] = None,
    require_all_original: bool = False,
) -> None:
    """Validate row-level values. Raises DatasetManifestError listing every
    problem found (not just the first) if anything is invalid.

    Checks:
      - class is one of the 10 canonical classes
      - split is a valid value (train/val/test), and matches
        expected_split if given (e.g. every row in val_original.csv must
        have split=="val")
      - is_original is "true" or "false"
      - is_original is consistent with augmentation_type
        ("true" <=> augmentation_type == "original")
      - if require_all_original: every row must be is_original=="true"
        (enforced for validation/test manifests, which must never contain
        an augmented sample)
    """
    canonical_set = set(canonical_classes)
    errors: List[str] = []

    for i, row in enumerate(rows):
        label = f"row {i} ({row.get('path', '?')})"

        canonical_class = row.get("class")
        if canonical_class not in canonical_set:
            errors.append(f"{label}: invalid class '{canonical_class}'")

        split = row.get("split")
        if split not in VALID_SPLITS:
            errors.append(f"{label}: invalid split '{split}' (must be one of {VALID_SPLITS})")
        elif expected_split is not None and split != expected_split:
            errors.append(f"{label}: split '{split}' does not match expected '{expected_split}' for {manifest_name}")

        is_original = row.get("is_original")
        if is_original not in IS_ORIGINAL_VALUES:
            errors.append(f"{label}: invalid is_original '{is_original}' (must be 'true' or 'false')")
        else:
            augmentation_type = row.get("augmentation_type")
            if (is_original == "true") != (augmentation_type == "original"):
                errors.append(
                    f"{label}: is_original={is_original} is inconsistent with "
                    f"augmentation_type='{augmentation_type}'"
                )
            if require_all_original and is_original != "true":
                errors.append(
                    f"{label}: {manifest_name} must contain only original images, "
                    f"found is_original='{is_original}' (augmentation_type='{augmentation_type}')"
                )

    if errors:
        formatted = "\n".join(f"  - {e}" for e in errors[:50])
        more = f"\n  ... and {len(errors) - 50} more" if len(errors) > 50 else ""
        raise DatasetManifestError(f"Invalid rows in {manifest_name}:\n{formatted}{more}")


def resolve_image_path(row: Dict[str, str], raw_dir: Union[str, Path], processed_train_dir: Union[str, Path]) -> Path:
    """A row's `path` is relative to raw_dir when is_original=="true"
    (exactly as in train_original.csv / val_original.csv / test_original.csv),
    and relative to processed_train_dir when is_original=="false" (as
    written by generate_balanced_dataset.py)."""
    base_dir = Path(raw_dir) if row["is_original"] == "true" else Path(processed_train_dir)
    return base_dir / row["path"]


def assert_paths_exist(
    rows: Sequence[Dict[str, str]],
    raw_dir: Union[str, Path],
    processed_train_dir: Union[str, Path],
    manifest_name: str = "manifest",
) -> None:
    missing = []
    for row in rows:
        path = resolve_image_path(row, raw_dir, processed_train_dir)
        if not path.is_file():
            missing.append(str(path))
    if missing:
        formatted = "\n".join(f"  - {p}" for p in missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise DatasetManifestError(f"{manifest_name}: {len(missing)} referenced file(s) do not exist:\n{formatted}{more}")


def assert_no_cross_manifest_overlap(manifests: Dict[str, Sequence[Dict[str, str]]]) -> None:
    """Given e.g. {"train": train_rows, "val": val_rows, "test": test_rows},
    fail loudly if any path or original_id appears in more than one of
    them. This is an independent, dataset-loader-level re-check of the
    same invariant build_split.py already enforces at split-build time."""
    names = list(manifests.keys())
    errors: List[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            paths_a = {r["path"] for r in manifests[name_a]}
            paths_b = {r["path"] for r in manifests[name_b]}
            ids_a = {r["original_id"] for r in manifests[name_a]}
            ids_b = {r["original_id"] for r in manifests[name_b]}
            overlap_paths = paths_a & paths_b
            overlap_ids = ids_a & ids_b
            if overlap_paths or overlap_ids:
                errors.append(
                    f"{name_a} and {name_b} overlap: {len(overlap_paths)} shared path(s), "
                    f"{len(overlap_ids)} shared original_id(s)"
                )
    if errors:
        raise DatasetManifestError("Cross-manifest overlap detected:\n" + "\n".join(f"  - {e}" for e in errors))


def validate_dataset_manifests(
    train_balanced_rows: Sequence[Dict[str, str]],
    val_rows: Sequence[Dict[str, str]],
    test_rows: Sequence[Dict[str, str]],
    canonical_classes: Iterable[str],
) -> None:
    """One-call consistency check across the three manifests actually used
    for model development (train_balanced/val_original/test_original):
    per-manifest value validity, val/test original-only enforcement, and
    no cross-manifest path/ID overlap. Raises DatasetManifestError on any
    violation; never repairs anything."""
    validate_manifest_rows(train_balanced_rows, canonical_classes, manifest_name="train_balanced.csv", expected_split="train")
    validate_manifest_rows(
        val_rows, canonical_classes, manifest_name="val_original.csv", expected_split="val", require_all_original=True
    )
    validate_manifest_rows(
        test_rows, canonical_classes, manifest_name="test_original.csv", expected_split="test", require_all_original=True
    )
    assert_no_cross_manifest_overlap({"train": train_balanced_rows, "val": val_rows, "test": test_rows})


class RetinalDataset(Dataset):
    """A torch Dataset over one manifest (train_original / train_balanced /
    val_original / test_original). Each sample is a dict:

        {
            "image": tensor (after `transform`, or a PIL.Image if transform
                is None),
            "label": int (0..9, via build_class_to_idx),
            "class_name": str,
            "image_path": str (resolved, absolute-or-relative-to-cwd path
                actually opened),
            "original_id": str,
            "parent_original_id": str,
            "is_original": bool,
        }

    This is intentionally rich enough that later evaluation/prediction code
    can reconstruct results/raw_predictions/ rows without touching this
    class again.
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        canonical_classes: Iterable[str],
        raw_dir: Union[str, Path],
        processed_train_dir: Union[str, Path],
        transform=None,
    ):
        self.rows = list(rows)
        self.canonical_classes = sorted(canonical_classes)
        self.class_to_idx = build_class_to_idx(self.canonical_classes)
        self.raw_dir = Path(raw_dir)
        self.processed_train_dir = Path(processed_train_dir)
        self.transform = transform

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        canonical_classes: Iterable[str],
        raw_dir: Union[str, Path],
        processed_train_dir: Union[str, Path],
        transform=None,
        manifest_name: Optional[str] = None,
        expected_split: Optional[str] = None,
        require_all_original: bool = False,
        check_paths_exist: bool = True,
    ) -> "RetinalDataset":
        manifest_path = Path(manifest_path)
        name = manifest_name or manifest_path.name
        rows = load_manifest_rows(manifest_path)
        validate_manifest_rows(
            rows, canonical_classes, manifest_name=name, expected_split=expected_split, require_all_original=require_all_original
        )
        if check_paths_exist:
            assert_paths_exist(rows, raw_dir, processed_train_dir, manifest_name=name)
        return cls(rows, canonical_classes, raw_dir, processed_train_dir, transform=transform)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        image_path = resolve_image_path(row, self.raw_dir, self.processed_train_dir)

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return {
            "image": image,
            "label": self.class_to_idx[row["class"]],
            "class_name": row["class"],
            "image_path": str(image_path),
            "original_id": row["original_id"],
            "parent_original_id": row["parent_original_id"],
            "is_original": row["is_original"] == "true",
        }
