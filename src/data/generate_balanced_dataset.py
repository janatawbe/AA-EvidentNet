"""Balanced training-set generation: expand data/manifests/train_original.csv
up to `target_samples_per_class` per class via augmentation, producing
data/manifests/train_balanced.csv.

Pipeline position:

    ... -> ORIGINAL 70/20/10 SPLIT (build_split.py)
        -> THIS MODULE: balanced training set (train ONLY)
        -> future model training (not yet implemented)

Absolute rules enforced throughout this module (see also README.md):
  - Only rows from train_original.csv may ever be augmentation PARENTS.
    val_original.csv / test_original.csv are loaded ONLY to run an explicit
    negative-containment check (assert_no_val_test_contamination) — never
    to select parents, never as a generation source.
  - A parent must be an ORIGINAL image (is_original=true). An already
    generated (augmented) image can never become a parent — no recursive
    augmentation.
  - data/raw/ is only ever opened for reading (to load a parent image);
    nothing under data/raw/, data/manifests/val_original.csv, or
    data/manifests/test_original.csv is ever written to.
  - Generated files are written only under data/processed/train/.

Manifest path convention (train_balanced.csv): a row's `path` is relative
to raw_dir when is_original=true (exactly as in train_original.csv), and
relative to processed_train_dir when is_original=false. The `is_original`
column is what tells a reader which base directory to prepend — this
avoids inventing an extra column while keeping both provenance and
storage explicit.

Deterministic generated IDs: see compute_generated_id(). Deterministic
per-sample augmentation randomness: see _sample_rng() — each generated
sample gets its own random.Random seeded from
(parent_original_id, canonical_class, augmentation_index, seed,
augmentation_config_hash), so the entire generation is independent of
processing order and fully reproducible from those five stable values.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageEnhance

from src.data.build_split import (
    AUGMENTATION_TYPE_ORIGINAL,
    IS_ORIGINAL_TRUE,
    MANIFEST_COLUMNS,
    SPLIT_NAMES,
    CheckResult,
)
from src.data.eligibility import load_eligibility_csv
from src.data.records import write_csv
from src.utils.config import hash_config, load_config
from src.utils.env_info import collect_environment_info
from src.utils.git_info import get_git_commit
from src.utils.hashing import hash_file, hash_string
from src.utils.seeding import DEFAULT_SEED, set_seed

IS_ORIGINAL_FALSE = "false"

AUG_HORIZONTAL_FLIP = "horizontal_flip"
AUG_ROTATION = "rotation"
AUG_BRIGHTNESS_CONTRAST = "brightness_contrast"
AUG_AFFINE = "affine"
AUG_COLOR_JITTER = "color_jitter"
AUG_COMBINED = "combined"

RECIPE_ORDER = [
    AUG_HORIZONTAL_FLIP,
    AUG_ROTATION,
    AUG_BRIGHTNESS_CONTRAST,
    AUG_AFFINE,
    AUG_COLOR_JITTER,
]

DEFAULT_AUGMENTATION_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "target_samples_per_class": 2000,
    "horizontal_flip": {"enabled": True, "probability": 0.5},
    "rotation": {"enabled": True, "degrees": 10},
    "brightness": {"enabled": True, "factor": 0.15},
    "contrast": {"enabled": True, "factor": 0.15},
    "affine": {"enabled": True, "translate": 0.05, "scale": {"min": 0.95, "max": 1.05}},
    "color_jitter": {"enabled": True, "factor": 0.10},
}

AUGMENTATION_STATISTICS_COLUMNS = [
    "class",
    "original_count",
    "generated_count",
    "total_count",
    "unique_parent_count",
    "mean_generated_per_parent",
    "max_generated_per_parent",
    "augmentation_type",
    "augmentation_count",
]


class BalancedDatasetBuildError(Exception):
    """Fatal, unrecoverable problem building the balanced training set
    (missing train_original.csv, a class with more originals than the
    target, no augmentation recipe available while augmentation is
    needed, val/test contamination detected, etc.)."""


class BalancedDatasetValidationError(Exception):
    """The balanced manifest was built but failed a mandatory integrity
    check (see validate_balanced_manifest)."""


def compute_generated_id(
    parent_original_id: str,
    canonical_class: str,
    augmentation_index: int,
    seed: int,
    augmentation_config_hash: str,
) -> str:
    """Deterministic ID for a generated sample.

    sha256("<parent_original_id>|<canonical_class>|<augmentation_index>|<seed>|<augmentation_config_hash>")
    Depends only on stable, reproducible inputs — never a random UUID, an
    absolute path, or a timestamp — so the same original manifest + seed +
    augmentation config always regenerates the same IDs.
    """
    return hash_string(
        f"{parent_original_id}|{canonical_class}|{augmentation_index}|{seed}|{augmentation_config_hash}"
    )


def _sample_rng(
    parent_original_id: str,
    canonical_class: str,
    augmentation_index: int,
    seed: int,
    augmentation_config_hash: str,
) -> random.Random:
    """A dedicated RNG for one generated sample's augmentation parameters,
    seeded from the same stable tuple as compute_generated_id(). This makes
    generation independent of iteration/processing order: any two runs
    with the same inputs produce identical augmentation parameters for a
    given (parent, class, augmentation_index), regardless of what else ran
    before or after."""
    seed_str = f"rng|{parent_original_id}|{canonical_class}|{augmentation_index}|{seed}|{augmentation_config_hash}"
    return random.Random(seed_str)


# --- Augmentation operators (Pillow-based; deterministic given `rng`) ------
#
# Every operator always visibly changes the image (never returns pixels
# identical to the input) so that a generated sample is never a disguised
# duplicate. Parameter ranges avoid a dead zone around the identity value.


def apply_horizontal_flip(image: Image.Image, rng: random.Random, cfg: Dict[str, Any]) -> Image.Image:
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def apply_rotation(image: Image.Image, rng: random.Random, cfg: Dict[str, Any]) -> Image.Image:
    degrees = cfg.get("rotation", {}).get("degrees", 10)
    angle = rng.uniform(-degrees, degrees)
    if -1.0 < angle < 1.0:
        angle = 1.0 if rng.random() < 0.5 else -1.0
    return image.rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))


def apply_brightness_contrast(image: Image.Image, rng: random.Random, cfg: Dict[str, Any]) -> Image.Image:
    b_range = cfg.get("brightness", {}).get("factor", 0.15)
    c_range = cfg.get("contrast", {}).get("factor", 0.15)
    b_factor = 1.0 + rng.uniform(-b_range, b_range)
    c_factor = 1.0 + rng.uniform(-c_range, c_range)
    image = ImageEnhance.Brightness(image).enhance(b_factor)
    image = ImageEnhance.Contrast(image).enhance(c_factor)
    return image


def apply_affine(image: Image.Image, rng: random.Random, cfg: Dict[str, Any]) -> Image.Image:
    affine_cfg = cfg.get("affine", {})
    translate_frac = affine_cfg.get("translate", 0.05)
    scale_cfg = affine_cfg.get("scale", {"min": 0.95, "max": 1.05})

    tx = rng.uniform(-translate_frac, translate_frac) * image.width
    ty = rng.uniform(-translate_frac, translate_frac) * image.height
    scale = rng.uniform(scale_cfg.get("min", 0.95), scale_cfg.get("max", 1.05))
    if abs(scale - 1.0) < 0.01:
        scale = 1.01 if rng.random() < 0.5 else 0.99

    # PIL's AFFINE transform maps *output* coordinates to *input*
    # coordinates, so we invert the scale/translate here.
    inv_scale = 1.0 / scale
    matrix = (inv_scale, 0.0, -tx * inv_scale, 0.0, inv_scale, -ty * inv_scale)
    return image.transform(image.size, Image.AFFINE, matrix, resample=Image.BICUBIC, fillcolor=(0, 0, 0))


def apply_color_jitter(image: Image.Image, rng: random.Random, cfg: Dict[str, Any]) -> Image.Image:
    factor_range = cfg.get("color_jitter", {}).get("factor", 0.10)
    factor = 1.0 + rng.uniform(-factor_range, factor_range)
    return ImageEnhance.Color(image).enhance(factor)


def apply_combined(image: Image.Image, rng: random.Random, cfg: Dict[str, Any]) -> Image.Image:
    if cfg.get("horizontal_flip", {}).get("enabled", True):
        probability = cfg.get("horizontal_flip", {}).get("probability", 0.5)
        if rng.random() < probability:
            image = apply_horizontal_flip(image, rng, cfg)
    if cfg.get("rotation", {}).get("enabled", True):
        image = apply_rotation(image, rng, cfg)
    if cfg.get("brightness", {}).get("enabled", True) or cfg.get("contrast", {}).get("enabled", True):
        image = apply_brightness_contrast(image, rng, cfg)
    if cfg.get("affine", {}).get("enabled", True):
        image = apply_affine(image, rng, cfg)
    if cfg.get("color_jitter", {}).get("enabled", True):
        image = apply_color_jitter(image, rng, cfg)
    return image


RECIPES = {
    AUG_HORIZONTAL_FLIP: apply_horizontal_flip,
    AUG_ROTATION: apply_rotation,
    AUG_BRIGHTNESS_CONTRAST: apply_brightness_contrast,
    AUG_AFFINE: apply_affine,
    AUG_COLOR_JITTER: apply_color_jitter,
    AUG_COMBINED: apply_combined,
}


def get_active_recipes(cfg: Dict[str, Any]) -> List[str]:
    """Named recipes whose underlying transform(s) are enabled, in a fixed
    order (round-robin assignment relies on this order being stable).
    "combined" is included whenever at least one solo transform is active.
    """
    enabled_by_name = {
        AUG_HORIZONTAL_FLIP: cfg.get("horizontal_flip", {}).get("enabled", True),
        AUG_ROTATION: cfg.get("rotation", {}).get("enabled", True),
        AUG_BRIGHTNESS_CONTRAST: (
            cfg.get("brightness", {}).get("enabled", True) or cfg.get("contrast", {}).get("enabled", True)
        ),
        AUG_AFFINE: cfg.get("affine", {}).get("enabled", True),
        AUG_COLOR_JITTER: cfg.get("color_jitter", {}).get("enabled", True),
    }
    active = [name for name in RECIPE_ORDER if enabled_by_name[name]]
    if active:
        active.append(AUG_COMBINED)
    return active


def assert_no_val_test_contamination(
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
) -> None:
    """Explicit, mandatory safety check: no path or original_id used as a
    training parent may also appear in val/test. This should be structurally
    impossible given build_split.py's own leakage checks, but Task 4 requires
    an independent, explicit re-check here as well."""
    train_paths = {r["path"] for r in train_rows}
    train_ids = {r["original_id"] for r in train_rows}

    for name, rows in (("validation", val_rows), ("test", test_rows)):
        overlap_paths = train_paths & {r["path"] for r in rows}
        overlap_ids = train_ids & {r["original_id"] for r in rows}
        if overlap_paths or overlap_ids:
            raise BalancedDatasetBuildError(
                f"Data isolation violation: {len(overlap_paths)} path(s) / {len(overlap_ids)} "
                f"original_id(s) appear in BOTH train_original.csv and {name}_original.csv. "
                "Refusing to generate augmentation parents from contaminated data."
            )


@dataclass(frozen=True)
class GeneratedFile:
    generated_id: str
    canonical_class: str
    output_path: Path
    parent_path: Path


def generate_records_for_class(
    canonical_class: str,
    original_rows: Sequence[Dict[str, Any]],
    target: int,
    seed: int,
    augmentation_config: Dict[str, Any],
    augmentation_config_hash: str,
    raw_dir: Path,
    processed_train_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[GeneratedFile]]:
    """Build this class's manifest rows (all originals + however many
    generated rows are needed) and write the generated image files.

    Parent sampling strategy: round-robin over the class's original rows
    (sorted by path for determinism), paired with round-robin over the
    active augmentation recipes — so generation is spread evenly across
    parents and transform types rather than exhausting one parent or one
    recipe before moving to the next. When `needed` exceeds the number of
    originals, parents are naturally reused (repeated selection), which is
    expected and documented, not an error.
    """
    original_count = len(original_rows)
    needed = target - original_count

    if needed < 0:
        raise BalancedDatasetBuildError(
            f"class '{canonical_class}' already has {original_count} original training images, "
            f"more than the target of {target}. Cannot shrink original training data to fit a "
            "target - the balancing methodology only ever adds augmented samples."
        )

    manifest_rows = [dict(row) for row in original_rows]
    if needed == 0:
        return manifest_rows, []

    if not original_rows:
        raise BalancedDatasetBuildError(
            f"class '{canonical_class}' has zero original training images but needs {needed} "
            "generated samples - there is no valid parent image to augment from."
        )

    active_recipes = get_active_recipes(augmentation_config)
    if not active_recipes:
        raise BalancedDatasetBuildError(
            "augmentation is required (needed > 0 for at least one class) but no augmentation "
            "recipes are enabled in configs/dataset.yaml: augmentation."
        )

    sorted_originals = sorted(original_rows, key=lambda r: r["path"])
    class_dir = processed_train_dir / canonical_class

    generated_rows: List[Dict[str, Any]] = []
    generated_files: List[GeneratedFile] = []

    for augmentation_index in range(needed):
        parent_row = sorted_originals[augmentation_index % len(sorted_originals)]
        recipe_name = active_recipes[augmentation_index % len(active_recipes)]
        parent_original_id = parent_row["original_id"]

        generated_id = compute_generated_id(
            parent_original_id, canonical_class, augmentation_index, seed, augmentation_config_hash
        )
        rng = _sample_rng(parent_original_id, canonical_class, augmentation_index, seed, augmentation_config_hash)

        source_path = raw_dir / parent_row["path"]
        output_path = class_dir / f"{generated_id}.jpg"

        with Image.open(source_path) as img:
            rgb_image = img.convert("RGB")
            augmented = RECIPES[recipe_name](rgb_image, rng, augmentation_config)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            augmented.save(output_path, format="JPEG", quality=95)

        generated_rows.append(
            {
                "path": f"{canonical_class}/{generated_id}.jpg",
                "class": canonical_class,
                "split": "train",
                "original_id": generated_id,
                "parent_original_id": parent_original_id,
                "is_original": IS_ORIGINAL_FALSE,
                "augmentation_type": recipe_name,
            }
        )
        generated_files.append(
            GeneratedFile(
                generated_id=generated_id,
                canonical_class=canonical_class,
                output_path=output_path,
                parent_path=source_path,
            )
        )

    return manifest_rows + generated_rows, generated_files


def validate_balanced_manifest(
    balanced_rows: Sequence[Dict[str, Any]],
    train_original_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    raw_eligibility_rows: Sequence[Dict[str, Any]],
    canonical_classes: Iterable[str],
    target_per_class: int,
    raw_dir: Optional[Path] = None,
    processed_train_dir: Optional[Path] = None,
) -> List[CheckResult]:
    """Run every mandatory balanced-manifest integrity check (see module
    docstring / README for the full list). Reusable by later tasks; never
    auto-fixes anything, only reports."""
    results: List[CheckResult] = []
    canonical_classes = list(canonical_classes)
    expected_total = target_per_class * len(canonical_classes)

    results.append(
        CheckResult(
            "total_row_count",
            "PASS" if len(balanced_rows) == expected_total else "FAIL",
            f"{len(balanced_rows)} rows (expected {expected_total})",
        )
    )

    class_counts = Counter(r["class"] for r in balanced_rows)
    bad_classes = {c: class_counts.get(c, 0) for c in canonical_classes if class_counts.get(c, 0) != target_per_class}
    results.append(
        CheckResult(
            "per_class_exact_target",
            "PASS" if not bad_classes else "FAIL",
            f"classes not at exactly {target_per_class}: {bad_classes}" if bad_classes else "all classes exact",
        )
    )

    val_paths = {r["path"] for r in val_rows}
    val_ids = {r["original_id"] for r in val_rows}
    val_leaks = [r["path"] for r in balanced_rows if r["path"] in val_paths or r["original_id"] in val_ids]
    results.append(
        CheckResult(
            "no_validation_images",
            "PASS" if not val_leaks else "FAIL",
            f"{len(val_leaks)} validation image(s) found in balanced manifest" if val_leaks else "none found",
        )
    )

    test_paths = {r["path"] for r in test_rows}
    test_ids = {r["original_id"] for r in test_rows}
    test_leaks = [r["path"] for r in balanced_rows if r["path"] in test_paths or r["original_id"] in test_ids]
    results.append(
        CheckResult(
            "no_test_images",
            "PASS" if not test_leaks else "FAIL",
            f"{len(test_leaks)} test image(s) found in balanced manifest" if test_leaks else "none found",
        )
    )

    original_ids_in_train = {r["original_id"] for r in train_original_rows}
    generated = [r for r in balanced_rows if r["is_original"] == IS_ORIGINAL_FALSE]

    no_parent = [r["original_id"] for r in generated if not r.get("parent_original_id")]
    results.append(
        CheckResult(
            "generated_has_parent",
            "PASS" if not no_parent else "FAIL",
            f"{len(no_parent)} generated row(s) with no parent_original_id" if no_parent else "all generated rows have a parent",
        )
    )

    bad_parent = [r["original_id"] for r in generated if r.get("parent_original_id") not in original_ids_in_train]
    results.append(
        CheckResult(
            "parent_is_original_training_image",
            "PASS" if not bad_parent else "FAIL",
            f"{len(bad_parent)} generated row(s) whose parent is not an original training image "
            "(possible recursive augmentation)" if bad_parent else "every parent is an original training image",
        )
    )

    by_id = {r["original_id"]: r for r in balanced_rows}
    class_mismatches = [
        r["original_id"]
        for r in generated
        if r["parent_original_id"] in by_id and by_id[r["parent_original_id"]]["class"] != r["class"]
    ]
    results.append(
        CheckResult(
            "generated_class_matches_parent",
            "PASS" if not class_mismatches else "FAIL",
            f"{len(class_mismatches)} generated row(s) with a class differing from their parent"
            if class_mismatches
            else "all generated classes match their parent",
        )
    )

    wrong_split = [r["original_id"] for r in balanced_rows if r["split"] != "train"]
    results.append(
        CheckResult(
            "split_is_train",
            "PASS" if not wrong_split else "FAIL",
            f"{len(wrong_split)} row(s) with split != 'train'" if wrong_split else "all rows are split=train",
        )
    )

    balanced_ids = {r["original_id"] for r in balanced_rows if r["is_original"] == IS_ORIGINAL_TRUE}
    missing_originals = original_ids_in_train - balanced_ids
    results.append(
        CheckResult(
            "all_originals_retained",
            "PASS" if not missing_originals else "FAIL",
            f"{len(missing_originals)} original training image(s) missing from balanced manifest"
            if missing_originals
            else "every original training image is retained",
        )
    )

    id_counts = Counter(r["original_id"] for r in balanced_rows)
    dupes = [i for i, c in id_counts.items() if c > 1]
    results.append(
        CheckResult(
            "no_duplicate_ids",
            "PASS" if not dupes else "FAIL",
            f"{len(dupes)} duplicate original_id(s)" if dupes else "no duplicate IDs",
        )
    )

    flag_mismatches = [
        r["original_id"]
        for r in balanced_rows
        if (r["is_original"] == IS_ORIGINAL_TRUE) != (r["augmentation_type"] == AUGMENTATION_TYPE_ORIGINAL)
    ]
    results.append(
        CheckResult(
            "is_original_flag_consistency",
            "PASS" if not flag_mismatches else "FAIL",
            f"{len(flag_mismatches)} row(s) where is_original does not match augmentation_type=='original'"
            if flag_mismatches
            else "is_original is consistent with augmentation_type for every row",
        )
    )

    if processed_train_dir is not None:
        missing_files = []
        unreadable = []
        for r in generated:
            # r["path"] is "<class>/<generated_id>.jpg", relative to processed_train_dir.
            file_path = processed_train_dir / r["path"]
            if not file_path.is_file():
                missing_files.append(r["path"])
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img2:
                    img2.load()
            except Exception:  # noqa: BLE001
                unreadable.append(r["path"])
        results.append(
            CheckResult(
                "generated_files_exist",
                "PASS" if not missing_files else "FAIL",
                f"{len(missing_files)} generated file(s) missing on disk" if missing_files else "all generated files exist",
            )
        )
        results.append(
            CheckResult(
                "generated_files_readable",
                "PASS" if not unreadable else "FAIL",
                f"{len(unreadable)} generated file(s) unreadable/corrupt" if unreadable else "all generated files are readable",
            )
        )
    else:
        results.append(CheckResult("generated_files_exist", "PASS", "skipped (no processed_train_dir given)"))
        results.append(CheckResult("generated_files_readable", "PASS", "skipped (no processed_train_dir given)"))

    if raw_dir is not None:
        eligibility_by_path = {r["path"]: r["sha256"] for r in raw_eligibility_rows}
        mismatches = []
        for r in balanced_rows:
            if r["is_original"] != IS_ORIGINAL_TRUE:
                continue
            recorded_sha256 = eligibility_by_path.get(r["path"])
            if recorded_sha256 is None:
                continue
            file_path = raw_dir / r["path"]
            if not file_path.is_file():
                mismatches.append(r["path"])
                continue
            if hash_file(file_path) != recorded_sha256:
                mismatches.append(r["path"])
        results.append(
            CheckResult(
                "raw_dataset_unmodified",
                "PASS" if not mismatches else "FAIL",
                f"{len(mismatches)} original path(s) missing or changed since the audit" if mismatches else "raw_dir matches the audited state",
            )
        )
    else:
        results.append(CheckResult("raw_dataset_unmodified", "PASS", "skipped (no raw_dir given)"))

    return results


def build_augmentation_statistics_rows(
    canonical_class: str, original_rows: Sequence[Dict[str, Any]], generated_rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    original_count = len(original_rows)
    generated_count = len(generated_rows)
    total_count = original_count + generated_count

    parent_counts = Counter(r["parent_original_id"] for r in generated_rows)
    unique_parent_count = len(parent_counts)
    mean_generated_per_parent = round(generated_count / unique_parent_count, 4) if unique_parent_count else 0.0
    max_generated_per_parent = max(parent_counts.values()) if parent_counts else 0

    type_counts = Counter(r["augmentation_type"] for r in generated_rows)

    base = {
        "class": canonical_class,
        "original_count": original_count,
        "generated_count": generated_count,
        "total_count": total_count,
        "unique_parent_count": unique_parent_count,
        "mean_generated_per_parent": mean_generated_per_parent,
        "max_generated_per_parent": max_generated_per_parent,
    }

    if not type_counts:
        return [{**base, "augmentation_type": "", "augmentation_count": 0}]

    return [
        {**base, "augmentation_type": aug_type, "augmentation_count": type_counts[aug_type]}
        for aug_type in sorted(type_counts.keys())
    ]


@dataclass
class BalancedDatasetSummary:
    seed: int
    target_per_class: int
    balanced_rows: List[Dict[str, Any]]
    manifest_path: Path
    manifest_hash: str
    check_results: List[CheckResult]
    per_class_counts: Dict[str, Dict[str, int]]
    source_manifest_hash: str
    configuration_hash: str
    git_commit: Optional[str]
    metadata_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        return len(self.balanced_rows)

    def format_report(self) -> str:
        lines = [
            "=" * 70,
            "BALANCED TRAINING SET SUMMARY (train split only)",
            "=" * 70,
            f"Seed: {self.seed}",
            f"Target samples per class: {self.target_per_class}",
            f"Git commit: {self.git_commit}",
            f"Total balanced samples: {self.total_samples}",
            "",
            "Per-class (original / generated / total):",
        ]
        for canonical_class in sorted(self.per_class_counts.keys()):
            c = self.per_class_counts[canonical_class]
            lines.append(f"  {canonical_class}: {c['original']} / {c['generated']} / {c['total']}")

        lines.append("")
        lines.append("Balanced manifest integrity checks:")
        for check in self.check_results:
            lines.append(f"  [{check.status}] {check.check}: {check.details}")

        lines.append("")
        lines.append(f"train_balanced.csv -> {self.manifest_path}")
        lines.append(f"train_balanced.csv sha256: {self.manifest_hash}")
        lines.append("=" * 70)
        return "\n".join(lines)


def _load_manifest_csv(path: Path) -> List[Dict[str, Any]]:
    import csv

    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_generate_balanced_dataset(
    config_path: Union[str, Path] = "configs/dataset.yaml",
    seed: int = DEFAULT_SEED,
    raw_dir_override: Optional[Union[str, Path]] = None,
    audit_dir_override: Optional[Union[str, Path]] = None,
    manifests_dir_override: Optional[Union[str, Path]] = None,
    processed_dir_override: Optional[Union[str, Path]] = None,
) -> BalancedDatasetSummary:
    """Build, validate, and write the balanced training manifest.

    Read-only with respect to data/raw/, data/manifests/val_original.csv,
    and data/manifests/test_original.csv. Writes generated images only
    under processed_dir/train/, and the manifest only to
    manifests_dir/train_balanced.csv.

    Raises:
        BalancedDatasetBuildError: fatal, unrecoverable problem (missing
            train_original.csv, data isolation violation, a class that
            cannot be balanced).
        BalancedDatasetValidationError: the manifest was built but failed
            a mandatory integrity check.
    """
    set_seed(seed)

    config = load_config(config_path)
    class_directory_mapping = config["class_directory_mapping"]
    canonical_classes = sorted(class_directory_mapping.keys())

    raw_dir = Path(raw_dir_override) if raw_dir_override is not None else Path(config["paths"]["raw_dir"])
    audit_dir = Path(audit_dir_override) if audit_dir_override is not None else Path(config["paths"]["audit_dir"])
    manifests_dir = (
        Path(manifests_dir_override) if manifests_dir_override is not None else Path(config["paths"]["manifests_dir"])
    )
    processed_dir = (
        Path(processed_dir_override) if processed_dir_override is not None else Path(config["paths"]["processed_dir"])
    )
    processed_train_dir = processed_dir / "train"

    train_path = manifests_dir / "train_original.csv"
    val_path = manifests_dir / "val_original.csv"
    test_path = manifests_dir / "test_original.csv"

    if not train_path.is_file():
        raise BalancedDatasetBuildError(
            f"{train_path} not found. Run `python run_pipeline.py prepare_dataset` (split stage) "
            "first to generate the original train/val/test manifests."
        )

    train_original_rows = _load_manifest_csv(train_path)
    val_rows = _load_manifest_csv(val_path)
    test_rows = _load_manifest_csv(test_path)

    assert_no_val_test_contamination(train_original_rows, val_rows, test_rows)

    augmentation_config = {**DEFAULT_AUGMENTATION_CONFIG, **(config.get("augmentation", {}) or {})}
    if not augmentation_config.get("enabled", True):
        raise BalancedDatasetBuildError(
            "configs/dataset.yaml: augmentation.enabled is false - balanced training set "
            "generation requires augmentation to be enabled."
        )
    target_per_class = int(
        augmentation_config.get("target_samples_per_class", config.get("target_train_samples_per_class", 2000))
    )
    augmentation_config_hash = hash_config(augmentation_config)

    rows_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in train_original_rows:
        rows_by_class[row["class"]].append(row)

    all_balanced_rows: List[Dict[str, Any]] = []
    per_class_counts: Dict[str, Dict[str, int]] = {}
    all_generated_files: List[GeneratedFile] = []
    stats_rows: List[Dict[str, Any]] = []

    for canonical_class in canonical_classes:
        original_rows = rows_by_class.get(canonical_class, [])
        class_rows, generated_files = generate_records_for_class(
            canonical_class,
            original_rows,
            target_per_class,
            seed,
            augmentation_config,
            augmentation_config_hash,
            raw_dir,
            processed_train_dir,
        )
        generated_rows = [r for r in class_rows if r["is_original"] == IS_ORIGINAL_FALSE]
        all_balanced_rows.extend(class_rows)
        all_generated_files.extend(generated_files)
        per_class_counts[canonical_class] = {
            "original": len(original_rows),
            "generated": len(generated_rows),
            "total": len(class_rows),
        }
        stats_rows.extend(build_augmentation_statistics_rows(canonical_class, original_rows, generated_rows))

    all_balanced_rows.sort(key=lambda r: (r["class"], r["is_original"] != IS_ORIGINAL_TRUE, r["path"]))

    raw_eligibility_rows = load_eligibility_csv(audit_dir / "dataset_eligibility.csv")

    check_results = validate_balanced_manifest(
        all_balanced_rows,
        train_original_rows,
        val_rows,
        test_rows,
        raw_eligibility_rows,
        canonical_classes,
        target_per_class,
        raw_dir=raw_dir,
        processed_train_dir=processed_train_dir,
    )
    failures = [c for c in check_results if c.status == "FAIL"]
    if failures:
        raise BalancedDatasetValidationError(
            "Balanced manifest failed mandatory integrity checks:\n"
            + "\n".join(f"  - {c.check}: {c.details}" for c in failures)
        )

    manifests_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_dir / "train_balanced.csv"
    write_csv(all_balanced_rows, MANIFEST_COLUMNS, manifest_path)
    manifest_hash = hash_file(manifest_path)
    (audit_dir / "train_balanced_manifest_hash.txt").write_text(manifest_hash + "\n", encoding="utf-8")

    write_csv(stats_rows, AUGMENTATION_STATISTICS_COLUMNS, audit_dir / "augmentation_statistics.csv")

    source_manifest_hash = hash_file(train_path)
    config_hash = hash_config(config)
    git_commit = get_git_commit()

    metadata = {
        "seed": seed,
        "target_samples_per_class": target_per_class,
        "total_samples": len(all_balanced_rows),
        "class_counts": {c: per_class_counts[c]["total"] for c in canonical_classes},
        "original_training_count": sum(v["original"] for v in per_class_counts.values()),
        "generated_count": sum(v["generated"] for v in per_class_counts.values()),
        "source_manifest_hash": source_manifest_hash,
        "configuration_hash": config_hash,
        "augmentation_configuration_hash": augmentation_config_hash,
        "git_commit": git_commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment_info(),
    }
    metadata_path = audit_dir / "balanced_dataset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    summary = BalancedDatasetSummary(
        seed=seed,
        target_per_class=target_per_class,
        balanced_rows=all_balanced_rows,
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        check_results=check_results,
        per_class_counts=per_class_counts,
        source_manifest_hash=source_manifest_hash,
        configuration_hash=config_hash,
        git_commit=git_commit,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    print(summary.format_report())
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.generate_balanced_dataset",
        description="Generate the balanced (2000/class) training manifest via augmentation.",
    )
    parser.add_argument("--config", type=str, default="configs/dataset.yaml")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Accepted for CLI consistency with run_pipeline.py; generation is currently single-threaded.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_generate_balanced_dataset(config_path=args.config, seed=args.seed)
    except (BalancedDatasetBuildError, BalancedDatasetValidationError) as e:
        print(f"[generate_balanced_dataset] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
