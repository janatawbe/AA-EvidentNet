"""Heuristic detection of filenames/directories suggesting the raw dataset
contains non-original (augmented/generated/duplicated) image variants.

The raw dataset is expected to contain ORIGINAL images only. This module
flags files for HUMAN REVIEW — it never deletes, moves, renames, or
excludes anything automatically.

Detection combines two signal types:
  1. Whole-token keyword matches (configurable, see configs/dataset.yaml:
     audit.augmentation_keywords) against the filename stem and the
     immediate parent directory name. Filenames are split into tokens on
     any non-alphanumeric character (so "_", "-", " ", "(", ")", "."  all
     act as separators) and each token is compared for an EXACT
     (case-insensitive) match against the keyword set. This is
     deliberately not a substring/regex-word-boundary check: "_" and "-"
     are common filename separators but are NOT word-boundary characters
     in regex (`\\b` treats "_" as a word character), and a naive
     substring check would also incorrectly flag things like "Copyright"
     for containing "copy". A match here is strong, direct evidence (e.g.
     "..._flipped.jpg") -> classified "highly-suspicious".
  2. Structural naming patterns that commonly indicate a copy/variant
     without using an explicit keyword (e.g. an OS-generated "(1)" suffix,
     a "-v2" version suffix, or an accidental double extension) -> weaker
     evidence -> classified "suspicious".

Plain sequential numbering (e.g. "CSCR1.jpg", "Glaucoma_045.jpg") matches
neither signal and is classified "normal-looking", since that is the
expected, ordinary naming convention for this dataset.
"""

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Sequence, Set, Tuple, Union

from src.data.records import ImageRecord, write_csv

AUGMENTATION_REPORT_COLUMNS = [
    "path",
    "filename",
    "class_directory",
    "canonical_class",
    "classification",
    "matched_patterns",
    "reason",
]

NORMAL = "normal-looking"
SUSPICIOUS = "suspicious"
HIGHLY_SUSPICIOUS = "highly-suspicious"

_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")

# Structural patterns are intentionally separate from the configurable
# keyword list: they detect naming *shapes* (not words), so they would not
# be expressible as simple keyword tokens.
_WEAK_STRUCTURAL_PATTERNS = {
    "os_copy_suffix": re.compile(r"\(\s*\d+\s*\)$"),  # "...image (1)"
    "version_suffix": re.compile(r"[-_]v\d+$", re.IGNORECASE),  # "...img-v2"
    "dup_suffix": re.compile(r"[-_](dup|dupe)\d*$", re.IGNORECASE),  # "...img_dup"
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_SPLIT_RE.split(text) if t]


def _build_keyword_set(keywords: Sequence[str]) -> Set[str]:
    return {k.strip().lower() for k in keywords if k.strip()}


def _keyword_token_matches(text: str, keyword_set: Set[str]) -> List[str]:
    return sorted(set(t for t in _tokenize(text) if t in keyword_set))


@dataclass(frozen=True)
class AugmentationFinding:
    record: ImageRecord
    classification: str
    matched_patterns: List[str]

    @property
    def reason(self) -> str:
        if not self.matched_patterns:
            return ""
        return "; ".join(self.matched_patterns)


def classify_filename(
    filename: str,
    directory_name: str,
    keyword_set: Set[str],
    supported_extensions: Sequence[str] = (".jpg", ".jpeg"),
) -> Tuple[str, List[str]]:
    """Classify a single filename (+ its parent directory name).

    Returns (classification, matched_patterns) where matched_patterns are
    human-readable strings identifying exactly what triggered the flag.
    """
    stem = PurePosixPath(filename).stem
    matched: List[str] = []

    for hit in _keyword_token_matches(stem, keyword_set):
        matched.append(f"keyword:{hit}")
    for hit in _keyword_token_matches(directory_name, keyword_set):
        matched.append(f"keyword:dir:{hit}")

    for pattern_name, pattern in _WEAK_STRUCTURAL_PATTERNS.items():
        if pattern.search(stem):
            matched.append(f"pattern:{pattern_name}")

    lower_extensions = {ext.lower().lstrip(".") for ext in supported_extensions}
    stem_suffix = PurePosixPath(stem).suffix.lstrip(".").lower()
    if stem_suffix in lower_extensions:
        matched.append("pattern:double_extension")

    if any(m.startswith("keyword:") for m in matched):
        classification = HIGHLY_SUSPICIOUS
    elif matched:
        classification = SUSPICIOUS
    else:
        classification = NORMAL

    return classification, matched


def analyze_augmentation_families(
    records: Iterable[ImageRecord],
    keywords: Sequence[str],
    supported_extensions: Sequence[str] = (".jpg", ".jpeg"),
) -> List[AugmentationFinding]:
    keyword_set = _build_keyword_set(keywords)

    findings = []
    for record in records:
        classification, matched = classify_filename(
            record.filename,
            record.class_directory,
            keyword_set,
            supported_extensions,
        )
        findings.append(
            AugmentationFinding(
                record=record, classification=classification, matched_patterns=matched
            )
        )
    return findings


def write_augmentation_report_csv(
    findings: Iterable[AugmentationFinding],
    output_path: Union[str, Path],
    include_normal: bool = False,
) -> None:
    """Write flagged findings to CSV.

    By default only suspicious/highly-suspicious findings are written (a
    "normal-looking" row has no reason to report), but the header is always
    written even if nothing was flagged.
    """
    rows = []
    for finding in findings:
        if not include_normal and finding.classification == NORMAL:
            continue
        rows.append(
            {
                "path": finding.record.path,
                "filename": finding.record.filename,
                "class_directory": finding.record.class_directory,
                "canonical_class": finding.record.canonical_class,
                "classification": finding.classification,
                "matched_patterns": ";".join(finding.matched_patterns),
                "reason": finding.reason,
            }
        )
    write_csv(rows, AUGMENTATION_REPORT_COLUMNS, output_path)
