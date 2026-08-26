"""File and manifest hashing utilities.

Used to fingerprint raw/processed dataset files and manifests so that any
change to the data underlying an experiment is detectable.
"""

import hashlib
from pathlib import Path
from typing import Iterable, Tuple, Union

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def hash_file(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute a hex digest of a file's contents, streamed in chunks.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Compute a hex digest of a UTF-8 encoded string.

    Used for deterministic IDs derived from stable dataset-relative
    information (see src.data.build_split.compute_original_id).
    """
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()


def hash_manifest(entries: Iterable[Tuple[str, str]], algorithm: str = "sha256") -> str:
    """Compute a single hex digest summarizing a manifest of (path, hash) pairs.

    Entries are sorted by path before hashing so the result is independent of
    input ordering (e.g. from filesystem iteration order).

    Args:
        entries: iterable of (relative_path, file_hash) tuples, e.g. produced
            by hash_file() over each file in a dataset manifest.
        algorithm: hash algorithm name, passed to hashlib.new.
    """
    sorted_entries = sorted(entries, key=lambda e: e[0])
    hasher = hashlib.new(algorithm)
    for rel_path, file_hash in sorted_entries:
        hasher.update(f"{rel_path}:{file_hash}\n".encode("utf-8"))
    return hasher.hexdigest()
