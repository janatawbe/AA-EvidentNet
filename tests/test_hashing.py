"""Tests for src.utils.hashing: file and manifest hashing."""

import pytest

from src.utils.hashing import hash_file, hash_manifest


def test_hash_file_deterministic(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello world")
    assert hash_file(f) == hash_file(f)


def test_hash_file_matches_known_sha256(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello world")
    # Precomputed sha256("hello world")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert hash_file(f) == expected


def test_hash_file_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"content one")
    f2.write_bytes(b"content two")
    assert hash_file(f1) != hash_file(f2)


def test_hash_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "missing.txt")


def test_hash_manifest_deterministic():
    entries = [("b.jpg", "hash_b"), ("a.jpg", "hash_a")]
    assert hash_manifest(entries) == hash_manifest(entries)


def test_hash_manifest_order_independent():
    entries_a = [("a.jpg", "hash_a"), ("b.jpg", "hash_b")]
    entries_b = [("b.jpg", "hash_b"), ("a.jpg", "hash_a")]
    assert hash_manifest(entries_a) == hash_manifest(entries_b)


def test_hash_manifest_sensitive_to_content_changes():
    entries_a = [("a.jpg", "hash_a"), ("b.jpg", "hash_b")]
    entries_b = [("a.jpg", "hash_a"), ("b.jpg", "hash_b_changed")]
    assert hash_manifest(entries_a) != hash_manifest(entries_b)


def test_hash_manifest_sensitive_to_added_entry():
    entries_a = [("a.jpg", "hash_a")]
    entries_b = [("a.jpg", "hash_a"), ("b.jpg", "hash_b")]
    assert hash_manifest(entries_a) != hash_manifest(entries_b)
