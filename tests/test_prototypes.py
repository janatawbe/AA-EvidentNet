"""Tests for src.models.prototypes - the shared class-prototype / cosine-
distance utilities used by both src.evaluation.ood_uncertainty (post-hoc
OOD detection) and src.losses.ambiguity / src.training.ambiguity_setup
(the learned class-ambiguity mechanism).

This module was relocated out of src/evaluation/ood_uncertainty.py as a
behavior-preserving refactor - tests/test_ood_uncertainty.py's own 45
tests are the regression safety net for that (they must still pass
unchanged, using ood_uncertainty.py's thin re-exporting wrappers). These
tests exercise the shared implementation directly.

CRITICAL: no test here touches the real dataset or data/raw/ - every
fixture builds its own tiny, self-contained synthetic dataset under
tmp_path.
"""

import numpy as np
import pytest
import torch
import yaml

from src.data.dataloaders import build_eval_dataloader
from src.data.dataset import RetinalDataset
from src.data.records import write_csv
from src.data.transforms import build_transforms_from_config
from src.models.factory import create_model
from src.models.prototypes import (
    ExtractedEmbeddings,
    PrototypeComputationError,
    compute_class_prototypes,
    extract_embeddings,
    nearest_prototype_cosine_distance,
)
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]  # already alphabetical - matches build_class_to_idx's ordering
MANIFEST_COLUMNS = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]


def _make_rows(raw_dir, canonical_classes, split, n_per_class, prefix):
    rows = []
    for class_name in canonical_classes:
        for i in range(n_per_class):
            filename = f"{prefix}_{class_name}_{i}.jpg"
            rel_path = f"{class_name}/{filename}"
            make_image(raw_dir / rel_path)
            sample_id = f"sample_{prefix}_{class_name}_{i}"
            rows.append(
                {
                    "path": rel_path,
                    "class": class_name,
                    "split": split,
                    "original_id": sample_id,
                    "parent_original_id": sample_id,
                    "is_original": "true",
                    "augmentation_type": "original",
                }
            )
    return rows


# --- cosine distance ---


def test_cosine_distance_identical_vectors_is_zero():
    embeddings = np.array([[1.0, 2.0, 3.0]])
    prototypes = np.array([[1.0, 2.0, 3.0]])
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_orthogonal_vectors_is_one():
    embeddings = np.array([[1.0, 0.0]])
    prototypes = np.array([[0.0, 1.0]])
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(1.0, abs=1e-6)


def test_cosine_distance_opposite_vectors_is_two():
    embeddings = np.array([[1.0, 0.0]])
    prototypes = np.array([[-1.0, 0.0]])
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(2.0, abs=1e-6)


def test_cosine_distance_scale_invariant():
    embeddings = np.array([[2.0, 0.0]])
    prototypes = np.array([[10.0, 0.0]])  # same direction, different magnitude
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_picks_nearest_of_multiple_prototypes():
    embeddings = np.array([[1.0, 0.0]])
    prototypes = np.array([[0.0, 1.0], [1.0, 0.0001], [-1.0, 0.0]])
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(0.0, abs=1e-3)


def test_cosine_distance_rejects_mismatched_dims():
    embeddings = np.zeros((2, 3))
    prototypes = np.zeros((2, 4))
    with pytest.raises(PrototypeComputationError, match="embedding dim"):
        nearest_prototype_cosine_distance(embeddings, prototypes)


def test_cosine_distance_rejects_non_2d_input():
    with pytest.raises(PrototypeComputationError, match="2-D"):
        nearest_prototype_cosine_distance(np.zeros(3), np.zeros((1, 3)))


# --- compute_class_prototypes ---


def _setup(tmp_path, n_per_class=3, num_classes=4):
    canonical_classes = CANONICAL_CLASSES[:num_classes]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_cfg = write_min_models_config(tmp_path, num_classes=num_classes, include_aa_evidentnet=True)

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rows = _make_rows(raw_dir, canonical_classes, "train", n_per_class, "train")
    write_csv(rows, MANIFEST_COLUMNS, manifests_dir / "train_original.csv")

    return dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir


def _build_loader(dataset_cfg, canonical_classes, manifests_dir, raw_dir, tmp_path, batch_size=4):
    dataset_config = yaml.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    _, eval_transform = build_transforms_from_config(dataset_config)
    dataset = RetinalDataset.from_manifest(
        manifests_dir / "train_original.csv", canonical_classes, raw_dir, tmp_path / "processed" / "train",
        transform=eval_transform, expected_split="train", require_all_original=True,
    )
    return dataset, build_eval_dataloader(dataset, batch_size=batch_size, num_workers=0)


def test_compute_class_prototypes_matches_hand_computed_mean(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=3)
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    model.eval()

    dataset, loader = _build_loader(dataset_cfg, canonical_classes, manifests_dir, raw_dir, tmp_path)
    prototypes, counts = compute_class_prototypes(model, loader, torch.device("cpu"), len(canonical_classes))
    assert prototypes.shape == (len(canonical_classes), model.embedding_dim)
    assert counts == [3] * len(canonical_classes)

    with torch.inference_mode():
        all_embeddings = []
        all_labels = []
        for batch in build_eval_dataloader(dataset, batch_size=4, num_workers=0):
            output = model(batch["image"], return_features=True)
            all_embeddings.append(output.embedding.numpy())
            all_labels.extend(batch["label"].numpy().tolist())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)
    for k in range(len(canonical_classes)):
        expected = all_embeddings[all_labels == k].mean(axis=0)
        assert prototypes[k] == pytest.approx(expected, abs=1e-5)


def test_compute_class_prototypes_is_deterministic_given_fixed_inputs(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    model.eval()

    dataset, loader_a = _build_loader(dataset_cfg, canonical_classes, manifests_dir, raw_dir, tmp_path)
    prototypes_a, counts_a = compute_class_prototypes(model, loader_a, torch.device("cpu"), len(canonical_classes))

    _, loader_b = _build_loader(dataset_cfg, canonical_classes, manifests_dir, raw_dir, tmp_path)
    prototypes_b, counts_b = compute_class_prototypes(model, loader_b, torch.device("cpu"), len(canonical_classes))

    assert counts_a == counts_b
    assert prototypes_a == pytest.approx(prototypes_b, abs=1e-8)


def test_compute_class_prototypes_raises_for_missing_class(tmp_path):
    canonical_classes = CANONICAL_CLASSES[:3]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_cfg = write_min_models_config(tmp_path, num_classes=len(canonical_classes), include_aa_evidentnet=True)

    # Deliberately omit all rows for the last class.
    rows = _make_rows(raw_dir, canonical_classes[:2], "train", 2, "train")
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, MANIFEST_COLUMNS, manifests_dir / "train_original.csv")

    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    model.eval()

    dataset_config = yaml.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    _, eval_transform = build_transforms_from_config(dataset_config)
    train_dataset = RetinalDataset.from_manifest(
        manifests_dir / "train_original.csv", canonical_classes, raw_dir, tmp_path / "processed" / "train",
        transform=eval_transform, expected_split="train", require_all_original=True,
    )
    loader = build_eval_dataloader(train_dataset, batch_size=4, num_workers=0)

    with pytest.raises(PrototypeComputationError, match=r"zero samples for class index"):
        compute_class_prototypes(model, loader, torch.device("cpu"), len(canonical_classes))


# --- extract_embeddings (shared by Phase 1's ambiguity_setup/ambiguity_validation
# and Phase 2's neighborhood_ambiguity_setup/neighborhood_ambiguity_validation) ---


def test_extract_embeddings_shapes_and_types(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=3)
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    model.eval()

    dataset, loader = _build_loader(dataset_cfg, canonical_classes, manifests_dir, raw_dir, tmp_path)
    result = extract_embeddings(model, loader, torch.device("cpu"))

    assert isinstance(result, ExtractedEmbeddings)
    n = len(canonical_classes) * 3
    assert result.embeddings.shape == (n, model.embedding_dim)
    assert result.labels.shape == (n,)
    assert result.predictions.shape == (n,)
    assert result.uncertainty.shape == (n,)
    assert np.all(result.uncertainty > 0.0) and np.all(result.uncertainty <= 1.0)


def test_extract_embeddings_labels_match_manifest_classes(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=3)
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    model.eval()

    dataset, loader = _build_loader(dataset_cfg, canonical_classes, manifests_dir, raw_dir, tmp_path)
    result = extract_embeddings(model, loader, torch.device("cpu"))

    counts = np.bincount(result.labels, minlength=len(canonical_classes))
    assert list(counts) == [3] * len(canonical_classes)
    assert set(result.labels.tolist()) <= set(range(len(canonical_classes)))


def test_extract_embeddings_matches_compute_class_prototypes_embeddings(tmp_path):
    # Cross-check: manually averaging extract_embeddings' per-sample
    # embeddings by class must equal compute_class_prototypes' own output,
    # proving both share consistent forward-pass semantics.
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=3)
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    model.eval()

    dataset, loader = _build_loader(dataset_cfg, canonical_classes, manifests_dir, raw_dir, tmp_path)
    prototypes, _ = compute_class_prototypes(model, loader, torch.device("cpu"), len(canonical_classes))

    _, loader2 = _build_loader(dataset_cfg, canonical_classes, manifests_dir, raw_dir, tmp_path)
    result = extract_embeddings(model, loader2, torch.device("cpu"))

    for k in range(len(canonical_classes)):
        expected = result.embeddings[result.labels == k].mean(axis=0)
        assert prototypes[k] == pytest.approx(expected, abs=1e-5)


def test_extract_embeddings_raises_on_empty_loader(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=1)
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    model.eval()

    dataset, _ = _build_loader(dataset_cfg, canonical_classes, manifests_dir, raw_dir, tmp_path)
    empty_loader = build_eval_dataloader(dataset, batch_size=4, num_workers=0)
    empty_loader.dataset.rows = []  # force zero batches without touching real data

    with pytest.raises(PrototypeComputationError, match="zero batches"):
        extract_embeddings(model, empty_loader, torch.device("cpu"))
