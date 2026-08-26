"""CPU-safe smoke testing for the baseline models: instantiate, run a tiny
synthetic forward pass, and report parameter/feature shapes. NEVER trains.

Two distinct modes:
  - Default (pretrained=False): fully offline, safe for unit tests / CI /
    any environment without internet access. This is what
    `python run_pipeline.py model_check` and the pytest suite both use.
  - `check_pretrained=True` (`python run_pipeline.py model_check --pretrained`):
    additionally verifies that real ImageNet-pretrained weights can be
    downloaded and instantiated. Requires internet access. NOT run by the
    default pytest suite — this is a manual/CI-optional verification step.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch

from src.data.records import write_csv
from src.models.factory import MODEL_NAMES, create_model
from src.utils.config import load_config

MODEL_PARAMETERS_COLUMNS = [
    "model",
    "architecture",
    "parameters_total",
    "parameters_trainable",
    "feature_dimension",
    "num_classes",
    "image_size",
]


class ModelCheckError(Exception):
    """Raised when a model fails to instantiate, produces the wrong output
    shape, or produces NaN/Inf during the smoke forward pass."""


@dataclass
class ModelCheckResult:
    name: str
    architecture: str
    parameters_total: int
    parameters_trainable: int
    feature_dimension: int
    num_classes: int
    image_size: int
    output_shape: tuple
    feature_shape: tuple
    has_nan_or_inf: bool


def check_model(
    name: str,
    config: Dict[str, Any],
    pretrained_override: Optional[bool] = None,
    batch_size: int = 2,
    image_size: Optional[int] = None,
) -> ModelCheckResult:
    """Instantiate one baseline and run a single synthetic forward pass.
    Never trains (no optimizer, no backward pass, model.eval() + no_grad)."""
    baselines = config.get("baselines", {}) or {}
    model_cfg = dict(baselines.get(name, {}))
    if pretrained_override is not None:
        model_cfg["pretrained"] = pretrained_override
    merged_config = {**config, "baselines": {**baselines, name: model_cfg}}

    model = create_model(name, merged_config)
    model.eval()

    size = image_size or model_cfg.get("image_size", config.get("image_size", 224))
    dummy_images = torch.randn(batch_size, 3, size, size)

    with torch.no_grad():
        output = model(dummy_images, return_features=True)

    logits, features = output.logits, output.features

    expected_logits_shape = (batch_size, model.num_classes)
    if tuple(logits.shape) != expected_logits_shape:
        raise ModelCheckError(f"{name}: expected logits shape {expected_logits_shape}, got {tuple(logits.shape)}")
    if features is None or features.shape[0] != batch_size or features.shape[1] == 0:
        raise ModelCheckError(f"{name}: invalid feature shape ({None if features is None else tuple(features.shape)})")

    has_bad = bool(
        torch.isnan(logits).any() or torch.isinf(logits).any() or torch.isnan(features).any() or torch.isinf(features).any()
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return ModelCheckResult(
        name=name,
        architecture=model.architecture,
        parameters_total=total_params,
        parameters_trainable=trainable_params,
        feature_dimension=model.feature_dim,
        num_classes=model.num_classes,
        image_size=size,
        output_shape=tuple(logits.shape),
        feature_shape=tuple(features.shape),
        has_nan_or_inf=has_bad,
    )


def run_model_check(
    config_path: Union[str, Path] = "configs/models.yaml",
    check_pretrained: bool = False,
    output_csv: Optional[Union[str, Path]] = "results/tables/model_parameters.csv",
    model_names: Optional[List[str]] = None,
) -> List[ModelCheckResult]:
    config = load_config(config_path)
    names = model_names or list(MODEL_NAMES)

    print("=" * 70)
    mode = "pretrained=True (requires internet)" if check_pretrained else "pretrained=False (offline)"
    print(f"MODEL SMOKE CHECK [{mode}]")
    print("=" * 70)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("")

    results: List[ModelCheckResult] = []
    for name in names:
        result = check_model(name, config, pretrained_override=check_pretrained)
        results.append(result)

        print(f"[{name}] architecture={result.architecture}")
        print(f"  parameters: total={result.parameters_total:,}  trainable={result.parameters_trainable:,}")
        print(f"  logits shape: {result.output_shape}   feature shape: {result.feature_shape} (dim={result.feature_dimension})")
        if result.has_nan_or_inf:
            raise ModelCheckError(f"{name}: NaN or Inf detected in the smoke forward pass output")
        print("  sanity: OK (no NaN/Inf)")
        print("")

    if output_csv is not None:
        rows = [
            {
                "model": r.name,
                "architecture": r.architecture,
                "parameters_total": r.parameters_total,
                "parameters_trainable": r.parameters_trainable,
                "feature_dimension": r.feature_dimension,
                "num_classes": r.num_classes,
                "image_size": r.image_size,
            }
            for r in results
        ]
        write_csv(rows, MODEL_PARAMETERS_COLUMNS, output_csv)
        print(f"Wrote {output_csv}")

    print("=" * 70)
    return results
