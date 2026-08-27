"""Model factory: create_model(name, config) -> a model exposing
forward(images) -> logits and forward(images, return_features=True).

`config` is a loaded configs/models.yaml dict. Every architecture
parameter is read from configuration - nothing here hard-codes a
specific architecture's constants. Two config sections are dispatched:
`baselines.*` (ResNet50/EfficientNetB0/MaxViT, via TimmBackboneModel) and
`proposed.*` (AA-EvidentNet, via AAEvidentNet - Task 7).
"""

from typing import Any, Dict, Union

from src.models.aa_evidentnet import AAEvidentNet
from src.models.base import TimmBackboneModel

MODEL_NAMES = ("resnet50", "efficientnetb0", "maxvit")
PROPOSED_MODEL_NAMES = ("aa_evidentnet",)


class ModelConfigError(Exception):
    """Raised for an unknown model name or a malformed baselines.*/proposed.* entry."""


def create_model(name: str, config: Dict[str, Any]) -> Union[TimmBackboneModel, AAEvidentNet]:
    """Factory used by training/evaluation/CLI code:

        create_model("resnet50", config)
        create_model("efficientnetb0", config)
        create_model("maxvit", config)
        create_model("aa_evidentnet", config)

    Raises ModelConfigError clearly for an unregistered name or a
    baselines.*/proposed.* entry missing its required key(s).
    """
    if name in PROPOSED_MODEL_NAMES:
        return _create_proposed_model(name, config)

    baselines = config.get("baselines", {}) or {}
    if name not in baselines:
        raise ModelConfigError(
            f"Unknown baseline model name '{name}'. Known models: {sorted(baselines.keys()) or list(MODEL_NAMES)}. "
            "Register a new one under configs/models.yaml: baselines.<name>."
        )

    model_cfg = baselines[name]
    architecture = model_cfg.get("architecture")
    if not architecture:
        raise ModelConfigError(f"configs/models.yaml: baselines.{name} is missing required key 'architecture'")

    return TimmBackboneModel(
        architecture=architecture,
        num_classes=model_cfg.get("num_classes", config.get("num_classes", 10)),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.0),
    )


def _create_proposed_model(name: str, config: Dict[str, Any]) -> AAEvidentNet:
    proposed = config.get("proposed", {}) or {}
    model_cfg = proposed.get(name)
    if model_cfg is None:
        raise ModelConfigError(
            f"Unknown proposed model name '{name}'. Known models: {sorted(proposed.keys()) or list(PROPOSED_MODEL_NAMES)}. "
            "Register a new one under configs/models.yaml: proposed.<name>."
        )

    if name == "aa_evidentnet":
        return AAEvidentNet(
            global_backbone=model_cfg.get("global_backbone", "maxvit_tiny_tf_224"),
            num_classes=model_cfg.get("num_classes", config.get("num_classes", 10)),
            embedding_dim=model_cfg.get("embedding_dim", 256),
            local_feature_dim=model_cfg.get("local_feature_dim", 128),
            pretrained=model_cfg.get("pretrained", True),
            dropout=model_cfg.get("dropout", 0.0),
        )

    raise ModelConfigError(f"No constructor registered for proposed model '{name}'")
