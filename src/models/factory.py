"""Model factory: create_model(name, config) -> TimmBackboneModel.

`config` is a loaded configs/models.yaml dict (or an equivalent dict with
a `baselines` section). Every architecture parameter (backbone name,
pretrained flag, num_classes, dropout) is read from configuration —
nothing here hard-codes a specific architecture's constants.
"""

from typing import Any, Dict

from src.models.base import TimmBackboneModel

MODEL_NAMES = ("resnet50", "efficientnetb0", "maxvit")


class ModelConfigError(Exception):
    """Raised for an unknown model name or a malformed baselines.* entry."""


def create_model(name: str, config: Dict[str, Any]) -> TimmBackboneModel:
    """Factory used by training/evaluation/CLI code:

        create_model("resnet50", config)
        create_model("efficientnetb0", config)
        create_model("maxvit", config)

    Raises ModelConfigError clearly for an unregistered name or a
    baselines.* entry missing its required `architecture` key.
    """
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
