"""Model architectures.

Implemented: a uniform baseline-model interface (base.TimmBackboneModel)
around timm.create_model, a factory (factory.create_model) for the three
required baselines (resnet50, efficientnetb0, maxvit), and an offline
CPU-safe smoke-test utility (model_check). AA-EvidentNet (the proposed
model) is not yet implemented.
"""
