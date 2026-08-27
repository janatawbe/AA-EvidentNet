"""Reusable training engine and baseline-model training orchestration.

Implemented: Trainer (train/validate loop, AMP, gradient accumulation/
clipping, early stopping - trainer.py), checkpointing (save/load/resume/
compatibility - checkpointing.py), per-run structured logging
(logging.py), the experiment registry (registry.py), and baseline-model
orchestration (run_baseline.py). AA-EvidentNet training is not yet
implemented (a later task) but will reuse the same Trainer.
"""
