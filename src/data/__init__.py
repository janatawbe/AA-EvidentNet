"""Dataset auditing, splitting, balanced-set generation, and the PyTorch
data-loading layer.

Implemented: raw dataset audit (audit_dataset, detect_duplicates,
detect_augmented_families, dataset_statistics, records), the eligibility
layer (eligibility, duplicate_review), the original 70/20/10 split
(build_split), the balanced 2000/class training set
(generate_balanced_dataset), and the PyTorch dataset/transform/dataloader
layer (dataset, transforms, dataloaders).
"""
