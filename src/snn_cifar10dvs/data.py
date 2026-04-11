"""Dataset loading and split helpers for CIFAR10-DVS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import tonic
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from .config import CIFAR10DVSConfig, ProjectPaths


Transform = Callable[[object], object]


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Container for reproducible train/validation/test subsets."""

    train: Subset
    validation: Subset
    test: Subset


def load_cifar10dvs(
    paths: ProjectPaths | None = None,
    transform: Transform | None = None,
    target_transform: Transform | None = None,
    transforms: Transform | None = None,
) -> tonic.datasets.CIFAR10DVS:
    """Load CIFAR10-DVS through Tonic.

    Tonic handles the download from the official Figshare-hosted archive when the
    dataset is not already present under ``data/raw``.
    """

    project_paths = paths or ProjectPaths()
    return tonic.datasets.CIFAR10DVS(
        save_to=str(project_paths.raw_data_dir),
        transform=transform,
        target_transform=target_transform,
        transforms=transforms,
    )


def extract_targets(dataset: Dataset) -> np.ndarray:
    """Return integer class labels for a dataset or subset."""

    if isinstance(dataset, Subset):
        subset_targets = extract_targets(dataset.dataset)
        return subset_targets[np.asarray(dataset.indices, dtype=np.int64)]

    for attribute_name in ("targets", "labels"):
        if hasattr(dataset, attribute_name):
            targets = getattr(dataset, attribute_name)
            if targets is not None:
                return np.asarray(targets, dtype=np.int64)

    return np.fromiter(
        (int(dataset[index][1]) for index in range(len(dataset))),
        dtype=np.int64,
        count=len(dataset),
    )


def create_dataset_splits(
    dataset: Dataset,
    config: CIFAR10DVSConfig | None = None,
) -> DatasetSplits:
    """Create stratified train/validation/test splits from a single dataset."""

    dataset_config = config or CIFAR10DVSConfig()
    dataset_config.validate_split_ratios()

    targets = extract_targets(dataset)
    indices = np.arange(len(dataset), dtype=np.int64)

    train_indices, temp_indices, _, temp_targets = train_test_split(
        indices,
        targets,
        train_size=dataset_config.train_ratio,
        random_state=dataset_config.random_seed,
        shuffle=True,
        stratify=targets,
    )

    val_ratio_within_temp = dataset_config.val_ratio / (
        dataset_config.val_ratio + dataset_config.test_ratio
    )

    validation_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio_within_temp,
        random_state=dataset_config.random_seed,
        shuffle=True,
        stratify=temp_targets,
    )

    return DatasetSplits(
        train=Subset(dataset, train_indices.tolist()),
        validation=Subset(dataset, validation_indices.tolist()),
        test=Subset(dataset, test_indices.tolist()),
    )


def build_dataloaders(
    splits: DatasetSplits,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Construct standard PyTorch dataloaders for each dataset split."""

    return {
        "train": DataLoader(
            splits.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "validation": DataLoader(
            splits.validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            splits.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def get_class_distribution(dataset: Dataset) -> dict[int, int]:
    """Count samples per class for quick dataset inspection."""

    targets = extract_targets(dataset)
    classes, counts = np.unique(targets, return_counts=True)
    return {int(class_id): int(count) for class_id, count in zip(classes, counts)}

