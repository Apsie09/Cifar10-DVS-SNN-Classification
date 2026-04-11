"""Shared project paths and default experiment settings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Common project directories used across scripts and notebooks."""

    root_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    notebooks_dir: Path = PROJECT_ROOT / "notebooks"
    scripts_dir: Path = PROJECT_ROOT / "scripts"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    checkpoints_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    metrics_dir: Path = PROJECT_ROOT / "outputs" / "metrics"


@dataclass(frozen=True, slots=True)
class CIFAR10DVSConfig:
    """Default settings for CIFAR10-DVS experiments."""

    dataset_name: str = "CIFAR10-DVS"
    sensor_size: tuple[int, int, int] = (128, 128, 2)
    num_classes: int = 10
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    batch_size: int = 16
    num_workers: int = 0
    n_time_bins: int = 10

    def validate_split_ratios(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "Dataset split ratios must sum to 1.0. "
                f"Received train={self.train_ratio}, val={self.val_ratio}, "
                f"test={self.test_ratio}."
            )

