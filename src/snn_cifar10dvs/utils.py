"""Small shared utilities for experiments and training scripts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select the best available PyTorch device."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def spike_count_predictions(spike_record: torch.Tensor) -> torch.Tensor:
    """Convert a spike record of shape ``(time, batch, classes)`` into class IDs."""

    if spike_record.ndim != 3:
        raise ValueError(
            "Expected spike record with shape (time, batch, classes). "
            f"Received shape {tuple(spike_record.shape)}."
        )

    return spike_record.sum(dim=0).argmax(dim=1)
