"""Evaluation helpers for SNN classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from .utils import spike_count_predictions


@dataclass(frozen=True, slots=True)
class EvaluationOutputs:
    """Collected predictions and labels for one evaluation pass."""

    targets: np.ndarray
    predictions: np.ndarray
    spike_counts: np.ndarray


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> EvaluationOutputs:
    """Run inference and collect predictions, labels, and spike counts."""

    model.eval()
    model.to(device)

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_spike_counts: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            spike_record = model(inputs)
            spike_counts = spike_record.sum(dim=0)
            predictions = spike_count_predictions(spike_record)

            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_spike_counts.append(spike_counts.cpu().numpy())

    if not all_targets:
        raise ValueError("Dataloader is empty; cannot collect evaluation outputs.")

    return EvaluationOutputs(
        targets=np.concatenate(all_targets, axis=0),
        predictions=np.concatenate(all_predictions, axis=0),
        spike_counts=np.concatenate(all_spike_counts, axis=0),
    )


def summarize_classification(
    evaluation_outputs: EvaluationOutputs,
    class_names: list[str] | None = None,
) -> dict[str, object]:
    """Compute common classification metrics for a prediction set."""

    targets = evaluation_outputs.targets
    predictions = evaluation_outputs.predictions

    labels = np.unique(targets)
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels),
        "classification_report": report,
    }
