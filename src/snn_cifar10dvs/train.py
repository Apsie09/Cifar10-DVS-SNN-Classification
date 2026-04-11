"""Training helpers for baseline SNN experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import ensure_directory, spike_count_predictions


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Average metrics for one pass over a dataset split."""

    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class TrainingEpochResult:
    """Training summary for a single epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float | None = None
    validation_accuracy: float | None = None


def compute_loss_and_predictions(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and return the loss with class predictions."""

    spike_record = model(inputs)
    spike_counts = spike_record.sum(dim=0)
    loss = criterion(spike_counts, targets)
    predictions = spike_count_predictions(spike_record)
    return loss, predictions


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    description: str | None = None,
) -> EpochMetrics:
    """Run one training or evaluation epoch."""

    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress_bar = tqdm(
        dataloader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )

    for inputs, targets in progress_bar:
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.long)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            loss, predictions = compute_loss_and_predictions(
                model=model,
                inputs=inputs,
                targets=targets,
                criterion=criterion,
            )

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()
        total_examples += batch_size

        progress_bar.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            acc=f"{total_correct / total_examples:.4f}",
        )

    if total_examples == 0:
        raise ValueError("Dataloader is empty; cannot compute epoch metrics.")

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    checkpoint_path: str | Path | None = None,
) -> list[TrainingEpochResult]:
    """Train a model and optionally save the best validation checkpoint."""

    model.to(device)
    best_validation_loss = float("inf")
    history: list[TrainingEpochResult] = []

    checkpoint_target: Path | None = None
    if checkpoint_path is not None:
        checkpoint_target = Path(checkpoint_path)
        ensure_directory(checkpoint_target.parent)

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            description=f"Train {epoch}/{num_epochs}",
        )

        validation_metrics: EpochMetrics | None = None
        if validation_loader is not None:
            validation_metrics = run_epoch(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                description=f"Val {epoch}/{num_epochs}",
            )

        epoch_result = TrainingEpochResult(
            epoch=epoch,
            train_loss=train_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            validation_loss=None if validation_metrics is None else validation_metrics.loss,
            validation_accuracy=(
                None if validation_metrics is None else validation_metrics.accuracy
            ),
        )
        history.append(epoch_result)

        progress_message = (
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={train_metrics.loss:.4f} | "
            f"train_acc={train_metrics.accuracy:.4f}"
        )
        if validation_metrics is not None:
            progress_message += (
                f" | val_loss={validation_metrics.loss:.4f} | "
                f"val_acc={validation_metrics.accuracy:.4f}"
            )
        print(progress_message, flush=True)

        if (
            checkpoint_target is not None
            and validation_metrics is not None
            and validation_metrics.loss < best_validation_loss
        ):
            best_validation_loss = validation_metrics.loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": asdict(train_metrics),
                    "validation_metrics": asdict(validation_metrics),
                },
                checkpoint_target,
            )
            print(
                f"Saved improved checkpoint to {checkpoint_target}",
                flush=True,
            )

    return history
