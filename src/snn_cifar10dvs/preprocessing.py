"""Preprocessing utilities for event-based CIFAR10-DVS samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tonic import transforms


@dataclass(frozen=True, slots=True)
class FrameTransformConfig:
    """Configuration for converting event streams into dense frame tensors."""

    sensor_size: tuple[int, int, int] = (128, 128, 2)
    time_window: int | None = None
    event_count: int | None = None
    n_time_bins: int | None = 10
    n_event_bins: int | None = None
    overlap: float = 0.0
    include_incomplete: bool = False
    normalize: bool = True
    clip_value: float | None = None

    def validate(self) -> None:
        slicing_options = (
            self.time_window,
            self.event_count,
            self.n_time_bins,
            self.n_event_bins,
        )
        active_options = sum(option is not None for option in slicing_options)
        if active_options != 1:
            raise ValueError(
                "Exactly one of time_window, event_count, n_time_bins, or "
                "n_event_bins must be set."
            )


class FramesToTensor:
    """Convert numpy frames into float32 torch tensors."""

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frames).to(dtype=torch.float32)


class ClampFrames:
    """Clamp dense frame values to a fixed upper bound."""

    def __init__(self, max_value: float) -> None:
        self.max_value = max_value

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.clamp(frames, min=0.0, max=self.max_value)


class NormalizeFrames:
    """Scale each sample by its maximum event count."""

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        max_value = torch.amax(frames)
        if max_value <= 0:
            return frames
        return frames / max_value


def build_frame_transform(
    config: FrameTransformConfig | None = None,
) -> transforms.Compose:
    """Build a reusable Tonic transform that returns dense frame tensors.

    ``ToFrame`` returns data with shape ``(time, polarity, height, width)``.
    """

    transform_config = config or FrameTransformConfig()
    transform_config.validate()

    transform_steps: list[object] = [
        transforms.ToFrame(
            sensor_size=transform_config.sensor_size,
            time_window=transform_config.time_window,
            event_count=transform_config.event_count,
            n_time_bins=transform_config.n_time_bins,
            n_event_bins=transform_config.n_event_bins,
            overlap=transform_config.overlap,
            include_incomplete=transform_config.include_incomplete,
        ),
        FramesToTensor(),
    ]

    if transform_config.clip_value is not None:
        transform_steps.append(ClampFrames(transform_config.clip_value))

    if transform_config.normalize:
        transform_steps.append(NormalizeFrames())

    return transforms.Compose(transform_steps)


def flatten_time_and_polarity(frames: torch.Tensor) -> torch.Tensor:
    """Convert ``(time, polarity, height, width)`` into ``(channels, height, width)``."""

    if frames.ndim != 4:
        raise ValueError(
            "Expected frame tensor with shape (time, polarity, height, width). "
            f"Received shape {tuple(frames.shape)}."
        )

    time_steps, polarities, height, width = frames.shape
    return frames.reshape(time_steps * polarities, height, width)
