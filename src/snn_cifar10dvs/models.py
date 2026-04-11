"""Baseline SNN models for CIFAR10-DVS classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import snntorch as snn
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class BaselineSNNConfig:
    """Configuration for the baseline convolutional SNN classifier."""

    input_channels: int = 2
    num_classes: int = 10
    conv_channels: tuple[int, int, int] = (16, 32, 64)
    hidden_features: int = 256
    beta: float = 0.9
    dropout: float = 0.2
    use_batch_norm: bool = False


ModelVariant = Literal["baseline", "wider", "wider_bn", "nengo_like"]


def build_model_config(
    variant: ModelVariant,
    *,
    input_channels: int = 2,
    num_classes: int = 10,
) -> BaselineSNNConfig:
    """Return a named model preset for controlled architecture experiments."""

    if variant == "baseline":
        return BaselineSNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
        )

    if variant == "wider":
        return BaselineSNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            conv_channels=(32, 64, 128),
            hidden_features=512,
            dropout=0.3,
        )

    if variant == "wider_bn":
        return BaselineSNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            conv_channels=(32, 64, 128),
            hidden_features=512,
            dropout=0.3,
            use_batch_norm=True,
        )

    if variant == "nengo_like":
        return BaselineSNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            conv_channels=(32, 64, 128),
            hidden_features=0,
            dropout=0.3,
        )

    raise ValueError(
        "Unsupported model variant "
        f"'{variant}'. Expected 'baseline', 'wider', 'wider_bn', or 'nengo_like'."
    )


class BaselineConvSNN(nn.Module):
    """A compact convolutional SNN for frame-based CIFAR10-DVS inputs.

    The model expects tensors with shape ``(batch, time, channels, height, width)``
    and returns a spike tensor with shape ``(time, batch, num_classes)``.
    """

    def __init__(self, config: BaselineSNNConfig | None = None) -> None:
        super().__init__()
        model_config = config or BaselineSNNConfig()
        c1, c2, c3 = model_config.conv_channels

        self.config = model_config
        feature_layers: list[nn.Module] = [
            nn.Conv2d(model_config.input_channels, c1, kernel_size=5, padding=2),
        ]
        if model_config.use_batch_norm:
            feature_layers.append(nn.BatchNorm2d(c1))
        feature_layers.extend(
            [
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            ]
        )
        if model_config.use_batch_norm:
            feature_layers.append(nn.BatchNorm2d(c2))
        feature_layers.extend(
            [
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            ]
        )
        if model_config.use_batch_norm:
            feature_layers.append(nn.BatchNorm2d(c3))
        feature_layers.extend(
            [
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.AdaptiveAvgPool2d((4, 4)),
            ]
        )
        self.features = nn.Sequential(*feature_layers)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=model_config.dropout)
        self.fc_hidden = nn.Linear(c3 * 4 * 4, model_config.hidden_features)
        self.lif_hidden = snn.Leaky(beta=model_config.beta)
        self.fc_out = nn.Linear(model_config.hidden_features, model_config.num_classes)
        self.lif_out = snn.Leaky(beta=model_config.beta)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)

        if frames.ndim != 5:
            raise ValueError(
                "Expected input with shape (batch, time, channels, height, width) "
                "or (time, channels, height, width). "
                f"Received shape {tuple(frames.shape)}."
            )

        batch_size, num_steps, _, _, _ = frames.shape
        hidden_mem = self.lif_hidden.init_leaky()
        output_mem = self.lif_out.init_leaky()
        spike_record: list[torch.Tensor] = []

        for step in range(num_steps):
            frame_t = frames[:, step]
            current = self.features(frame_t)
            current = self.flatten(current)
            current = self.dropout(current)
            current = self.fc_hidden(current)
            hidden_spikes, hidden_mem = self.lif_hidden(current, hidden_mem)
            output_current = self.fc_out(hidden_spikes)
            output_spikes, output_mem = self.lif_out(output_current, output_mem)
            spike_record.append(output_spikes)

        return torch.stack(spike_record, dim=0).view(
            num_steps, batch_size, self.config.num_classes
        )


class NengoInspiredConvSNN(nn.Module):
    """A more SNN-heavy conv architecture inspired by NengoDL-style blocks.

    The network applies spiking nonlinearities throughout the feature extractor,
    using strided convolutions for progressive spatial reduction before a linear
    readout layer produces output spikes over time.
    """

    def __init__(self, config: BaselineSNNConfig | None = None) -> None:
        super().__init__()
        model_config = config or build_model_config("nengo_like")
        c1, c2, c3 = model_config.conv_channels

        self.config = model_config
        self.conv1 = nn.Conv2d(
            model_config.input_channels,
            c1,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.lif1 = snn.Leaky(beta=model_config.beta)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=0)
        self.lif2 = snn.Leaky(beta=model_config.beta)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=0)
        self.lif3 = snn.Leaky(beta=model_config.beta)
        self.readout_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.dropout = nn.Dropout(p=model_config.dropout)
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(c3 * 5 * 5, model_config.num_classes)
        self.lif_out = snn.Leaky(beta=model_config.beta)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)

        if frames.ndim != 5:
            raise ValueError(
                "Expected input with shape (batch, time, channels, height, width) "
                "or (time, channels, height, width). "
                f"Received shape {tuple(frames.shape)}."
            )

        batch_size, num_steps, _, _, _ = frames.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        output_mem = self.lif_out.init_leaky()
        spike_record: list[torch.Tensor] = []

        for step in range(num_steps):
            x = frames[:, step]
            x = self.conv1(x)
            x, mem1 = self.lif1(x, mem1)
            x = self.conv2(x)
            x, mem2 = self.lif2(x, mem2)
            x = self.conv3(x)
            x, mem3 = self.lif3(x, mem3)
            x = self.readout_pool(x)
            x = self.dropout(x)
            x = self.flatten(x)
            x = self.fc_out(x)
            output_spikes, output_mem = self.lif_out(x, output_mem)
            spike_record.append(output_spikes)

        return torch.stack(spike_record, dim=0).view(
            num_steps, batch_size, self.config.num_classes
        )


def build_model(
    variant: ModelVariant,
    *,
    input_channels: int = 2,
    num_classes: int = 10,
) -> nn.Module:
    """Instantiate a model for a named experiment variant."""

    model_config = build_model_config(
        variant,
        input_channels=input_channels,
        num_classes=num_classes,
    )

    if variant in {"baseline", "wider"}:
        return BaselineConvSNN(model_config)

    if variant == "wider_bn":
        return BaselineConvSNN(model_config)

    if variant == "nengo_like":
        return NengoInspiredConvSNN(model_config)

    raise ValueError(
        "Unsupported model variant "
        f"'{variant}'. Expected 'baseline', 'wider', 'wider_bn', or 'nengo_like'."
    )
