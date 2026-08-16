from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("model_ema", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


class LayerNorm(nn.Module):
    """
    LayerNorm hỗ trợ cả channels_last và channels_first.
    Cách đặt tên/tham số tương thích implementation ConvNeXt V2 chính thức.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()

        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(
                "data_format phải là 'channels_last' hoặc 'channels_first'"
            )

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = float(eps)
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)

        return (
            self.weight[:, None, None] * x
            + self.bias[:, None, None]
        )


class GRN(nn.Module):
    """Global Response Normalization của ConvNeXt V2."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        return residual + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(
                in_chans,
                dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm(
                dims[0],
                eps=1e-6,
                data_format="channels_first",
            ),
        )
        self.downsample_layers.append(stem)

        for index in range(3):
            downsample = nn.Sequential(
                LayerNorm(
                    dims[index],
                    eps=1e-6,
                    data_format="channels_first",
                ),
                nn.Conv2d(
                    dims[index],
                    dims[index + 1],
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.downsample_layers.append(downsample)

        drop_path_values = torch.linspace(
            0,
            float(drop_path_rate),
            sum(depths),
        ).tolist()

        self.stages = nn.ModuleList()
        cursor = 0

        for stage_index in range(4):
            blocks = [
                Block(
                    dim=dims[stage_index],
                    drop_path=drop_path_values[cursor + block_index],
                )
                for block_index in range(depths[stage_index])
            ]
            self.stages.append(nn.Sequential(*blocks))
            cursor += depths[stage_index]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for index in range(4):
            x = self.downsample_layers[index](x)
            x = self.stages[index](x)

        x = x.mean(dim=(-2, -1))
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def convnextv2_tiny(
    num_classes: int = 1000,
    drop_path_rate: float = 0.0,
) -> ConvNeXtV2:
    return ConvNeXtV2(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=drop_path_rate,
    )


class ConvNeXtV2GAD(nn.Module):
    """
    ConvNeXt V2-Tiny GAD.

    Không import models/utils.py của repository chính thức, vì file đó kéo
    MinkowskiEngine phục vụ sparse ConvNeXt. Nhánh dense 2D này không cần
    MinkowskiEngine.
    """

    def __init__(
        self,
        checkpoint_path: str,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        checkpoint_file = Path(checkpoint_path)

        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy ConvNeXt V2 checkpoint: {checkpoint_file}"
            )

        network = convnextv2_tiny(
            num_classes=21841,
            drop_path_rate=float(drop_path_rate),
        )

        checkpoint = _safe_torch_load(str(checkpoint_file))
        state = _extract_state_dict(checkpoint)

        cleaned_state = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            key = key.removeprefix("model.")
            cleaned_state[key] = value

        # Checkpoint 22K có thể có head 21841 lớp.
        # Nếu số lớp/head khác, bỏ riêng head nhưng vẫn strict với backbone.
        head_weight = cleaned_state.get("head.weight")
        head_bias = cleaned_state.get("head.bias")

        if (
            head_weight is not None
            and tuple(head_weight.shape) != tuple(network.head.weight.shape)
        ):
            cleaned_state.pop("head.weight", None)
            cleaned_state.pop("head.bias", None)

        missing, unexpected = network.load_state_dict(
            cleaned_state,
            strict=False,
        )

        allowed_missing = {"head.weight", "head.bias"}
        bad_missing = [
            key
            for key in missing
            if key not in allowed_missing
        ]

        if bad_missing or unexpected:
            raise RuntimeError(
                "ConvNeXt V2 backbone không khớp checkpoint. "
                f"missing={bad_missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )

        self.feature_dim = int(network.head.in_features)
        network.head = nn.Identity()
        self.encoder = network

        print(f"Loaded ConvNeXt V2 checkpoint: {checkpoint_file}")
        print(
            "ConvNeXt V2 backbone matched; "
            f"feature_dim={self.feature_dim}"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)

        if features.ndim > 2:
            features = features.flatten(1)

        return features