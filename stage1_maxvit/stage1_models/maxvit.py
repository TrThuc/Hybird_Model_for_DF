from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import timm
import torch
import torch.nn as nn


class MaxViTStage1(nn.Module):
    """timm MaxViT-Tiny 384 with a binary Stage-1 classifier."""

    MODEL_NAME = "maxvit_tiny_tf_384.in1k"
    INPUT_SIZE = 384

    def __init__(self, num_classes: int = 2,
                 pretrained: str | bool | None = "imagenet1k_v1",
                 checkpoint: Optional[str] = None) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.net = timm.create_model(
            self.MODEL_NAME,
            pretrained=self._use_pretrained(pretrained),
            num_classes=num_classes,
        )
        classifier = self.net.get_classifier()
        if not isinstance(classifier, nn.Linear):
            raise RuntimeError("Unexpected timm MaxViT classifier layout")

        if checkpoint:
            self._load_backbone_checkpoint(Path(checkpoint))

    @staticmethod
    def _use_pretrained(value: str | bool | None) -> bool:
        if value in (None, False, "none", "None", ""):
            return False
        if value is True or str(value).lower() in {
            "imagenet1k_v1", "default", "true"
        }:
            return True
        raise ValueError(
            "model.pretrained must be imagenet1k_v1, true, false, or null"
        )

    @staticmethod
    def _torch_load(path: Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_backbone_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"MaxViT checkpoint not found: {path}")
        checkpoint = self._torch_load(path)
        state: Any = checkpoint
        if isinstance(checkpoint, Mapping):
            for key in ("model", "state_dict", "model_state_dict"):
                if isinstance(checkpoint.get(key), Mapping):
                    state = checkpoint[key]
                    break
        if not isinstance(state, Mapping):
            raise TypeError("MaxViT checkpoint does not contain a state dict")

        cleaned = {}
        for key, value in state.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                continue
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if not key.startswith("net."):
                key = f"net.{key}"
            if key.startswith("net.head.fc."):
                continue
            cleaned[key] = value

        incompatible = self.load_state_dict(cleaned, strict=False)
        missing_backbone = [
            key for key in incompatible.missing_keys
            if not key.startswith("net.head.fc.")
        ]
        if missing_backbone or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible MaxViT-Tiny 384 checkpoint: "
                f"missing={missing_backbone[:10]}, "
                f"unexpected={list(incompatible.unexpected_keys)[:10]}"
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W], received {tuple(image.shape)}")
        if tuple(image.shape[-2:]) != (self.INPUT_SIZE, self.INPUT_SIZE):
            raise ValueError(
                f"MaxViT-Tiny 384 requires {self.INPUT_SIZE}x{self.INPUT_SIZE} "
                f"input; received {tuple(image.shape[-2:])}"
            )
        return self.net(image)
