from __future__ import annotations

from pathlib import Path
from typing import Dict

import timm
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
    for key in ("model", "state_dict", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


class FrozenBAD(nn.Module):
    def __init__(self, model_name: str, checkpoint_path: str) -> None:
        super().__init__()
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Không tìm thấy BAD checkpoint: {checkpoint_path}")

        network = timm.create_model(model_name, pretrained=False, num_classes=2)
        checkpoint = _safe_torch_load(checkpoint_path)
        state = _extract_state_dict(checkpoint)
        cleaned_state = {}
        for key, value in state.items():
            key = key.removeprefix("module.").removeprefix("net.")
            cleaned_state[key] = value
        network.load_state_dict(cleaned_state, strict=True)

        print(f"Loaded MaxViT-Tiny BAD checkpoint: {checkpoint_path}")
        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            print(f"BAD checkpoint epoch: {checkpoint['epoch']}")
        print("BAD checkpoint matched with strict=True")

        self.feature_dim = int(network.num_features)
        network.reset_classifier(0)
        self.encoder = network
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if features.ndim > 2:
            features = features.flatten(1)
        return features
