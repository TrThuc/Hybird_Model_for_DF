from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SiglipVisionModel


class FrozenSigLIP2(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()

        # SigLIP 2 FixRes checkpoints use the backward-compatible SigLIP
        # architecture in Transformers (model_type="siglip").
        self.encoder = SiglipVisionModel.from_pretrained(
            model_name,
            attn_implementation="sdpa",
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.feature_dim = int(self.encoder.config.hidden_size)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.encoder.eval()

    def train(self, mode: bool = True):
        # Nhánh SFE luôn frozen và luôn ở eval mode.
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=images, return_dict=True)
        features = outputs.pooler_output

        if features.ndim == 1:
            features = features.unsqueeze(0)
        elif features.ndim > 2:
            features = features.flatten(start_dim=1)

        return features


