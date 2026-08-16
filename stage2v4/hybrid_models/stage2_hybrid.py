from __future__ import annotations

import torch
import torch.nn as nn


def ensure_2d_feature(feature: torch.Tensor) -> torch.Tensor:
    if not isinstance(feature, torch.Tensor):
        raise TypeError(
            f"Feature phải là torch.Tensor, nhận được {type(feature)!r}"
        )

    if feature.ndim == 1:
        feature = feature.unsqueeze(0)
    elif feature.ndim > 2:
        feature = feature.flatten(start_dim=1)

    if feature.ndim != 2:
        raise RuntimeError(
            "Feature không thể chuẩn hóa về [batch, dim]: "
            f"shape={tuple(feature.shape)}"
        )

    return feature


class Projection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(ensure_2d_feature(features))


class Stage2Hybrid(nn.Module):
    def __init__(
        self,
        bad: nn.Module,
        sfe: nn.Module,
        gad: nn.Module,
        bad_dim: int,
        sfe_dim: int,
        gad_dim: int,
        projection_dim: int = 512,
        dropout: float = 0.3,
        bad_branch_dropout: float = 0.0,
        sfe_branch_dropout: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.bad = bad
        self.sfe = sfe
        self.gad = gad
        self.bad_branch_dropout = float(bad_branch_dropout)
        self.sfe_branch_dropout = float(sfe_branch_dropout)
        if not 0.0 <= self.bad_branch_dropout < 1.0 or not 0.0 <= self.sfe_branch_dropout < 1.0:
            raise ValueError("Branch dropout must be in [0, 1).")

        self.bad_projection = Projection(bad_dim, projection_dim, dropout)
        self.sfe_projection = Projection(sfe_dim, projection_dim, dropout)
        self.gad_projection = Projection(gad_dim, projection_dim, dropout)

        # Aux classifiers cho cả 3 nhánh: mỗi nhánh có tín hiệu giám sát riêng,
        # đảm bảo không nhánh nào bị classifier fusion "nuốt mất" đặc trưng.
        # Inference: ensemble mean của 4 logits (3 aux + 1 fusion).
        self.bad_aux_classifier = nn.Linear(projection_dim, num_classes)
        self.sfe_aux_classifier = nn.Linear(projection_dim, num_classes)
        self.gad_aux_classifier = nn.Linear(projection_dim, num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),
        )

    def _maybe_drop_branch(self, x: torch.Tensor, probability: float) -> torch.Tensor:
        if not self.training or probability == 0.0:
            return x
        mask = torch.rand(x.shape[0], 1, device=x.device) < probability
        return x.masked_fill(mask, 0.0)

    def train(self, mode: bool = True):
        super().train(mode)

        # BAD và SFE frozen.
        self.bad.eval()
        self.sfe.eval()

        # GAD và toàn bộ fusion head được train.
        self.gad.train(mode)
        self.bad_projection.train(mode)
        self.sfe_projection.train(mode)
        self.gad_projection.train(mode)
        self.bad_aux_classifier.train(mode)
        self.sfe_aux_classifier.train(mode)
        self.gad_aux_classifier.train(mode)
        self.classifier.train(mode)

        return self

    def forward(
        self,
        bad_images: torch.Tensor,
        sfe_images: torch.Tensor,
        gad_images: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            bad_features = ensure_2d_feature(self.bad(bad_images))
            sfe_features = ensure_2d_feature(self.sfe(sfe_images))

        gad_features = ensure_2d_feature(self.gad(gad_images))

        bad_projected = self._maybe_drop_branch(
            self.bad_projection(bad_features), self.bad_branch_dropout
        )
        sfe_projected = self._maybe_drop_branch(
            self.sfe_projection(sfe_features), self.sfe_branch_dropout
        )
        gad_projected = self.gad_projection(gad_features)

        batch_sizes = {
            bad_projected.shape[0],
            sfe_projected.shape[0],
            gad_projected.shape[0],
        }

        if len(batch_sizes) != 1:
            raise RuntimeError(
                "Batch size giữa các nhánh không khớp: "
                f"BAD={tuple(bad_projected.shape)}, "
                f"SFE={tuple(sfe_projected.shape)}, "
                f"GAD={tuple(gad_projected.shape)}"
            )

        fused = torch.cat([bad_projected, sfe_projected, gad_projected], dim=1)
        fusion_logits = self.classifier(fused)

        if return_aux:
            bad_aux = self.bad_aux_classifier(bad_projected)
            sfe_aux = self.sfe_aux_classifier(sfe_projected)
            gad_aux = self.gad_aux_classifier(gad_projected)
            return fusion_logits, bad_aux, sfe_aux, gad_aux

        return fusion_logits
