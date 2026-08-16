from __future__ import annotations

from typing import Any, Dict

import albumentations as A
import numpy as np
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_compression(probability: float):
    try:
        return A.ImageCompression(quality_range=(40, 100), p=probability)
    except TypeError:
        return A.ImageCompression(
            quality_lower=40,
            quality_upper=100,
            p=probability,
        )


def _shared_augmentation():
    """DeepfakeBench-style augmentation applied once to the shared RGB image."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-10, 10), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.1),
                        contrast_limit=(-0.1, 0.1),
                        p=1.0,
                    ),
                    A.FancyPCA(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ],
                p=0.5,
            ),
            _image_compression(0.5),
        ]
    )


class SharedBranchTransforms:
    """Keep all stochastic augmentation aligned across BAD, SFE, and GAD."""

    def __init__(self, sfe_processor, image_size: int, training: bool, bad_encoder):
        self.augmentation = _shared_augmentation() if training else None
        self.sfe_processor = sfe_processor

        bad_config = resolve_model_data_config(bad_encoder)
        self.bad_transform = create_transform(**bad_config, is_training=False)
        self.gad_transform = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size))),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __call__(self, image: Image.Image) -> Dict[str, Any]:
        if self.augmentation is not None:
            rgb = np.asarray(image.convert("RGB"))
            rgb = self.augmentation(image=rgb)["image"]
            image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")

        sfe = self.sfe_processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        return {
            "bad": self.bad_transform(image),
            "sfe": sfe,
            "gad": self.gad_transform(image),
        }


def build_branch_transforms(
    sfe_processor,
    image_size: int,
    training: bool,
    bad_encoder=None,
) -> SharedBranchTransforms:
    if bad_encoder is None:
        raise ValueError("bad_encoder is required for BAD preprocessing")
    return SharedBranchTransforms(
        sfe_processor=sfe_processor,
        image_size=int(image_size),
        training=bool(training),
        bad_encoder=bad_encoder,
    )

