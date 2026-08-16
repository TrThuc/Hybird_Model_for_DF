from __future__ import annotations

import numpy as np

from .DeepFakeMask import (
    dfl_full,
    extended,
    components,
    facehull,
)


def random_get_hull(
    landmark: np.ndarray,
    image: np.ndarray,
    hull_type: int,
) -> np.ndarray:
    """Create one of four SBI face masks from facial landmarks.

    Args:
        landmark: Facial landmarks with shape (68, 2) or (81, 2).
        image: RGB face image.
        hull_type:
            0 = dfl_full
            1 = extended
            2 = components
            3 = facehull

    Returns:
        Float mask with shape (H, W, 3), values in [0, 1].
    """

    landmark = np.asarray(
        landmark,
        dtype=np.int32,
    )

    if landmark.ndim != 2 or landmark.shape[1] != 2:
        raise ValueError(
            "landmark must have shape (N, 2), "
            f"received {landmark.shape}"
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape (H, W, 3), "
            f"received {image.shape}"
        )

    mask_classes = {
        0: dfl_full,
        1: extended,
        2: components,
        3: facehull,
    }

    if hull_type not in mask_classes:
        raise ValueError(
            f"Unsupported hull_type={hull_type}. "
            "Expected one of 0, 1, 2, 3."
        )

    mask_class = mask_classes[hull_type]
    mask = mask_class(
        landmarks=landmark,
        face=image,
        channels=3,
    ).mask

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    if mask.ndim != 3 or mask.shape[2] != 3:
        raise RuntimeError(
            "Generated mask must have shape (H, W, 3), "
            f"received {mask.shape}"
        )

    return mask / 255.0