from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from sbi.sbi_api import SBI_API, dynamic_blend
from sbi.bi_online_generation import random_get_hull


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = ensure_uint8(image)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    if image.max() <= 1.0:
        image = image * 255.0
    cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = ensure_uint8(image).astype(np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask = np.clip(mask, 0.0, 1.0)[..., None]

    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay = overlay * (1.0 - 0.45 * mask) + red * (0.45 * mask)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def annotate(image: np.ndarray, text: str) -> np.ndarray:
    image = ensure_uint8(image).copy()
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def make_panel(stages: Dict[str, np.ndarray], output_path: Path) -> None:
    ordered = [
        ("01_input", "Input I"),
        ("02_initial_mask", "Initial mask"),
        ("03_source_after_T", "Source after T"),
        ("04_target_after_T", "Target"),
        ("05_affine_source", "Affine source"),
        ("06_affine_mask", "Affine mask"),
        ("07_blend_mask", "Blend mask"),
        ("08_fake_before_common_aug", "SBI before common aug"),
        ("09_real_after_common_aug", "Final real"),
        ("10_fake_after_common_aug", "Final SBI"),
    ]

    tiles = []
    target_size = 256
    for key, title in ordered:
        image = stages[key]
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.max() <= 1.0:
            image = image * 255.0
        image = ensure_uint8(image)
        image = cv2.resize(
            image,
            (target_size, target_size),
            interpolation=cv2.INTER_NEAREST if "mask" in key else cv2.INTER_CUBIC,
        )
        tiles.append(annotate(image, title))

    rows = []
    for start in range(0, len(tiles), 2):
        pair = tiles[start:start + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows.append(np.concatenate(pair, axis=1))
    panel = np.concatenate(rows, axis=0)
    save_rgb(output_path, panel)


def export_one_sample(image_path: Path, landmark_path: Path, output_dir: Path, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    landmarks = np.load(str(landmark_path)).astype(np.float32)
    if landmarks.ndim == 3 and landmarks.shape[0] == 1:
        landmarks = landmarks[0]
    if landmarks.shape != (81, 2):
        raise ValueError(f"Expected landmark shape (81, 2), got {landmarks.shape}")

    api = SBI_API(phase="train", image_size=image.shape[0])

    working_landmarks = landmarks.copy()
    use_68 = np.random.rand() < 0.25
    if use_68:
        working_landmarks = working_landmarks[:68]

    hull_type = random.choice([0, 1, 2, 3])
    initial_mask = random_get_hull(working_landmarks, image, hull_type)[:, :, 0]

    source = image.copy()
    target = image.copy()
    transform_branch = "source" if np.random.rand() < 0.5 else "target"
    if transform_branch == "source":
        source = api.source_transforms(image=source.astype(np.uint8))["image"]
    else:
        target = api.source_transforms(image=target.astype(np.uint8))["image"]

    source_after_t = source.copy()
    target_after_t = target.copy()

    affine_source, affine_mask = api.randaffine(source.copy(), initial_mask.copy())
    fake_before_common_aug, blend_mask = dynamic_blend(affine_source, target, affine_mask)
    fake_before_common_aug = ensure_uint8(fake_before_common_aug)
    real_before_common_aug = ensure_uint8(target)

    transformed = api.transforms(
        image=fake_before_common_aug.astype(np.uint8),
        image1=real_before_common_aug.astype(np.uint8),
    )
    fake_after_common_aug = transformed["image"]
    real_after_common_aug = transformed["image1"]

    stages = {
        "01_input": image,
        "02_initial_mask": initial_mask,
        "03_source_after_T": source_after_t,
        "04_target_after_T": target_after_t,
        "05_affine_source": affine_source,
        "06_affine_mask": affine_mask,
        "07_blend_mask": blend_mask,
        "08_fake_before_common_aug": fake_before_common_aug,
        "09_real_after_common_aug": real_after_common_aug,
        "10_fake_after_common_aug": fake_after_common_aug,
    }

    sample_dir = output_dir / image_path.parent.name / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(sample_dir / "01_input_I.png", image)
    save_gray(sample_dir / "02_initial_mask.png", initial_mask)
    save_rgb(sample_dir / "02b_initial_mask_overlay.png", overlay_mask(image, initial_mask))
    save_rgb(sample_dir / "03_source_after_T.png", source_after_t)
    save_rgb(sample_dir / "04_target_after_T.png", target_after_t)
    save_rgb(sample_dir / "05_source_after_affine.png", affine_source)
    save_gray(sample_dir / "06_mask_after_affine.png", affine_mask)
    save_gray(sample_dir / "07_blend_mask.png", blend_mask)
    save_rgb(sample_dir / "07b_blend_mask_overlay.png", overlay_mask(target, blend_mask))
    save_rgb(sample_dir / "08_fake_before_common_aug.png", fake_before_common_aug)
    save_rgb(sample_dir / "09_real_after_common_aug.png", real_after_common_aug)
    save_rgb(sample_dir / "10_fake_after_common_aug.png", fake_after_common_aug)
    make_panel(stages, sample_dir / "00_pipeline_panel.png")

    metadata = {
        "image_path": str(image_path),
        "landmark_path": str(landmark_path),
        "seed": seed,
        "hull_type": hull_type,
        "hull_name": ["dfl_full", "extended", "components", "facehull"][hull_type],
        "landmarks_used": 68 if use_68 else 81,
        "T_applied_to": transform_branch,
        "output_directory": str(sample_dir),
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved report sample to: {sample_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export each SBI generation stage for reports.")
    parser.add_argument("--image", required=True, help="Path to a frame PNG")
    parser.add_argument("--landmark", required=True, help="Path to a landmark NPY")
    parser.add_argument("--output", default="outputs/sbi_report_samples")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    export_one_sample(
        image_path=Path(args.image),
        landmark_path=Path(args.landmark),
        output_dir=Path(args.output),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()