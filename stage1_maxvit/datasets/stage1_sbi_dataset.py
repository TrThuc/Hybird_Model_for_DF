from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from sbi.sbi_api import SBI_API


logger = logging.getLogger(__name__)


class Stage1SBIDataset(Dataset):
    """FF++ real-face dataset using DeepfakeBench-preprocessed frames/landmarks."""

    def __init__(
        self,
        root: str,
        phase: str,
        compression: str = "c23",
        frames_per_video: int = 8,
        source_resolution: int = 256,
        model_resolution: int = 256,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        max_retries: int = 5,
    ) -> None:
        if phase != "train":
            raise ValueError(
                "Stage1SBIDataset is training-only; use "
                "FFPPEvaluationDataset for validation or testing"
            )
        if frames_per_video <= 0:
            raise ValueError("frames_per_video must be positive")
        if source_resolution <= 0 or model_resolution <= 0:
            raise ValueError("Image resolutions must be positive")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")

        self.root = Path(root)
        self.phase = phase
        self.compression = compression
        self.frames_per_video = int(frames_per_video)
        self.source_resolution = int(source_resolution)
        self.model_resolution = int(model_resolution)
        self.mean = mean
        self.std = std
        self.max_retries = int(max_retries)
        self.retry_count = 0

        split_path = self.root / f"{phase}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        split_pairs = json.loads(split_path.read_text(encoding="utf-8"))
        video_ids = {
            str(video_id).zfill(3)
            for pair in split_pairs
            for video_id in pair
        }

        frames_root = (
            self.root
            / "original_sequences"
            / "youtube"
            / compression
            / "frames"
        )
        landmarks_root = (
            self.root
            / "original_sequences"
            / "youtube"
            / compression
            / "landmarks"
        )

        if not frames_root.exists():
            raise FileNotFoundError(f"Missing frames directory: {frames_root}")
        if not landmarks_root.exists():
            raise FileNotFoundError(f"Missing landmarks directory: {landmarks_root}")

        self.samples: List[Tuple[Path, Path, str]] = []
        for video_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
            video_id = video_dir.name[:3]
            if video_id not in video_ids:
                continue

            frame_paths = sorted(
                video_dir.glob("*.png"),
                key=lambda path: int(path.stem),
            )
            frame_paths = self._even_sample(frame_paths, self.frames_per_video)

            for frame_path in frame_paths:
                landmark_path = (
                    landmarks_root
                    / video_dir.name
                    / f"{frame_path.stem}.npy"
                )
                if landmark_path.exists():
                    self.samples.append(
                        (frame_path, landmark_path, video_dir.name)
                    )
                else:
                    logger.warning("Missing landmark file: %s", landmark_path)

        if not self.samples:
            raise RuntimeError(
                f"No valid {phase} samples found under {frames_root}"
            )

        self.sbi = SBI_API(
            phase="train",
            image_size=self.source_resolution,
        )

        logger.info(
            "Loaded %d source samples for phase=%s, compression=%s",
            len(self.samples),
            self.phase,
            self.compression,
        )

    @staticmethod
    def _even_sample(paths: List[Path], n: int) -> List[Path]:
        if len(paths) <= n:
            return paths
        indices = np.linspace(0, len(paths) - 1, n, dtype=int)
        return [paths[index] for index in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        frame_path, landmark_path, _ = self.samples[index]

        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {frame_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = rgb.shape[:2]

        if original_w <= 0 or original_h <= 0:
            raise ValueError(f"Invalid image size: {frame_path}")

        if (
            original_w != self.source_resolution
            or original_h != self.source_resolution
        ):
            rgb = cv2.resize(
                rgb,
                (self.source_resolution, self.source_resolution),
                interpolation=cv2.INTER_CUBIC,
            )

        landmarks = np.load(str(landmark_path)).astype(np.float32)
        if landmarks.ndim == 3 and landmarks.shape[0] == 1:
            landmarks = landmarks[0]

        if landmarks.shape != (81, 2):
            raise ValueError(
                f"Unexpected landmark shape {landmarks.shape}: {landmark_path}"
            )
        if not np.isfinite(landmarks).all():
            raise ValueError(f"Non-finite landmarks: {landmark_path}")

        if (
            original_w != self.source_resolution
            or original_h != self.source_resolution
        ):
            landmarks[:, 0] *= self.source_resolution / original_w
            landmarks[:, 1] *= self.source_resolution / original_h

        tolerance = 2.0
        if (
            landmarks[:, 0].min() < -tolerance
            or landmarks[:, 1].min() < -tolerance
            or landmarks[:, 0].max() >= self.source_resolution + tolerance
            or landmarks[:, 1].max() >= self.source_resolution + tolerance
        ):
            raise ValueError(
                f"Landmarks outside image boundary: {landmark_path}"
            )

        fake, real = self.sbi(rgb.copy(), landmarks.copy())
        if fake is None or real is None:
            raise RuntimeError(
                f"SBI generation returned None for {frame_path}"
            )
        if fake.shape != rgb.shape or real.shape != rgb.shape:
            raise ValueError(
                f"Unexpected SBI output shape for {frame_path}: "
                f"real={real.shape}, fake={fake.shape}, expected={rgb.shape}"
            )

        return real, fake

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if self.model_resolution != self.source_resolution:
            image = cv2.resize(
                image,
                (self.model_resolution, self.model_resolution),
                interpolation=cv2.INTER_CUBIC,
            )

        tensor = TF.to_tensor(np.ascontiguousarray(image))
        return TF.normalize(tensor, self.mean, self.std)

    def __getitem__(self, index: int) -> Dict[str, object]:
        original_index = int(index)
        errors: List[str] = []

        for attempt in range(1, self.max_retries + 1):
            try:
                real, fake = self._load_pair(index)
                return {
                    "real": self._to_tensor(real),
                    "fake": self._to_tensor(fake),
                    "video": self.samples[index][2],
                }
            except Exception as exc:
                self.retry_count += 1
                message = (
                    f"phase={self.phase}, original_index={original_index}, "
                    f"current_index={index}, attempt={attempt}/"
                    f"{self.max_retries}, error={exc}"
                )
                errors.append(message)
                logger.warning("SBI dataset retry: %s", message)
                index = random.randrange(len(self.samples))

        raise RuntimeError(
            "Failed to load a valid SBI sample after "
            f"{self.max_retries} attempts:\n" + "\n".join(errors)
        )

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, object]],
    ) -> Dict[str, object]:
        real = torch.stack([item["real"] for item in batch])
        fake = torch.stack([item["fake"] for item in batch])

        images = torch.cat([real, fake], dim=0)
        labels = torch.cat(
            [
                torch.zeros(len(batch), dtype=torch.long),
                torch.ones(len(batch), dtype=torch.long),
            ],
            dim=0,
        )
        videos = [item["video"] for item in batch] * 2

        return {
            "image": images,
            "label": labels,
            "video": videos,
        }
