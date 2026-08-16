from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


logger = logging.getLogger(__name__)


class FFPPEvaluationDataset(Dataset):
    """Deterministic FF++ real/fake frames for validation and testing.

    The official ``<phase>.json`` pair list is the split source of truth.
    Real videos use IDs from the pairs; manipulated videos must match either
    direction of one of those same pairs.
    """

    DEFAULT_MANIPULATIONS = (
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
    )

    def __init__(
        self,
        root: str,
        phase: str = "val",
        compression: str = "c23",
        frames_per_video: int = 8,
        model_resolution: int = 384,
        manipulations: Sequence[str] = DEFAULT_MANIPULATIONS,
        balance_fake_frames: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        if phase not in {"val", "test"}:
            raise ValueError("phase must be 'val' or 'test'")
        if frames_per_video <= 0 or model_resolution <= 0:
            raise ValueError("frames_per_video and model_resolution must be positive")

        self.root = Path(root)
        self.phase = phase
        self.compression = compression
        self.frames_per_video = int(frames_per_video)
        self.model_resolution = int(model_resolution)
        self.mean = mean
        self.std = std
        self.retry_count = 0

        split_path = self.root / f"{phase}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing official FF++ split: {split_path}")
        raw_pairs = json.loads(split_path.read_text(encoding="utf-8"))
        pairs = [(str(a).zfill(3), str(b).zfill(3)) for a, b in raw_pairs]
        real_ids = {video_id for pair in pairs for video_id in pair}
        fake_ids = {f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs}

        self.samples: List[Tuple[Path, int, str, str]] = []
        real_root = (
            self.root / "original_sequences" / "youtube" /
            compression / "frames"
        )
        self._add_videos(real_root, real_ids, label=0, manipulation="real")

        supported = set(self.DEFAULT_MANIPULATIONS)
        requested = tuple(manipulations)
        unknown = sorted(set(requested) - supported)
        if unknown:
            raise ValueError(f"Unsupported FF++ manipulations: {unknown}")
        for manipulation in requested:
            fake_root = (
                self.root / "manipulated_sequences" / manipulation /
                compression / "frames"
            )
            self._add_videos(fake_root, fake_ids, label=1, manipulation=manipulation)

        if balance_fake_frames:
            self._balance_fake_frames(requested)

        if not self.samples:
            raise RuntimeError(f"No FF++ {phase} evaluation samples found")
        real_count = sum(label == 0 for _, label, _, _ in self.samples)
        fake_count = len(self.samples) - real_count
        if not real_count or not fake_count:
            raise RuntimeError(f"FF++ {phase} must contain real and fake samples")
        logger.info(
            "Loaded FF++ %s evaluation: real=%d fake=%d total=%d",
            phase, real_count, fake_count, len(self.samples),
        )

    def _balance_fake_frames(self, manipulations: Sequence[str]) -> None:
        real_samples = [sample for sample in self.samples if sample[1] == 0]
        if not real_samples:
            raise RuntimeError("Cannot balance FF++ validation without real samples")
        if not manipulations:
            raise RuntimeError("Cannot balance FF++ validation without manipulations")

        target_total = len(real_samples)
        base, remainder = divmod(target_total, len(manipulations))
        balanced_fake: List[Tuple[Path, int, str, str]] = []
        for index, manipulation in enumerate(manipulations):
            candidates = [
                sample for sample in self.samples
                if sample[1] == 1 and sample[3] == manipulation
            ]
            target = base + (1 if index < remainder else 0)
            if len(candidates) < target:
                raise RuntimeError(
                    f"Not enough {manipulation} frames to balance validation: "
                    f"need={target}, found={len(candidates)}"
                )
            balanced_fake.extend(self._even_sample(candidates, target))

        self.samples = real_samples + balanced_fake

    @staticmethod
    def _even_sample(paths: List[Path], n: int) -> List[Path]:
        if len(paths) <= n:
            return paths
        indices = np.linspace(0, len(paths) - 1, n, dtype=int)
        return [paths[index] for index in indices]

    def _add_videos(
        self,
        frames_root: Path,
        allowed_names: set[str],
        label: int,
        manipulation: str,
    ) -> None:
        if not frames_root.exists():
            raise FileNotFoundError(f"Missing FF++ frames directory: {frames_root}")
        found_names = set()
        for video_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
            if video_dir.name not in allowed_names:
                continue
            found_names.add(video_dir.name)
            frames = sorted(video_dir.glob("*.png"), key=lambda path: int(path.stem))
            if not frames:
                logger.warning("No frames in FF++ video: %s", video_dir)
                continue
            for frame_path in self._even_sample(frames, self.frames_per_video):
                video_key = f"{manipulation}/{video_dir.name}"
                self.samples.append((frame_path, label, video_key, manipulation))
        missing = allowed_names - found_names
        if missing:
            logger.warning(
                "FF++ %s/%s missing %d split videos (first 10: %s)",
                self.phase, manipulation, len(missing), sorted(missing)[:10],
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path, label, video, manipulation = self.samples[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.model_resolution, self.model_resolution):
            rgb = cv2.resize(
                rgb,
                (self.model_resolution, self.model_resolution),
                interpolation=cv2.INTER_CUBIC,
            )
        image = TF.normalize(
            TF.to_tensor(np.ascontiguousarray(rgb)), self.mean, self.std
        )
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "video": video,
            "manipulation": manipulation,
        }
