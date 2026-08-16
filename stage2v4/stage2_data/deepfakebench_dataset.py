from __future__ import annotations

import json

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _label_to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if value not in (0, 1):
            raise ValueError(f"Unsupported numeric label: {value}")
        return value

    text = str(value).strip().lower()

    if text in {"0", "real", "original", "authentic", "ff-real"}:
        return 0

    if text in {
        "1",
        "fake",
        "manipulated",
        "deepfake",
        "ff-df",
        "ff-f2f",
        "ff-fs",
        "ff-nt",
        "ff-fh",
    }:
        return 1

    if any(
        token in text
        for token in (
            "fake",
            "deepfake",
            "faceswap",
            "face2face",
            "neuraltextures",
            "ff-df",
            "ff-f2f",
            "ff-fs",
            "ff-nt",
            "ff-fh",
            "faceshifter",
        )
    ):
        return 1

    if "real" in text or "original" in text:
        return 0

    raise ValueError(f"Cannot map label to real/fake: {value!r}")


def _contains_token(
    path_tokens: Sequence[str],
    expected: Optional[str],
) -> bool:
    if expected is None:
        return True

    expected = str(expected).strip().lower()
    return any(
        str(token).strip().lower() == expected
        for token in path_tokens
    )


def _matches_methods(
    path_tokens: Sequence[str],
    methods: Sequence[str],
) -> bool:
    if not methods:
        return True

    token_set = {
        str(token).strip().lower()
        for token in path_tokens
    }

    return any(
        str(method).strip().lower() in token_set
        for method in methods
    )


def _resolve_frame_path(frame: str, dataset_root: str) -> str:
    # JSON files may have been generated on Windows and therefore contain
    # backslash separators.  Normalize them before resolving on Linux/Kaggle.
    normalized_frame = str(frame).replace("\\", "/")
    path = Path(normalized_frame)

    if path.is_absolute() or not dataset_root:
        return str(path)

    return str((Path(dataset_root) / path).resolve())


def collect_records(
    json_path: str,
    dataset_root: str = "",
    split: Optional[str] = None,
    compression: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
    max_frames_per_video: Optional[int] = None,
) -> List[Dict[str, Any]]:
    with Path(json_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    methods = list(methods or [])
    records: List[Dict[str, Any]] = []

    def walk(node: Any, path_tokens: Tuple[str, ...]) -> None:
        if isinstance(node, dict):
            if (
                "label" in node
                and "frames" in node
                and isinstance(node["frames"], list)
            ):
                if not _contains_token(path_tokens, split):
                    return

                if compression is not None and not _contains_token(
                    path_tokens,
                    compression,
                ):
                    return

                if not _matches_methods(path_tokens, methods):
                    return

                label = _label_to_int(node["label"])
                frames = node["frames"]

                if max_frames_per_video is not None and int(max_frames_per_video) > 0:
                    target = int(max_frames_per_video)
                    if len(frames) > target:
                        indices = np.linspace(0, len(frames) - 1, target, dtype=int)
                        frames = [frames[int(index)] for index in indices]

                method = next(
                    (
                        str(token)
                        for token in path_tokens
                        if str(token).lower().startswith("ff-")
                    ),
                    str(node["label"]),
                )

                video_id = "/".join(path_tokens)

                for frame in frames:
                    records.append(
                        {
                            "image_path": _resolve_frame_path(
                                str(frame),
                                dataset_root,
                            ),
                            "label": label,
                            "video_id": video_id,
                            "source": method,
                        }
                    )
                return

            for key, value in node.items():
                walk(value, path_tokens + (str(key),))

        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path_tokens + (str(index),))

    walk(data, tuple())

    if not records:
        raise RuntimeError(
            "Không tìm thấy frame phù hợp trong JSON. "
            f"json={json_path}, split={split}, "
            f"compression={compression}, methods={methods}"
        )

    return records


class DeepfakeBenchDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        transforms: Dict[str, Any],
        dataset_root: str = "",
        split: Optional[str] = None,
        compression: Optional[str] = None,
        methods: Optional[Sequence[str]] = None,
        max_frames_per_video: Optional[int] = None,
    ) -> None:
        self.transforms = transforms
        self.records = collect_records(
            json_path=json_path,
            dataset_root=dataset_root,
            split=split,
            compression=compression,
            methods=methods,
            max_frames_per_video=max_frames_per_video,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image_path = record["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Không đọc được ảnh: {image_path}"
            ) from exc

        branch_images = self.transforms(image)
        return {
            **branch_images,
            "label": int(record["label"]),
            "video_id": record["video_id"],
            "image_path": image_path,
            "source": record["source"],
        }
