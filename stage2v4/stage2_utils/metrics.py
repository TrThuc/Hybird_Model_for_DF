from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_auc(labels, probabilities) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def _safe_eer(labels, probabilities) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, probabilities, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) * 0.5)


def _class_accuracy(labels, predictions, class_id: int) -> float:
    selected = [
        int(prediction == label)
        for label, prediction in zip(labels, predictions)
        if label == class_id
    ]
    return float(np.mean(selected)) if selected else float("nan")


def _level_metrics(labels, probabilities, prefix: str) -> Dict[str, float]:
    predictions = [int(value >= 0.5) for value in probabilities]
    return {
        f"{prefix}_acc": float(accuracy_score(labels, predictions)),
        f"{prefix}_bal_acc": float(balanced_accuracy_score(labels, predictions)),
        f"{prefix}_acc_real": _class_accuracy(labels, predictions, 0),
        f"{prefix}_acc_fake": _class_accuracy(labels, predictions, 1),
        f"{prefix}_auc": _safe_auc(labels, probabilities),
        f"{prefix}_eer": _safe_eer(labels, probabilities),
        f"{prefix}_ap": float(average_precision_score(labels, probabilities)),
        f"{prefix}_precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        f"{prefix}_recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
    }


def compute_metrics(
    labels: List[int],
    probabilities: List[float],
    video_ids: List[str],
) -> Dict[str, float]:
    if not (len(labels) == len(probabilities) == len(video_ids)):
        raise ValueError("labels, probabilities and video_ids must have equal length")
    if not labels:
        raise ValueError("Cannot compute metrics for an empty dataset")

    grouped_probs = defaultdict(list)
    grouped_labels = {}
    for label, probability, video_id in zip(labels, probabilities, video_ids):
        grouped_probs[video_id].append(probability)
        previous = grouped_labels.setdefault(video_id, label)
        if previous != label:
            raise ValueError(f"Inconsistent labels for video_id={video_id!r}")

    video_labels = []
    video_probs = []
    for video_id, values in grouped_probs.items():
        video_labels.append(grouped_labels[video_id])
        video_probs.append(float(np.mean(values)))

    return {
        **_level_metrics(labels, probabilities, "frame"),
        **_level_metrics(video_labels, video_probs, "video"),
    }
