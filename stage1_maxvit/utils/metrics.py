from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def binary_metrics(labels: Iterable[int], probabilities: Iterable[float]) -> Dict[str, float]:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)
    result = {
        "acc": float(accuracy_score(y, pred)),
        "ap": float(average_precision_score(y, p)),
    }
    result["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return result
