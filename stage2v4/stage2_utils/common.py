from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str, base: Path) -> str:
    value = Path(path)
    if value.is_absolute():
        return str(value)
    return str((base / value).resolve())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
