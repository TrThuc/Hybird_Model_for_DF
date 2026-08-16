import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from datasets import Stage1SBIDataset
from utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/stage1_maxvit.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ds = Stage1SBIDataset(
        root=cfg["data"]["root"],
        phase="train",
        compression=cfg["data"]["compression"],
        frames_per_video=cfg["data"]["frames_per_video"]["train"],
        source_resolution=cfg["data"]["source_resolution"],
        model_resolution=cfg["data"]["model_resolution"],
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn, num_workers=0)
    batch = next(iter(loader))
    print("dataset length:", len(ds))
    print("image shape:", tuple(batch["image"].shape))
    print("labels:", batch["label"].tolist())
    print("finite:", bool(batch["image"].isfinite().all()))


if __name__ == "__main__":
    main()
