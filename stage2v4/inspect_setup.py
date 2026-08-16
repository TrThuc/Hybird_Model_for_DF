from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from stage2_data import DeepfakeBenchDataset, build_branch_transforms
from stage2_utils import build_stage2, load_yaml, resolve_path, seed_everything


def absolutize(cfg, base: Path):
    for section, keys in {
        "data": ["train_json", "val_json", "dataset_root"],
        "bad": ["checkpoint"],
        "gad": ["checkpoint"],
        "output": ["directory"],
    }.items():
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value:
                cfg[section][key] = resolve_path(value, base)
    return cfg


def report(name, dataset):
    labels = Counter(record["label"] for record in dataset.records)
    sources = Counter(record["source"] for record in dataset.records)
    missing = sum(not Path(record["image_path"]).is_file() for record in dataset.records)
    print(f"{name}: frames={len(dataset)}, labels={dict(labels)}, sources={dict(sources)}, missing={missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = absolutize(load_yaml(str(config_path)), config_path.parent.parent)
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = build_stage2(cfg).to(device)
    transforms = build_branch_transforms(
        model.sfe.processor,
        int(cfg["data"]["image_size"]),
        False,
        bad_encoder=model.bad.encoder,
    )
    common = dict(
        transforms=transforms,
        dataset_root=cfg["data"].get("dataset_root", ""),
        compression=cfg["data"].get("compression"),
        methods=cfg["data"].get("methods", []),
    )
    train = DeepfakeBenchDataset(
        json_path=cfg["data"]["train_json"],
        split=cfg["data"]["train_split"],
        max_frames_per_video=cfg["data"].get("max_frames_per_video_train"),
        **common,
    )
    val = DeepfakeBenchDataset(
        json_path=cfg["data"]["val_json"],
        split=cfg["data"]["val_split"],
        max_frames_per_video=cfg["data"].get("max_frames_per_video_val"),
        **common,
    )
    report("train", train)
    report("val", val)
    sample = val[0] if len(val) else train[0]
    model.eval()
    with torch.inference_mode():
        logits = model(
            sample["bad"].unsqueeze(0).to(device),
            sample["sfe"].unsqueeze(0).to(device),
            sample["gad"].unsqueeze(0).to(device),
        )
    print("BAD feature dim:", model.bad.feature_dim)
    print("SFE feature dim:", model.sfe.feature_dim)
    print("GAD feature dim:", model.gad.feature_dim)
    print("Forward logits:", tuple(logits.shape))


if __name__ == "__main__":
    main()
