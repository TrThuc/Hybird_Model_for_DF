from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_checkpoint(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    checkpoint = load_checkpoint(path)

    print("=" * 80)
    print("Checkpoint:", path)
    print("Type:", type(checkpoint))

    if not isinstance(checkpoint, dict):
        print("Checkpoint không phải dictionary.")
        return

    print("Keys:", list(checkpoint.keys()))

    for key in [
        "epoch",
        "lr_used",
        "lr_next",
        "train_loss",
        "val_loss",
        "train_metrics",
        "val_metrics",
        "resume_checkpoint",
    ]:
        if key in checkpoint:
            print(f"{key}: {checkpoint[key]}")

    config = checkpoint.get("config")
    if config is not None:
        print("\nCONFIG TRONG CHECKPOINT")
        print(config)

    state_dict = (
        checkpoint.get("model")
        or checkpoint.get("state_dict")
        or checkpoint.get("model_state_dict")
    )

    if state_dict is None:
        print("\nKhông tìm thấy model state_dict.")
        return

    print("\nSố tensor trong model:", len(state_dict))

    head_keys = [
        key
        for key in state_dict
        if "head" in key.lower()
        or "classifier" in key.lower()
        or "fc" in key.lower()
    ]

    print("\nCác khóa classifier/head:")
    for key in head_keys:
        tensor = state_dict[key]
        print(
            f"{key}: shape={tuple(tensor.shape)}, "
            f"mean={tensor.float().mean().item():.8f}, "
            f"std={tensor.float().std().item():.8f}"
        )


if __name__ == "__main__":
    main()