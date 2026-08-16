from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage2_data import DeepfakeBenchDataset, build_branch_transforms
from stage2_utils import build_stage2, resolve_path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def safe_auc(labels, probabilities):
    return float("nan") if len(set(labels)) < 2 else float(roc_auc_score(labels, probabilities))


def safe_eer(labels, probabilities):
    if len(set(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, probabilities, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def level_metrics(labels, probabilities, prefix):
    predictions = [int(value >= 0.5) for value in probabilities]
    class_acc = {}
    for class_id, name in ((0, "real"), (1, "fake")):
        values = [
            int(prediction == label)
            for label, prediction in zip(labels, predictions)
            if label == class_id
        ]
        class_acc[name] = float(np.mean(values)) if values else float("nan")
    ap = (
        float(average_precision_score(labels, probabilities))
        if len(set(labels)) > 1 else float("nan")
    )
    return {
        f"{prefix}_acc": float(accuracy_score(labels, predictions)),
        f"{prefix}_bal_acc": float(balanced_accuracy_score(labels, predictions)),
        f"{prefix}_acc_real": class_acc["real"],
        f"{prefix}_acc_fake": class_acc["fake"],
        f"{prefix}_auc": safe_auc(labels, probabilities),
        f"{prefix}_eer": safe_eer(labels, probabilities),
        f"{prefix}_ap": ap,
        f"{prefix}_precision": float(precision_score(labels, predictions, zero_division=0)),
        f"{prefix}_recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def aggregate_video_rows(frame_rows):
    grouped = defaultdict(list)
    labels = {}
    for row in frame_rows:
        video_id = row["video_id"]
        grouped[video_id].append(float(row["score"]))
        previous = labels.setdefault(video_id, int(row["label"]))
        if previous != int(row["label"]):
            raise ValueError(f"Inconsistent labels for video: {video_id}")
    return [
        {
            "video_id": video_id,
            "label": labels[video_id],
            "score": float(np.mean(scores)),
        }
        for video_id, scores in grouped.items()
    ]


def compute_metrics(frame_rows, mean_loss):
    video_rows = aggregate_video_rows(frame_rows)
    frame_labels = [int(row["label"]) for row in frame_rows]
    frame_probs = [float(row["score"]) for row in frame_rows]
    video_labels = [int(row["label"]) for row in video_rows]
    video_probs = [float(row["score"]) for row in video_rows]
    metrics = {
        **level_metrics(frame_labels, frame_probs, "frame"),
        **level_metrics(video_labels, video_probs, "video"),
        "loss": float(mean_loss),
        "num_frames": len(frame_rows),
        "num_videos": len(video_rows),
    }
    return metrics, video_rows


@torch.inference_mode()
def evaluate(model, loader, device, amp):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    rows = []
    loss_sum = 0.0
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp and device.type == "cuda" else nullcontext()
    )
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        targets = batch["label"].to(device, non_blocking=True)
        with autocast():
            fusion, bad_aux, sfe_aux, gad_aux = model(
                batch["bad"].to(device, non_blocking=True),
                batch["sfe"].to(device, non_blocking=True),
                batch["gad"].to(device, non_blocking=True),
                return_aux=True,
            )
            # Keep cross-test inference identical to train/validation metrics.
            logits = (fusion + bad_aux + sfe_aux + gad_aux) / 4.0
            loss = criterion(logits, targets)
        probabilities = logits.softmax(dim=1)[:, 1].float().cpu().tolist()
        loss_sum += float(loss.item())
        for path, video, source, label, score in zip(
            batch["image_path"], batch["video_id"], batch["source"],
            targets.cpu().tolist(), probabilities,
        ):
            rows.append({
                "image_path": path, "video_id": video, "source": source,
                "label": int(label), "score": float(score),
            })
    metrics, video_rows = compute_metrics(rows, loss_sum / max(1, len(rows)))
    return metrics, rows, video_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_cross_model_config(config, project_root):
    if "architecture" not in config:
        raise KeyError("cross-test config must contain an 'architecture' section")

    cfg = copy.deepcopy(config["architecture"])
    cfg.setdefault("data", {})
    cfg.setdefault("fusion", {})
    cfg["data"].setdefault("image_size", 384)
    cfg["fusion"].setdefault("projection_dim", 512)
    cfg["fusion"].setdefault("dropout", 0.3)
    cfg["fusion"].setdefault("bad_branch_dropout", 0.0)
    cfg["fusion"].setdefault("sfe_branch_dropout", 0.0)
    cfg["fusion"].setdefault("num_classes", 2)

    for section in ("bad", "gad"):
        checkpoint = cfg.get(section, {}).get("checkpoint")
        if checkpoint:
            cfg[section]["checkpoint"] = resolve_path(checkpoint, project_root)
    return cfg


def load_stage2_weights(model, checkpoint_path):
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {checkpoint_file}")
    checkpoint = safe_load(str(checkpoint_file))
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    print(f"Loaded Stage-2 checkpoint: {checkpoint_file}")
    if epoch is not None:
        print(f"Checkpoint epoch: {epoch}")


def main(config_path):
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed_everything(int(config.get("seed", 5)))
    force_cpu = bool(config.get("cpu", False))
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    print(f"Device: {device}")
    model_cfg = build_cross_model_config(config, project_root)
    model = build_stage2(model_cfg).to(device)
    load_stage2_weights(
        model,
        resolve_path(config["checkpoint"], project_root),
    )

    transforms = build_branch_transforms(
        model.sfe.processor,
        int(model_cfg["data"]["image_size"]),
        training=False,
        bad_encoder=model.bad.encoder,
    )

    output_dir = Path(resolve_path(config["output_dir"], project_root))
    summary = {}
    test_cfg = config["test"]
    for item in config["datasets"]:
        name = item["name"]
        print("\n" + "=" * 80)
        print(f"Dataset: {name}")
        root = resolve_path(item.get("dataset_root", "."), project_root)
        dataset = DeepfakeBenchDataset(
            transforms=transforms,
            json_path=resolve_path(item["json"], project_root),
            dataset_root=root,
            split=item.get("split", "test"),
            compression=item.get("compression"),
            methods=item.get("methods", []),
            max_frames_per_video=test_cfg.get("frames_per_video", 32),
        )
        print(f"Frames: {len(dataset):,}")
        workers = int(test_cfg.get("workers", 2))
        loader = DataLoader(
            dataset,
            batch_size=int(test_cfg.get("batch_size", 2)),
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(int(config.get("seed", 5))),
        )
        metrics, frame_rows, video_rows = evaluate(
            model, loader, device, bool(test_cfg.get("amp", True))
        )
        summary[name] = metrics
        print(json.dumps(metrics, indent=2, allow_nan=True))
        safe_name = name.replace("/", "_").replace("\\", "_")
        write_csv(output_dir / f"{safe_name}_frame_predictions.csv", frame_rows)
        write_csv(output_dir / f"{safe_name}_video_predictions.csv", video_rows)

        # Save accumulated metrics after every completed dataset.
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "summary_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
        write_csv(
            output_dir / "summary_metrics.csv",
            [
                {"dataset": dataset_name, **dataset_metrics}
                for dataset_name, dataset_metrics in summary.items()
            ],
        )
        print(f"Saved results for {name} to: {output_dir}")
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-test the Stage-2 hybrid detector")
    parser.add_argument("--config", default="configs/cross_test.yaml")
    args = parser.parse_args()
    main(args.config)
