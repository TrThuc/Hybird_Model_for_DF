from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FFPPEvaluationDataset, Stage1SBIDataset
from stage1_models import MaxViTStage1
from optimizers import (
    SAM,
    WarmupCosineLR,
    disable_running_stats,
    enable_running_stats,
)
from utils import binary_metrics, load_config, seed_everything


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    logger = logging.getLogger("stage1_maxvit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def make_train_loader(cfg: Dict) -> DataLoader:
    dataset = Stage1SBIDataset(
        root=cfg["data"]["root"],
        phase="train",
        compression=cfg["data"]["compression"],
        frames_per_video=cfg["data"]["frames_per_video"]["train"],
        source_resolution=cfg["data"]["source_resolution"],
        model_resolution=cfg["data"]["model_resolution"],
        max_retries=int(cfg["data"].get("max_retries", 5)),
    )

    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["source_batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["workers"]),
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        persistent_workers=int(cfg["train"]["workers"]) > 0,
    )


def make_val_loader(cfg: Dict) -> DataLoader:
    dataset = FFPPEvaluationDataset(
        root=cfg["data"]["root"],
        phase="val",
        compression=cfg["data"]["compression"],
        frames_per_video=cfg["data"]["frames_per_video"]["val"],
        model_resolution=cfg["data"]["model_resolution"],
        manipulations=cfg.get("eval", {}).get(
            "manipulations", FFPPEvaluationDataset.DEFAULT_MANIPULATIONS
        ),
        balance_fake_frames=bool(
            cfg.get("eval", {}).get("balance_fake_frames", True)
        ),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["source_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(cfg["train"]["workers"]) > 0,
    )


def sam_step(
    model: nn.Module,
    optimizer: SAM,
    criterion: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    amp_enabled: bool,
) -> Tuple[float, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)

    enable_running_stats(model)
    logits_first = model(images)
    loss_first = criterion(logits_first, labels)

    if not torch.isfinite(loss_first):
        raise FloatingPointError(
            f"Non-finite first SAM loss: {loss_first.item()}"
        )

    loss_first.backward()
    optimizer.first_step(zero_grad=True)

    disable_running_stats(model)
    logits_second = model(images)
    loss_second = criterion(logits_second, labels)

    if not torch.isfinite(loss_second):
        raise FloatingPointError(
            f"Non-finite second SAM loss: {loss_second.item()}"
        )

    loss_second.backward()
    optimizer.second_step(zero_grad=True)
    enable_running_stats(model)

    return float(loss_first.item()), logits_first.detach()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, Dict[str, float]]:
    model.eval()

    losses: List[float] = []
    labels_all: List[int] = []
    probs_all: List[float] = []
    video_labels: Dict[str, List[int]] = defaultdict(list)
    video_probs: Dict[str, List[float]] = defaultdict(list)

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite validation loss: {loss.item()}"
            )

        losses.append(float(loss.item()))
        batch_labels = labels.cpu().tolist()
        batch_probs = logits.softmax(dim=1)[:, 1].cpu().tolist()
        labels_all.extend(batch_labels)
        probs_all.extend(batch_probs)
        for video, label, probability in zip(
            batch["video"], batch_labels, batch_probs
        ):
            video_labels[video].append(label)
            video_probs[video].append(probability)

    mean_loss = sum(losses) / max(1, len(losses))
    metrics = binary_metrics(labels_all, probs_all)
    video_y: List[int] = []
    video_p: List[float] = []
    for video in sorted(video_probs):
        labels = video_labels[video]
        if len(set(labels)) != 1:
            raise RuntimeError(f"Inconsistent labels for video {video}: {labels}")
        video_y.append(labels[0])
        video_p.append(sum(video_probs[video]) / len(video_probs[video]))
    metrics.update(
        {f"video_{key}": value for key, value in binary_metrics(video_y, video_p).items()}
    )
    return mean_loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage1_maxvit.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))

    output_dir = Path(cfg["output"]["directory"])
    logger = configure_logging(output_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda" and bool(cfg["train"].get("require_cuda", True)):
        raise RuntimeError(
            "Stage 1 MaxViT training requires a CUDA-enabled PyTorch build. "
            "Set train.require_cuda=false only for smoke tests."
        )

    requested_amp = bool(cfg["train"].get("amp", False))
    if requested_amp:
        logger.warning(
            "AMP was requested but is disabled because this custom SAM "
            "implementation is not GradScaler-safe."
        )
    amp_enabled = False

    logger.info("Run started at %s", datetime.now().isoformat())
    logger.info("Config: %s", json.dumps(cfg, ensure_ascii=False))
    logger.info("Device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    train_loader = make_train_loader(cfg)
    val_loader = make_val_loader(cfg)

    logger.info(
        "Train source samples: %d",
        len(train_loader.dataset),
    )
    logger.info(
        "Val FF++ real/fake frames: %d",
        len(val_loader.dataset),
    )
    logger.info(
        "Effective train image batch: %d",
        int(cfg["train"]["source_batch_size"]) * 2,
    )

    if int(cfg["data"]["model_resolution"]) != MaxViTStage1.INPUT_SIZE:
        raise ValueError("MaxViT-Tiny 384 requires data.model_resolution=384")
    model = MaxViTStage1(
        num_classes=int(cfg["model"]["num_classes"]),
        pretrained=cfg["model"].get("pretrained", "imagenet1k_v1"),
        checkpoint=cfg["model"].get("checkpoint"),
    ).to(device)

    optimizer = SAM(
        model.parameters(),
        torch.optim.SGD,
        rho=float(cfg["train"]["sam_rho"]),
        lr=float(cfg["train"]["lr"]),
        momentum=float(cfg["train"]["momentum"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    if str(cfg["train"].get("scheduler", "warmup_cosine")) != "warmup_cosine":
        raise ValueError("Only train.scheduler=warmup_cosine is supported")
    scheduler = WarmupCosineLR(
        optimizer,
        n_epoch=epochs,
        warmup_epochs=int(cfg["train"].get("warmup_epochs", 1)),
        warmup_start_lr=float(cfg["train"].get("warmup_start_lr", 1e-4)),
        min_lr=float(cfg["train"].get("min_lr", 1e-5)),
    )
    criterion = nn.CrossEntropyLoss()

    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.jsonl"
    top_k = int(cfg["train"]["save_top_k"])
    saved: List[Tuple[float, Path]] = []

    for epoch in range(1, epochs + 1):
        model.train()

        lr_used = float(optimizer.param_groups[0]["lr"])

        train_losses: List[float] = []
        labels_all: List[int] = []
        probs_all: List[float] = []

        progress = tqdm(
            train_loader,
            desc=f"train {epoch}/{epochs} lr={lr_used:.8g}",
        )

        for batch in progress:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )
            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            loss_value, logits = sam_step(
                model,
                optimizer,
                criterion,
                images,
                labels,
                amp_enabled,
            )

            train_losses.append(loss_value)
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(
                logits.softmax(dim=1)[:, 1].cpu().tolist()
            )

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                lr=f"{lr_used:.3e}",
            )

        train_metrics = binary_metrics(
            labels_all,
            probs_all,
        )
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled,
        )

        scheduler.step()
        lr_next = float(optimizer.param_groups[0]["lr"])

        train_loss = sum(train_losses) / max(
            1,
            len(train_losses),
        )

        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "lr_used": lr_used,
            "lr_next": lr_next,
            "train_loss": train_loss,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            "val_loss": val_loss,
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
            "dataset_retry_count_train": int(
                train_loader.dataset.retry_count
            ),
            "dataset_retry_count_val": int(
                val_loader.dataset.retry_count
            ),
        }

        record_json = json.dumps(
            record,
            ensure_ascii=False,
        )
        logger.info("Epoch summary: %s", record_json)

        with history_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(record_json + "\n")

        metric_name = str(cfg.get("eval", {}).get("metric", "auc"))
        if metric_name not in val_metrics:
            raise KeyError(
                f"Unknown eval.metric={metric_name!r}; available={sorted(val_metrics)}"
            )
        score = float(val_metrics[metric_name])
        checkpoint_path = (
            weights_dir
            / f"epoch_{epoch:02d}_{metric_name}_{score:.4f}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
                "lr_used": lr_used,
                "lr_next": lr_next,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        saved.append((score, checkpoint_path))
        saved.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        while len(saved) > top_k:
            _, old_checkpoint = saved.pop()
            old_checkpoint.unlink(missing_ok=True)

    final_path = output_dir / "final.pth"
    torch.save(
        {
            "epoch": epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "lr": float(optimizer.param_groups[0]["lr"]),
        },
        final_path,
    )

    logger.info("Training completed. Final checkpoint: %s", final_path)


if __name__ == "__main__":
    main()
