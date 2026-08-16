from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage2_data import DeepfakeBenchDataset, build_branch_transforms
from stage2_utils.balanced_source_sampler import ExactSourceBalancedSampler
from stage2_utils import (
    build_stage2, compute_metrics, load_yaml, resolve_path,
    seed_everything, worker_init_fn,
)


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage2")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(
        output_dir / "train.log", mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def safe_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


CHECKPOINT_PATTERN = re.compile(
    r"^epoch_(?P<epoch>\d+)_frame_auc_(?P<frame_auc>[-+]?\d+(?:\.\d+)?|nan)\.pth$",
    re.IGNORECASE,
)


def keep_top_checkpoints(output_dir: Path, limit: int = 6):
    ranked = []
    for path in output_dir.glob("epoch_*_frame_auc_*.pth"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            auc = float(match.group("frame_auc"))
            ranked.append((auc if math.isfinite(auc) else float("-inf"),
                           int(match.group("epoch")), path))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    removed = []
    for _, _, path in ranked[limit:]:
        path.unlink()
        removed.append(path)
    return removed


def absolutize(cfg, base: Path):
    sections = {
        "data": ["train_json", "val_json", "dataset_root"],
        "bad": ["checkpoint"],
        "gad": ["checkpoint"],
        "output": ["directory"],
    }
    for section, keys in sections.items():
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value:
                cfg[section][key] = resolve_path(value, base)
    return cfg


def make_scheduler(optimizer, epochs, warmup_epochs, min_lr_ratio):
    

    def factor(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = min(max(epoch - warmup_epochs, 0), epochs - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def get_learning_rates(optimizer):
    """Return the learning rate currently used by each optimizer group."""
    group_names = ("gad", "head")
    return {
        group_names[index] if index < len(group_names) else f"group_{index}":
        float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def run_epoch(
    model, loader, criterion, device, optimizer=None, scaler=None,
    accumulation_steps=1, gradient_clip_norm=0.0, phase="train",
    epoch=None, total_epochs=None, aux_weight=0.0,
):
    training = optimizer is not None
    model.train(training)
    labels_all, probs_all, video_ids_all = [], [], []
    loss_sum = 0.0
    if training:
        optimizer.zero_grad(set_to_none=True)

    desc = phase if epoch is None else f"{phase} {epoch}/{total_epochs}"
    if training:
        lr_text = ",".join(
            f"{name}={lr:.3e}"
            for name, lr in get_learning_rates(optimizer).items()
        )
        desc = f"{desc} lr[{lr_text}]"
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for step, batch in enumerate(progress, 1):
        labels = batch["label"].to(device, non_blocking=True)
        amp_enabled = scaler is not None and scaler.is_enabled()
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                fusion_logits, bad_aux, sfe_aux, gad_aux = model(
                    batch["bad"].to(device, non_blocking=True),
                    batch["sfe"].to(device, non_blocking=True),
                    batch["gad"].to(device, non_blocking=True),
                    return_aux=True,
                )
                aux_loss = (
                    criterion(bad_aux, labels)
                    + criterion(sfe_aux, labels)
                    + criterion(gad_aux, labels)
                )
                loss = criterion(fusion_logits, labels) + aux_weight * aux_loss
                # Ensemble mean cho metrics — mỗi nhánh 1 phiếu ngang nhau.
                ensemble_logits = (
                    fusion_logits + bad_aux + sfe_aux + gad_aux
                ) / 4.0
            if training:
                scaler.scale(loss / accumulation_steps).backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    if gradient_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        loss_sum += float(loss.item()) * labels.size(0)
        labels_all.extend(labels.detach().cpu().tolist())
        probs_all.extend(ensemble_logits.softmax(1)[:, 1].detach().cpu().tolist())
        video_ids_all.extend(batch["video_id"])
        postfix = {"loss": f"{loss.item():.4f}"}
        if training:
            postfix.update({
                f"lr_{name}": f"{lr:.3e}"
                for name, lr in get_learning_rates(optimizer).items()
            })
        progress.set_postfix(postfix)

    metrics = compute_metrics(labels_all, probs_all, video_ids_all)
    metrics["loss"] = loss_sum / max(1, len(labels_all))
    return metrics


def main(argv=None, require_resume=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=require_resume, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gad-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--reset-scheduler", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    cfg = absolutize(load_yaml(str(config_path)), project_root)
    seed = int(cfg["seed"])
    seed_everything(seed)
    output_dir = Path(cfg["output"]["directory"])
    logger = configure_logging(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    model = build_stage2(cfg).to(device)
    frozen = list(model.bad.parameters()) + list(model.sfe.parameters())
    if any(p.requires_grad for p in frozen):
        raise RuntimeError("BAD and SFE must be frozen in stage 2.")

    train_tf = build_branch_transforms(
        model.sfe.processor, int(cfg["data"]["image_size"]), True,
        bad_encoder=model.bad.encoder,
    )
    val_tf = build_branch_transforms(
        model.sfe.processor, int(cfg["data"]["image_size"]), False,
        bad_encoder=model.bad.encoder,
    )
    common = dict(
        dataset_root=cfg["data"].get("dataset_root", ""),
        compression=cfg["data"].get("compression"),
        methods=cfg["data"].get("methods", []),
    )
    train_ds = DeepfakeBenchDataset(
        json_path=cfg["data"]["train_json"], transforms=train_tf,
        split=cfg["data"]["train_split"],
        max_frames_per_video=cfg["data"].get("max_frames_per_video_train"),
        **common,
    )
    val_ds = DeepfakeBenchDataset(
        json_path=cfg["data"]["val_json"], transforms=val_tf,
        split=cfg["data"]["val_split"],
        max_frames_per_video=cfg["data"].get("max_frames_per_video_val"),
        **common,
    )

    balance = cfg["data"].get("balanced_sampling", {})
    fake_sources = balance.get("fake_sources", ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"])
    train_sampler = None
    if balance.get("train_enabled", balance.get("enabled", True)):
        train_sampler = ExactSourceBalancedSampler(
            train_ds.records, balance.get("real_source", "FF-real"), fake_sources,
            int(balance["real_train"]), int(balance["fake_train"]), seed, True,
        )
    val_sampler = None
    if balance.get("val_enabled", balance.get("enabled", False)):
        val_sampler = ExactSourceBalancedSampler(
            val_ds.records, balance.get("real_source", "FF-real"), fake_sources,
            int(balance["real_val"]), int(balance["fake_val"]), seed, False,
        )

    generator = torch.Generator().manual_seed(seed)
    loader_args = dict(
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    train_loader = DataLoader(
        train_ds, shuffle=train_sampler is None, sampler=train_sampler, **loader_args
    )
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **loader_args)
    train_frames = len(train_sampler) if train_sampler is not None else len(train_ds)
    val_frames = len(val_sampler) if val_sampler is not None else len(val_ds)
    logger.info(
        "Frames per epoch: train=%s, val=%s | batches: train=%s, val=%s",
        f"{train_frames:,}", f"{val_frames:,}",
        f"{len(train_loader):,}", f"{len(val_loader):,}",
    )

    heads = (
        list(model.bad_projection.parameters()) +
        list(model.sfe_projection.parameters()) +
        list(model.gad_projection.parameters()) +
        list(model.bad_aux_classifier.parameters()) +
        list(model.sfe_aux_classifier.parameters()) +
        list(model.gad_aux_classifier.parameters()) +
        list(model.classifier.parameters())
    )
    gad_lr = args.gad_lr if args.gad_lr is not None else float(cfg["train"]["gad_lr"])
    head_lr = args.head_lr if args.head_lr is not None else float(cfg["train"]["head_lr"])
    optimizer = torch.optim.AdamW(
        [{"params": model.gad.parameters(), "lr": gad_lr},
         {"params": heads, "lr": head_lr}],
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = args.epochs if args.epochs is not None else int(cfg["train"]["epochs"])
    scheduler = make_scheduler(
        optimizer, epochs, int(cfg["train"]["warmup_epochs"]),
        float(cfg["train"]["min_lr_ratio"]),
    )
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    best_frame_auc = float("-inf")

    if args.resume:
        checkpoint = safe_load(resolve_path(args.resume, project_root))
        model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_frame_auc = float(
            checkpoint.get(
                "best_frame_auc",
                checkpoint.get("val_metrics", {}).get("frame_auc", float("-inf")),
            )
        )
        if args.gad_lr is not None:
            optimizer.param_groups[0]["lr"] = args.gad_lr
        if args.head_lr is not None:
            optimizer.param_groups[1]["lr"] = args.head_lr
        overridden_base_lrs = None
        if args.gad_lr is not None or args.head_lr is not None:
            overridden_base_lrs = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            for group, base_lr in zip(optimizer.param_groups, overridden_base_lrs):
                group["initial_lr"] = base_lr
        if args.reset_scheduler:
            scheduler = make_scheduler(
                optimizer,
                max(1, epochs - start_epoch + 1),
                int(cfg["train"]["warmup_epochs"]),
                float(cfg["train"]["min_lr_ratio"]),
            )
        elif "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if overridden_base_lrs is not None:
            scheduler.base_lrs = overridden_base_lrs
        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    class_counts = torch.bincount(torch.tensor([r["label"] for r in train_ds.records]), minlength=2).float()
    class_weights = (class_counts.sum() / (2 * class_counts.clamp_min(1))).to(device)
    logger.info("Full-data class counts=%s, CE weights=%s", class_counts.tolist(), class_weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(start_epoch, epochs + 1):
        lr_used = get_learning_rates(optimizer)
        formatted_lrs = ", ".join(
            f"{name}={lr:.8g}" for name, lr in lr_used.items()
        )
        logger.info(
            "Epoch %d/%d | frames: train=%s, val=%s | lr used: %s",
            epoch, epochs, f"{train_frames:,}", f"{val_frames:,}",
            formatted_lrs,
        )
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
            int(cfg["train"]["accumulation_steps"]),
            float(cfg["train"]["gradient_clip_norm"]), "train", epoch, epochs,
            float(cfg["train"].get("aux_weight", 0.0)),
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device,
            phase="val", epoch=epoch, total_epochs=epochs,
            aux_weight=float(cfg["train"].get("aux_weight", 0.0)),
        )
        scheduler.step()
        lr_next = get_learning_rates(optimizer)
        auc = float(val_metrics["frame_auc"])
        if math.isfinite(auc) and auc > best_frame_auc:
            best_frame_auc = auc
        state = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "config": cfg,
            "train_metrics": train_metrics, "val_metrics": val_metrics,
            "best_frame_auc": best_frame_auc,
            "lr_used": lr_used, "lr_next": lr_next,
            "train_frames": train_frames, "val_frames": val_frames,
        }
        path = output_dir / f"epoch_{epoch:02d}_frame_auc_{auc:.4f}.pth"
        torch.save(state, path)
        removed = keep_top_checkpoints(
            output_dir, int(cfg["output"].get("keep_checkpoints", 6))
        )
        summary = {
            "epoch": epoch,
            "frames": {"train": train_frames, "val": val_frames},
            "lr_used": lr_used,
            "lr_next": lr_next,
            "train": train_metrics,
            "val": val_metrics,
        }
        logger.info(
            "Epoch summary: %s",
            json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
        )
        for old in removed:
            logger.info("Removed checkpoint: %s", old)


if __name__ == "__main__":
    main()
