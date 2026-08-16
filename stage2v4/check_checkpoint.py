"""Kiểm tra cân bằng 3 nhánh trên một checkpoint (hoặc mọi checkpoint trong thư mục).

Cách dùng:
    python analyze_branch_balance.py --config configs/stage2.yaml --checkpoint outputs/stage2_h2_full_data/epoch_05_frame_auc_0.91.pth
    python analyze_branch_balance.py --config configs/stage2.yaml --checkpoint outputs/stage2_h2_full_data
    frame_auc	AUC riêng của từng phiếu	Nhánh nào cao → giỏi trên val
    conf	Độ tự tin trung bình (max softmax)	Cao + sai nhiều = "tự tin mù"
    |logit|	Độ "lớn tiếng" trung bình	Nhánh nào lớn → nói to trong tổng
    agree	Tỷ lệ phiếu này trùng quyết định với ensemble	Cao = phiếu này chi phối kết quả cuối
    flip	Nếu bỏ phiếu này, bao nhiêu % mẫu đổi quyết định	Cao = phiếu này đang nắm vai trò quyết định (dominate)
    all_agree	3 nhánh cùng chốt 1 kết luận	Thấp = 3 nhánh đang bổ sung nhau


"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage2_data import DeepfakeBenchDataset, build_branch_transforms
from stage2_utils import (
    build_stage2, compute_metrics, load_yaml, resolve_path,
    seed_everything, worker_init_fn,
)

VOTERS = ["bad", "sfe", "gad", "fusion", "ensemble"]
BRANCHES = ["bad", "sfe", "gad"]

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def absolutize(cfg, base):
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

@torch.inference_mode()
def collect(model, loader, device):
    """Chạy val loader, lưu logits của 4 phiếu + label + video_id."""
    rec = {k: [] for k in VOTERS if k != "ensemble"}
    rec["label"], rec["video"] = [], []
    model.eval()
    for batch in tqdm(loader, desc="collect", leave=False):
        fusion, bad_aux, sfe_aux, gad_aux = model(
            batch["bad"].to(device, non_blocking=True),
            batch["sfe"].to(device, non_blocking=True),
            batch["gad"].to(device, non_blocking=True),
            return_aux=True,
        )
        rec["fusion"].append(fusion)
        rec["bad"].append(bad_aux)
        rec["sfe"].append(sfe_aux)
        rec["gad"].append(gad_aux)
        rec["label"].append(batch["label"])
        rec["video"].extend(batch["video_id"])
    for k in ("fusion", "bad", "sfe", "gad"):
        rec[k] = torch.cat(rec[k], dim=0)
    rec["label"] = torch.cat(rec["label"], dim=0)
    rec["ensemble"] = sum(rec[k] for k in ("fusion", "bad", "sfe", "gad")) / 4.0
    return rec

def analyze(rec):
    labels = rec["label"].cpu().numpy()
    ens_pred = rec["ensemble"].argmax(1)

    print(f"\n{'voter':<10}{'frame_auc':>10}{'acc':>8}{'conf':>8}{'|logit|':>9}{'agree':>8}{'flip':>8}")
    result = {}
    for v in VOTERS:
        logits = rec[v]
        probs = logits.softmax(1)[:, 1].cpu().numpy()
        pred = probs >= 0.5
        auc = compute_metrics(labels.tolist(), probs.tolist(), rec["video"])["frame_auc"]
        acc = float((pred == labels).mean())
        conf = float(logits.softmax(1).max(1).values.mean())
        mag = float(logits[:, 1].abs().mean())
        agree = float((logits.argmax(1) == ens_pred).float().mean()) if v != "ensemble" else float("nan")
        # Bỏ phiếu v, 3 phiếu còn lại có đổi quyết định không
        if v in BRANCHES:
            others = (rec["ensemble"] * 4.0 - rec[v]) / 3.0
            flip = float((others.argmax(1) != ens_pred).float().mean())
        else:
            flip = float("nan")
        result[v] = dict(auc=auc, acc=acc, conf=conf, mag=mag, agree=agree, flip=flip)
        print(f"{v:<10}{auc:>10.4f}{acc:>8.4f}{conf:>8.4f}{mag:>9.4f}{agree:>8.4f}{flip:>8.4f}")

    # Bất đồng giữa 3 nhánh (trước khi fusion): 3 nhánh có cùng dự đoán không
    preds = torch.stack([rec[v].argmax(1) for v in BRANCHES], dim=0)
    all_agree = float((preds == preds[0]).all(0).float().mean())
    print(f"\nTỷ lệ 3 nhánh cùng chốt 1 kết luận: {all_agree:.4f} "
          f"({1 - all_agree:.4f} mẫu có nhánh bất đồng)")
    return result, all_agree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="File .pth hoặc thư mục chứa epoch_*.pth")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = absolutize(load_yaml(str(config_path)), config_path.parent.parent)
    seed_everything(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_stage2(cfg).to(device)
    transforms = build_branch_transforms(
        model.sfe.processor, int(cfg["data"]["image_size"]), False,
        bad_encoder=model.bad.encoder,
    )
    dataset = DeepfakeBenchDataset(
        json_path=cfg["data"]["val_json"], transforms=transforms,
        split=cfg["data"]["val_split"],
        max_frames_per_video=cfg["data"].get("max_frames_per_video_val"),
        dataset_root=cfg["data"].get("dataset_root", ""),
        compression=cfg["data"].get("compression"),
        methods=cfg["data"].get("methods", []),
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False, num_workers=int(cfg["train"]["workers"]),
        pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn,
    )

    checkpoint_path = Path(args.checkpoint).resolve()
    if checkpoint_path.is_dir():
        files = sorted(checkpoint_path.glob("epoch_*_frame_auc_*.pth"))
    else:
        files = [checkpoint_path]
    if not files:
        raise SystemExit(f"Không tìm thấy checkpoint trong {checkpoint_path}")

    for ckpt in files:
        print(f"\n{'='*70}\nCheckpoint: {ckpt.name}")
        state = safe_load(str(ckpt))
        model.load_state_dict(state["model"], strict=True)
        rec = collect(model, loader, device)
        analyze(rec)

if __name__ == "__main__":
    main()