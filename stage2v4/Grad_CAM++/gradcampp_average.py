from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


GRAD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GRAD_DIR.parent
if str(GRAD_DIR) not in sys.path:
    sys.path.insert(0, str(GRAD_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gradcampp_bad import build_bad, prepare_image as prepare_bad_image  # noqa: E402
from gradcampp_gad import (  # noqa: E402
    GADStage2Classifier,
    GradCAMPlusPlus,
    prepare_image as prepare_gad_image,
    resolve_from_root,
    save_visualization,
)


CLASS_NAMES = ("real", "fake")
CHECKPOINT_PATTERN = re.compile(r"frame_auc_([0-9]+(?:\.[0-9]+)?)\.pth$")


def resolve_stage2_checkpoint(value: str | None) -> Path:
    if value:
        path = resolve_from_root(value)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {path}")

    candidates = []
    for directory in (PROJECT_ROOT / "checkpoints" / "stage2", PROJECT_ROOT / "outputs"):
        if not directory.is_dir():
            continue
        for path in directory.glob("epoch_*_frame_auc_*.pth"):
            match = CHECKPOINT_PATTERN.search(path.name)
            if match:
                candidates.append((float(match.group(1)), path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(
            "No Stage-2 checkpoint found in checkpoints/stage2 or outputs"
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average BAD and GAD Grad-CAM++ heatmaps for one image"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--stage2-checkpoint",
        default=None,
        help="Optional Stage-2 checkpoint. By default, select the highest AUC checkpoint.",
    )
    parser.add_argument(
        "--gad-checkpoint",
        default="checkpoints/pretrained/convnextv2_tiny_22k_384_ema.pt",
    )
    parser.add_argument(
        "--bad-checkpoint",
        default="checkpoints/bad/epoch_18_auc_0.8776.pth",
    )
    parser.add_argument("--model-name", default="maxvit_tiny_tf_384.in1k")
    parser.add_argument("--class-index", type=int, choices=(0, 1), default=None)
    parser.add_argument("--combine", choices=("max", "average"), default="max")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    stage2_checkpoint = resolve_stage2_checkpoint(args.stage2_checkpoint)

    image_path = resolve_from_root(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    gad = GADStage2Classifier(
        resolve_from_root(args.gad_checkpoint),
        stage2_checkpoint,
    ).to(device).eval()
    bad = build_bad(
        args.model_name,
        resolve_from_root(args.bad_checkpoint),
    ).to(device).eval()

    gad_input, display = prepare_gad_image(image_path, 384)
    bad_input, _ = prepare_bad_image(image_path, bad)
    gad_input = gad_input.to(device)
    bad_input = bad_input.to(device)

    with torch.no_grad():
        gad_logits = gad(gad_input)
        bad_logits = bad(bad_input)
        average_logits = (gad_logits + bad_logits) / 2.0
        probabilities = average_logits.softmax(dim=1)[0]
    class_index = (
        int(probabilities.argmax().item())
        if args.class_index is None
        else args.class_index
    )

    gad_cam = GradCAMPlusPlus(gad, gad.target_layer)
    bad_cam = GradCAMPlusPlus(bad, bad.target_layer)
    try:
        gad_scores, gad_heatmap = gad_cam(gad_input, class_index)
        bad_scores, bad_heatmap = bad_cam(bad_input, class_index)
    finally:
        gad_cam.close()
        bad_cam.close()

    if gad_heatmap.shape != bad_heatmap.shape:
        raise RuntimeError(
            f"BAD/GAD heatmap shapes do not match: "
            f"BAD={bad_heatmap.shape}, GAD={gad_heatmap.shape}"
        )
    # Blend the two already-normalized CAMs, then normalize the blended map
    # once more.  Without this second normalization the average has a maximum
    # below 1 whenever BAD and GAD peak at different pixels, making the result
    # look washed out compared with either component.
    bad_map = bad_heatmap.astype(np.float32)
    gad_map = gad_heatmap.astype(np.float32)
    if args.combine == "max":
        combined_heatmap = np.maximum(bad_map, gad_map)
    else:
        combined_heatmap = (bad_map + gad_map) / 2.0
    combined_heatmap -= combined_heatmap.min()
    combined_heatmap /= max(float(combined_heatmap.max()), 1e-8)
    combined_heatmap = np.clip(combined_heatmap, 0.0, 1.0)

    output = (
        resolve_from_root(args.output)
        if args.output
        else GRAD_DIR / "outputs" / f"{image_path.stem}_average_gradcampp.png"
    )
    original_path, heatmap_path, overlay_path = save_visualization(
        output, display, combined_heatmap
    )

    gad_probabilities = gad_scores.softmax(dim=1)[0].cpu().tolist()
    bad_probabilities = bad_scores.softmax(dim=1)[0].cpu().tolist()
    print(f"Device: {device}")
    print(f"Stage-2 checkpoint: {stage2_checkpoint}")
    print(f"Target class: {class_index} ({CLASS_NAMES[class_index]})")
    print(
        f"BAD probabilities: real={bad_probabilities[0]:.6f}, "
        f"fake={bad_probabilities[1]:.6f}"
    )
    print(
        f"GAD probabilities: real={gad_probabilities[0]:.6f}, "
        f"fake={gad_probabilities[1]:.6f}"
    )
    print(
        f"Average-logit probabilities: real={probabilities[0].item():.6f}, "
        f"fake={probabilities[1].item():.6f}"
    )
    print(f"Original: {original_path}")
    print(f"Combined mode: {args.combine}")
    print(f"Combined heatmap: {heatmap_path}")
    print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
