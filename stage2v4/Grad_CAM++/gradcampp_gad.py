"-image ... --class-index 1 -> heatmap fake, --class-index 0 -> heatmap real "


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_models.convnextv2_gad import ConvNeXtV2GAD
from hybrid_models.stage2_hybrid import Projection


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("real", "fake")


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GADStage2Classifier(nn.Module):
    def __init__(self, gad_checkpoint: Path, stage2_checkpoint: Path) -> None:
        super().__init__()
        checkpoint = safe_load(stage2_checkpoint)
        state = checkpoint.get("model", checkpoint)
        if not isinstance(state, dict):
            raise TypeError("Stage-2 checkpoint does not contain a state dict")

        projection_weight = state.get("gad_projection.net.0.weight")
        classifier_weight = state.get("gad_aux_classifier.weight")
        if projection_weight is None or classifier_weight is None:
            raise KeyError(
                "Stage-2 checkpoint is missing GAD projection or auxiliary classifier"
            )

        projection_dim, gad_dim = projection_weight.shape
        num_classes = classifier_weight.shape[0]
        self.gad = ConvNeXtV2GAD(
            checkpoint_path=str(gad_checkpoint),
            drop_path_rate=0.0,
        )
        if self.gad.feature_dim != gad_dim:
            raise RuntimeError(
                f"GAD feature dim mismatch: model={self.gad.feature_dim}, "
                f"checkpoint={gad_dim}"
            )
        self.gad_projection = Projection(gad_dim, projection_dim, dropout=0.0)
        self.gad_aux_classifier = nn.Linear(projection_dim, num_classes)

        branch_state = {
            key: value
            for key, value in state.items()
            if key.startswith(("gad.", "gad_projection.", "gad_aux_classifier."))
        }
        self.load_state_dict(branch_state, strict=True)
        self.target_layer = self.gad.encoder.stages[-1][-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.gad(images)
        projected = self.gad_projection(features)
        return self.gad_aux_classifier(projected)


class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError(
                "Grad-CAM++ target layer must return a [N, C, H, W] tensor"
            )
        self.activations = output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient):
        self.gradients = gradient

    def __call__(self, input_tensor: torch.Tensor, class_index: int):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        logits[:, class_index].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients")

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        gradients_2 = gradients.pow(2)
        gradients_3 = gradients_2 * gradients
        activation_sum = activations.sum(dim=(2, 3), keepdim=True)
        denominator = 2.0 * gradients_2 + activation_sum * gradients_3
        denominator = torch.where(
            denominator.abs() > 1e-8,
            denominator,
            torch.ones_like(denominator),
        )
        alpha = gradients_2 / denominator

        # Grad-CAM++ reference implementation normalizes alpha separately
        # for every channel after suppressing negative first-order gradients.
        positive_gradients = torch.relu(gradients)
        alpha_thresholded = torch.where(
            positive_gradients > 0,
            alpha,
            torch.zeros_like(alpha),
        )
        alpha_normalizer = alpha_thresholded.sum(dim=(2, 3), keepdim=True)
        alpha_normalizer = torch.where(
            alpha_normalizer.abs() > 1e-8,
            alpha_normalizer,
            torch.ones_like(alpha_normalizer),
        )
        alpha = alpha / alpha_normalizer
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return logits.detach(), cam.cpu().numpy()

    def close(self) -> None:
        self.handle.remove()


def prepare_image(path: Path, image_size: int):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tensor = transform(image)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    display = (tensor.cpu() * std + mean).clamp(0, 1)
    display = display.permute(1, 2, 0).numpy()
    return tensor.unsqueeze(0), display


def save_visualization(path: Path, rgb: np.ndarray, heatmap: np.ndarray):
    heatmap_u8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(0.55 * rgb + 0.45 * colored, 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    original_path = path.with_name(f"{path.stem}_original{suffix}")
    heatmap_path = path.with_name(f"{path.stem}_heatmap{suffix}")
    overlay_path = path.with_name(f"{path.stem}_overlay{suffix}")
    Image.fromarray(np.uint8(rgb * 255)).save(original_path)
    Image.fromarray(np.uint8(colored * 255)).save(heatmap_path)
    Image.fromarray(np.uint8(overlay * 255)).save(overlay_path)
    return original_path, heatmap_path, overlay_path


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM++ heatmap for the Stage-2 GAD branch"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--stage2-checkpoint",
        default="checkpoints/stage2/epoch_19_frame_auc_0.9834.pth",
    )
    parser.add_argument(
        "--gad-checkpoint",
        default="checkpoints/pretrained/convnextv2_tiny_22k_384_ema.pt",
    )
    parser.add_argument("--class-index", type=int, choices=(0, 1), default=None)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    image_path = resolve_from_root(args.image)
    stage2_checkpoint = resolve_from_root(args.stage2_checkpoint)
    gad_checkpoint = resolve_from_root(args.gad_checkpoint)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    model = GADStage2Classifier(gad_checkpoint, stage2_checkpoint).to(device).eval()
    input_tensor, display = prepare_image(image_path, args.image_size)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        preview_logits = model(input_tensor)
        probabilities = preview_logits.softmax(dim=1)[0]
    class_index = (
        int(probabilities.argmax().item())
        if args.class_index is None
        else args.class_index
    )

    cam = GradCAMPlusPlus(model, model.target_layer)
    try:
        logits, heatmap = cam(input_tensor, class_index)
    finally:
        cam.close()
    probabilities = logits.softmax(dim=1)[0].cpu().tolist()

    output = (
        resolve_from_root(args.output)
        if args.output
        else Path(__file__).resolve().parent
        / "outputs"
        / f"{image_path.stem}_gad_stage2_gradcampp.png"
    )
    original_path, heatmap_path, overlay_path = save_visualization(
        output, display, heatmap
    )
    print(f"Device: {device}")
    print(f"Target class: {class_index} ({CLASS_NAMES[class_index]})")
    print(f"Probabilities: real={probabilities[0]:.6f}, fake={probabilities[1]:.6f}")
    print(f"Original: {original_path}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
