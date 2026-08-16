"""Inspect one image's Stage-2 score and branch contributions.
python test/inspect_image_score.py --image pat/to/image.png --output test/image_score.json """
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage2_data import build_branch_transforms  # noqa: E402
from stage2_utils import build_stage2, resolve_path, seed_everything  # noqa: E402


def safe_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def resolve_input(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_config(config_arg: str) -> tuple[Path, dict]:
    config_path = Path(config_arg).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return config_path, yaml.safe_load(handle)


def build_model_config(config: dict, project_root: Path) -> dict:
    architecture = config["architecture"]
    architecture["data"].setdefault("image_size", 384)
    architecture["fusion"].setdefault("projection_dim", 512)
    architecture["fusion"].setdefault("dropout", 0.3)
    architecture["fusion"].setdefault("bad_branch_dropout", 0.0)
    architecture["fusion"].setdefault("sfe_branch_dropout", 0.0)
    architecture["fusion"].setdefault("num_classes", 2)
    for section in ("bad", "gad"):
        architecture[section]["checkpoint"] = resolve_path(
            architecture[section]["checkpoint"], project_root
        )
    return architecture


def load_stage2_model(config: dict, project_root: Path, checkpoint_arg: str | None):
    model_config = build_model_config(config, project_root)
    model = build_stage2(model_config)
    checkpoint_value = checkpoint_arg or config["checkpoint"]
    checkpoint_path = Path(resolve_path(checkpoint_value, project_root))
    checkpoint = safe_load(checkpoint_path)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, model_config, checkpoint_path


def probs(logits: torch.Tensor) -> dict[str, float]:
    values = logits.softmax(dim=1)[0].float().cpu()
    return {"real": float(values[0]), "fake": float(values[1])}


def margin(logits: torch.Tensor) -> float:
    return float((logits[0, 1] - logits[0, 0]).item())


def project_features(model, bad, sfe, gad):
    with torch.no_grad():
        bad_features = model.bad(bad)
        sfe_features = model.sfe(sfe)
        gad_features = model.gad(gad)
        return {
            "bad": model.bad_projection(bad_features),
            "sfe": model.sfe_projection(sfe_features),
            "gad": model.gad_projection(gad_features),
        }


def fusion_from_parts(model, parts, disabled: str | None = None):
    values = [
        torch.zeros_like(parts[name]) if name == disabled else parts[name]
        for name in ("bad", "sfe", "gad")
    ]
    return model.classifier(torch.cat(values, dim=1))


def inspect_image(model, transforms, image_path: Path, device: torch.device):
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    branch = transforms(image)
    bad = branch["bad"].unsqueeze(0).to(device)
    sfe = branch["sfe"].unsqueeze(0).to(device)
    gad = branch["gad"].unsqueeze(0).to(device)

    with torch.no_grad():
        fusion, bad_aux, sfe_aux, gad_aux = model(
            bad, sfe, gad, return_aux=True
        )
        branch_logits = {
            "bad": bad_aux,
            "sfe": sfe_aux,
            "gad": gad_aux,
            "fusion": fusion,
        }
        ensemble = (fusion + bad_aux + sfe_aux + gad_aux) / 4.0
        parts = project_features(model, bad, sfe, gad)
        full_fusion = fusion_from_parts(model, parts)

        ablation = {}
        for name in ("bad", "sfe", "gad"):
            without = fusion_from_parts(model, parts, disabled=name)
            ablation[name] = {
                "without_probability": probs(without),
                "fake_probability_delta": probs(full_fusion)["fake"] - probs(without)["fake"],
                "fake_margin_delta": margin(full_fusion) - margin(without),
            }
        total_influence = sum(
            abs(values["fake_margin_delta"]) for values in ablation.values()
        )
        for values in ablation.values():
            values["absolute_influence_share"] = (
                abs(values["fake_margin_delta"]) / total_influence
                if total_influence > 0
                else 0.0
            )

    branch_report = {}
    for name, logits in branch_logits.items():
        branch_report[name] = {
            "probability": probs(logits),
            "fake_margin": margin(logits),
            "ensemble_margin_contribution": margin(logits) / 4.0,
        }

    ensemble_probability = probs(ensemble)
    return {
        "image": str(image_path),
        "predicted_class": "fake" if ensemble_probability["fake"] >= 0.5 else "real",
        "ensemble_probability": ensemble_probability,
        "ensemble_fake_margin": margin(ensemble),
        "branches": branch_report,
        "fusion_ablation": ablation,
        "ablation_interpretation": (
            "Positive fake_probability_delta/fake_margin_delta means the branch "
            "pushes the fusion result toward fake; negative means toward real. "
            "absolute_influence_share is normalized from absolute margin deltas."
        ),
        "projected_feature_norm": {
            name: float(value.float().norm().item()) for name, value in parts.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Stage-2 score and BAD/SFE/GAD contributions for one image."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/cross_test.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    image_path = resolve_input(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    config_path, config = load_config(args.config)
    seed_everything(int(config.get("seed", 5)))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model, model_config, checkpoint_path = load_stage2_model(
        config, PROJECT_ROOT, args.checkpoint
    )
    model = model.to(device)
    transforms = build_branch_transforms(
        model.sfe.processor,
        int(model_config["data"]["image_size"]),
        training=False,
        bad_encoder=model.bad.encoder,
    )
    report = inspect_image(model, transforms, image_path, device)
    report.update(
        {
            "device": str(device),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
        }
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        output_path = resolve_input(args.output)
        if output_path.suffix.lower() != ".json":
            output_path = output_path / f"{image_path.stem}_score.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
