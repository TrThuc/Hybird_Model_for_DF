from __future__ import annotations

from hybrid_models import ConvNeXtV2GAD, FrozenBAD, FrozenSigLIP2, Stage2Hybrid


def build_stage2(cfg):
    bad = FrozenBAD(
        model_name=cfg["bad"]["model_name"],
        checkpoint_path=cfg["bad"]["checkpoint"],
    )
    sfe = FrozenSigLIP2(model_name=cfg["sfe"]["model_name"])
    gad = ConvNeXtV2GAD(
        checkpoint_path=cfg["gad"]["checkpoint"],
        drop_path_rate=cfg["gad"].get("drop_path_rate", 0.1),
    )
    for name, model, section in (
        ("BAD", bad, "bad"),
        ("SFE", sfe, "sfe"),
        ("GAD", gad, "gad"),
    ):
        expected = int(cfg[section]["feature_dim"])
        if model.feature_dim != expected:
            raise RuntimeError(
                f"{name} feature dim actual={model.feature_dim}, config={expected}"
            )
    return Stage2Hybrid(
        bad=bad,
        sfe=sfe,
        gad=gad,
        bad_dim=bad.feature_dim,
        sfe_dim=sfe.feature_dim,
        gad_dim=gad.feature_dim,
        projection_dim=int(cfg["fusion"]["projection_dim"]),
        dropout=float(cfg["fusion"]["dropout"]),
        bad_branch_dropout=float(cfg["fusion"].get("bad_branch_dropout", 0.0)),
        sfe_branch_dropout=float(cfg["fusion"].get("sfe_branch_dropout", 0.0)),
        num_classes=int(cfg["fusion"]["num_classes"]),
    )
