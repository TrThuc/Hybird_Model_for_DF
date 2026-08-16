from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = [
    ("celebdf", "Celeb-DF-v2"),
    ("dfd", "DeepFakeDetection"),
    ("dfdc", "DFDC"),
    ("dfdcp", "DFDCP"),
    ("faceshifter", "FaceShifter"),
    ("uadfv", "UADFV"),
]


def load_csv(path: Path, label_col: str, score_col: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    for column in (label_col, score_col):
        if column not in df.columns:
            raise ValueError(
                f"{path.name} is missing column '{column}'. "
                f"Available columns: {list(df.columns)}"
            )

    df = df[[label_col, score_col]].copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[label_col, score_col])
    df = df[df[label_col].isin([0, 1])]
    df = df[df[score_col].between(0, 1)]
    df[label_col] = df[label_col].astype(int)
    if df.empty:
        raise ValueError(f"{path.name} contains no valid prediction rows.")
    return df


def plot_one(ax, df, name, label_col, score_col, bins, y_max, density):
    real = df.loc[df[label_col] == 0, score_col].to_numpy()
    fake = df.loc[df[label_col] == 1, score_col].to_numpy()
    edges = np.linspace(0, 1, bins + 1)

    ax.hist(real, bins=edges, alpha=0.55, label=f"Real {name}", density=density)
    ax.hist(fake, bins=edges, alpha=0.55, label=f"Fake {name}", density=density)
    ax.set_title(f"Histogram of Predictions for {name}")
    ax.set_xlabel("Predicted fake probability")
    ax.set_ylabel("Density" if density else "Frequency")
    ax.set_xlim(0, 1)
    if y_max is not None and not density:
        ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)

    return {
        "dataset": name,
        "total_frames": len(df),
        "real_frames": len(real),
        "fake_frames": len(fake),
        "real_mean_score": real.mean() if len(real) else np.nan,
        "fake_mean_score": fake.mean() if len(fake) else np.nan,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot real/fake prediction-score histograms for cross-test datasets."
    )
    parser.add_argument("--celebdf", type=Path, default=SCRIPT_DIR / "Celeb-DF-v2.csv")
    parser.add_argument(
        "--dfd", type=Path, default=SCRIPT_DIR / "DeepFakeDetection.csv"
    )
    parser.add_argument("--dfdc", type=Path, default=SCRIPT_DIR / "DFDC.csv")
    parser.add_argument("--dfdcp", type=Path, default=SCRIPT_DIR / "DFDCP.csv")
    parser.add_argument(
        "--faceshifter", type=Path, default=SCRIPT_DIR / "FaceShifter.csv"
    )
    parser.add_argument("--uadfv", type=Path, default=SCRIPT_DIR / "UADFV.csv")
    parser.add_argument(
        "--output", type=Path, default=SCRIPT_DIR / "histogram_results"
    )
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Maximum frequency on the Y axis; omit for automatic scaling.",
    )
    parser.add_argument("--density", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be greater than zero.")

    y_max = None if args.y_max is None or args.y_max < 0 else args.y_max
    args.output.mkdir(parents=True, exist_ok=True)
    individual_dir = args.output / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for key, name in DATASETS:
        df = load_csv(getattr(args, key), args.label_column, args.score_column)
        loaded[key] = df
        print(
            f"{name}: total={len(df)}, "
            f"real={(df[args.label_column] == 0).sum()}, "
            f"fake={(df[args.label_column] == 1).sum()}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    summaries = []
    for ax, (key, name) in zip(axes.flatten(), DATASETS):
        df = loaded[key]
        summaries.append(
            plot_one(
                ax,
                df,
                name,
                args.label_column,
                args.score_column,
                args.bins,
                y_max,
                args.density,
            )
        )

        single_fig, single_ax = plt.subplots(figsize=(8, 5.5))
        plot_one(
            single_ax,
            df,
            name,
            args.label_column,
            args.score_column,
            args.bins,
            y_max,
            args.density,
        )
        single_fig.tight_layout()
        safe_name = name.lower().replace("-", "_").replace(" ", "_")
        single_fig.savefig(
            individual_dir / f"{safe_name}_histogram.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(single_fig)

    fig.tight_layout()
    combined_name = (
        "prediction_histograms_density.png"
        if args.density
        else "prediction_histograms_frequency.png"
    )
    combined_path = args.output / combined_name
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(summaries).to_csv(
        args.output / "histogram_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved: {combined_path.resolve()}")
    print("All valid prediction rows were used; y_max only limits the Y axis.")


if __name__ == "__main__":
    main()
