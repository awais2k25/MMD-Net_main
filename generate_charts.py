import argparse
import os
from typing import Dict, List, cast

import matplotlib.pyplot as plt
import pandas as pd


def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def epoch_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "epoch" not in df.columns:
        return df
    # Sort so that "last" corresponds to the final batch/progress within the epoch
    sort_keys = [k for k in ["epoch", "batch", "epoch_progress"] if k in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys)

    loss_components = ["loss_binary", "loss_multi", "loss_rec_video", "loss_rec_audio", "loss_triplet", "loss_orthogonality"]

    # Build aggregation map
    agg_map = {}
    for col in df.columns:
        if col == "epoch":
            continue
        if col in loss_components:
            # Component losses are 0.0 in the epoch summary row, so we take the mean of non-zero batch values
            agg_map[col] = lambda x: x[x != 0].mean() if (x != 0).any() else 0.0
        else:
            # For overall loss, accuracies, and other metrics, the last row contains the exact epoch summary value
            agg_map[col] = "last"

    grouped = df.groupby("epoch", as_index=False).agg(agg_map)
    return cast(pd.DataFrame, grouped)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_metric(metric: str, train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    has_any = False
    if not train_df.empty and metric in train_df.columns:
        plt.plot(train_df["epoch"], train_df[metric], label=f"train_{metric}")
        has_any = True
    if not val_df.empty and metric in val_df.columns:
        plt.plot(val_df["epoch"], val_df[metric], label=f"val_{metric}")
        has_any = True
    if not has_any:
        plt.close()
        return
    plt.title(metric)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.3)
    if (not train_df.empty and metric in train_df.columns) and (not val_df.empty and metric in val_df.columns):
        plt.legend(loc="best")
    out_path = os.path.join(out_dir, f"{metric}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-metric charts by epoch from CSV logs.")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="/home/i237606/FakeAV_sample/MMD-Net_main_v2_7/logs_test_140_v7.1_64_e1_ablation_recon_full",
        help="Directory containing train_metrics.csv and val_metrics.csv",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="charts",
        help="Subdirectory under logs-dir to save PNG charts",
    )
    args = parser.parse_args()

    train_csv = os.path.join(args.logs_dir, "train_metrics.csv")
    val_csv = os.path.join(args.logs_dir, "val_metrics.csv")

    train_raw: pd.DataFrame = read_csv_safe(train_csv)
    val_raw: pd.DataFrame = read_csv_safe(val_csv)

    # Filter expected split if present, then aggregate by epoch
    if not train_raw.empty and "split" in train_raw.columns:
        train_raw = cast(pd.DataFrame, train_raw[train_raw["split"] == "train"])
    if not val_raw.empty and "split" in val_raw.columns:
        val_raw = cast(pd.DataFrame, val_raw[val_raw["split"] == "val"])

    train_epoch = epoch_aggregate(train_raw)
    val_epoch = epoch_aggregate(val_raw)

    out_dir = os.path.join(args.logs_dir, args.out_subdir)
    ensure_dir(out_dir)

    # Metrics we will attempt to plot if present
    metrics: List[str] = [
        # Losses
        "loss",
        "loss_binary",
        "loss_multi",
        "loss_rec_video",
        "loss_rec_audio",
        "loss_triplet",
        "loss_orthogonality",
        # Binary classification metrics
        "binary_acc",
        "binary_auc",
        "binary_precision",
        "binary_recall",
        "binary_f1",
        # Multi-class metrics
        "multi_acc",
        "multi_auc",
        "multi_precision",
        "multi_recall",
        "multi_f1",
    ]

    # Plot each metric if available in either split
    for m in metrics:
        plot_metric(m, train_epoch, val_epoch, out_dir)

    # Also create a simple combined loss figure: train/val overall loss only
    plot_metric("loss", train_epoch, val_epoch, out_dir)


if __name__ == "__main__":
    main()


