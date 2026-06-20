#!/usr/bin/env python3
"""Cross-Architecture PEFT — Publication Figures

Fig 1: Architecture x Method heatmap (mean accuracy, aggregated over tasks/sizes)
Fig 2: Accuracy vs sample size by architecture (one panel per method)
Fig 3: Collapse rate by architecture x task
Fig 4: Within-cell architecture rank distribution (rank-1/2/3 stacked bar)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "data" / "results.csv"
STATS_JSON = ROOT / "data" / "statistical_analysis.json"
ROBUSTNESS_JSON = ROOT / "data" / "robustness_analysis.json"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ARCHS = ["encoder-only", "decoder-only", "encoder-decoder"]
ARCH_LABELS = {"encoder-only": "BERT", "decoder-only": "Qwen3", "encoder-decoder": "T5-base"}
METHODS = ["lora", "bitfit", "ia3"]
METHOD_LABELS = {"lora": "LoRA", "bitfit": "BitFit", "ia3": "(IA)³"}
TASKS = ["sst2", "mrpc", "qnli", "rte"]
TASK_LABELS = {"sst2": "SST-2", "mrpc": "MRPC", "qnli": "QNLI", "rte": "RTE"}
SIZES = [48, 96, 192, 384]
COLORS = {"encoder-only": "#2196F3", "decoder-only": "#FF9800", "encoder-decoder": "#4CAF50"}


def load():
    df = pd.read_csv(RESULTS_CSV)
    df["accuracy"] = df["accuracy"].astype(float)
    df["sample_size"] = df["sample_size"].astype(int)
    with open(STATS_JSON) as f:
        stats = json.load(f)
    with open(ROBUSTNESS_JSON) as f:
        robustness = json.load(f)
    return df, stats, robustness


def fig1_heatmap(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    data = np.zeros((len(METHODS), len(ARCHS)))
    for i, method in enumerate(METHODS):
        for j, arch in enumerate(ARCHS):
            sub = df[(df["method"]==method) & (df["arch"]==arch)]
            data[i, j] = sub["accuracy"].mean()

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=0.85)
    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS], fontsize=10)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=10)
    for i in range(len(METHODS)):
        for j in range(len(ARCHS)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean Accuracy", shrink=0.8)
    ax.set_title("Mean Accuracy by Architecture and PEFT Method", fontsize=11)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig1_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig 1: heatmap saved")


def fig2_scaling(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        for arch in ARCHS:
            means, stds = [], []
            for size in SIZES:
                sub = df[(df["method"]==method) & (df["arch"]==arch) & (df["sample_size"]==size)]
                means.append(sub["accuracy"].mean())
                stds.append(sub["accuracy"].std())
            ax.errorbar(SIZES, means, yerr=stds, marker="o", label=ARCH_LABELS[arch],
                        color=COLORS[arch], capsize=3, linewidth=1.5, markersize=5)
        ax.set_xlabel("Sample Size", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(METHOD_LABELS[method], fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(s) for s in SIZES])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle("Accuracy vs Sample Size by Architecture", fontsize=12, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig2_scaling.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig 2: scaling curves saved")


def fig3_collapse(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(TASKS))
    width = 0.25
    for i, arch in enumerate(ARCHS):
        rates = []
        for task in TASKS:
            sub = df[(df["arch"]==arch) & (df["task"]==task)]
            rates.append(sub["collapsed"].mean())
        bars = ax.bar(x + i*width, rates, width, label=ARCH_LABELS[arch], color=COLORS[arch], alpha=0.85)
        for bar, r in zip(bars, rates):
            if r > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{r:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=10)
    ax.set_ylabel("Collapse Rate", fontsize=10)
    ax.set_title("Collapse Rate by Architecture and Task", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.65)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig3_collapse.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig 3: collapse rates saved")


def fig4_rank_distribution(robustness):
    ra = robustness["analysis_2_rank_regret"]["architecture_ranking"]
    fig, ax = plt.subplots(figsize=(6, 3.8))

    rank1, rank2, rank3 = [], [], []
    for arch in ARCHS:
        r = ra[arch]
        n = r["n_cells"]
        r1 = r["rank_1_count"]
        r3 = r["rank_3_count"]
        r2 = n - r1 - r3
        rank1.append(r1)
        rank2.append(r2)
        rank3.append(r3)

    x = np.arange(len(ARCHS))
    labels = [ARCH_LABELS[a] for a in ARCHS]
    b1 = ax.bar(x, rank1, color="#2E7D32", label="Rank 1 (best)")
    b2 = ax.bar(x, rank2, bottom=rank1, color="#FBC02D", label="Rank 2")
    b3 = ax.bar(x, rank3, bottom=np.array(rank1)+np.array(rank2), color="#C62828", label="Rank 3 (worst)")

    for i in range(len(ARCHS)):
        ax.text(i, rank1[i]/2, str(rank1[i]), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(i, rank1[i]+rank2[i]/2, str(rank2[i]), ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(i, rank1[i]+rank2[i]+rank3[i]/2, str(rank3[i]), ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Number of cells (out of 144)", fontsize=10)
    ax.set_title("Within-Cell Architecture Rank Distribution", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 160)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig4_rank_distribution.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("Fig 4: rank distribution saved")


def main():
    df, stats, robustness = load()
    print(f"Loaded {len(df)} runs")
    fig1_heatmap(df)
    fig2_scaling(df)
    fig3_collapse(df)
    fig4_rank_distribution(robustness)
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
