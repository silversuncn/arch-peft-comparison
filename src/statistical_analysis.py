#!/usr/bin/env python3
"""Cross-Architecture PEFT — Statistical Analysis

Block A: Architecture main effect (ANOVA per method x task stratum)
Block B: Pairwise architecture comparisons (paired t-test, BH-FDR)
Block C: Architecture x method interaction
Block D: Collapse analysis
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "data" / "results.csv"
OUTPUT = ROOT / "data" / "statistical_analysis.json"

ARCHS = ["encoder-only", "decoder-only", "encoder-decoder"]
METHODS = ["lora", "bitfit", "ia3"]
TASKS = ["sst2", "mrpc", "qnli", "rte"]
SIZES = [48, 96, 192, 384]
SEEDS = [7, 13, 29]
ARCH_PAIRS = [("encoder-only", "decoder-only"), ("encoder-only", "encoder-decoder"), ("decoder-only", "encoder-decoder")]


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    df["accuracy"] = df["accuracy"].astype(float)
    df["sample_size"] = df["sample_size"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["collapsed"] = df["collapsed"].astype(bool)
    return df


def bh_correction(p_values):
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(sorted_p[i] * n / (i + 1), adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result.tolist()


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def block_a_anova(df):
    results = {}
    for method in METHODS:
        for task in TASKS:
            key = f"{method}__{task}"
            sub = df[(df["method"] == method) & (df["task"] == task)]
            groups = [sub[sub["arch"] == a]["accuracy"].values for a in ARCHS]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                results[key] = {"error": "insufficient groups"}
                continue
            try:
                F, p = stats.f_oneway(*groups)
                results[key] = {
                    "F": float(F) if np.isfinite(F) else None,
                    "p": float(p) if np.isfinite(p) else None,
                    "significant": bool(p < 0.05) if np.isfinite(p) else False,
                    "n_per_arch": [len(g) for g in groups],
                    "means_per_arch": {a: float(np.mean(sub[sub["arch"]==a]["accuracy"])) for a in ARCHS},
                }
            except Exception as e:
                results[key] = {"error": str(e)}
    return results


def block_b_pairwise(df):
    comparisons = []
    for method in METHODS:
        for task in TASKS:
            for size in SIZES:
                for a1, a2 in ARCH_PAIRS:
                    accs1 = df[(df["arch"]==a1) & (df["method"]==method) & (df["task"]==task) & (df["sample_size"]==size)].set_index("seed")["accuracy"]
                    accs2 = df[(df["arch"]==a2) & (df["method"]==method) & (df["task"]==task) & (df["sample_size"]==size)].set_index("seed")["accuracy"]
                    common = sorted(set(accs1.index) & set(accs2.index))
                    if len(common) < 2:
                        continue
                    d = (accs1[common] - accs2[common]).values
                    mean_d = float(np.mean(d))
                    t_stat, p_raw = stats.ttest_1samp(d, 0) if np.std(d, ddof=1) > 0 else (0.0, 1.0)
                    cd = cohens_d(accs1[common].values, accs2[common].values)
                    comparisons.append({
                        "arch1": a1, "arch2": a2,
                        "method": method, "task": task, "sample_size": size,
                        "mean_delta": mean_d,
                        "t_stat": float(t_stat), "p_raw": float(p_raw),
                        "cohens_d": cd,
                        "n_seeds": len(common),
                        "direction": f"{a1} > {a2}" if mean_d > 0 else f"{a2} > {a1}" if mean_d < 0 else "equal",
                    })

    p_raw = [c["p_raw"] for c in comparisons]
    p_adj = bh_correction(p_raw)
    for i, c in enumerate(comparisons):
        c["p_adjusted"] = p_adj[i]
        c["significant"] = bool(p_adj[i] < 0.05)

    n_sig = sum(1 for c in comparisons if c["significant"])
    direction_counts = Counter()
    for c in comparisons:
        if c["significant"]:
            direction_counts[c["direction"]] += 1

    return {
        "comparisons": comparisons,
        "n_total": len(comparisons),
        "n_significant": n_sig,
        "significant_direction_counts": dict(direction_counts),
        "fdr_method": "Benjamini-Hochberg",
        "alpha": 0.05,
    }


def block_c_interaction(df):
    results = {}
    for task in TASKS:
        for size in SIZES:
            key = f"{task}__n{size}"
            sub = df[(df["task"]==task) & (df["sample_size"]==size)]
            best_method_per_arch = {}
            for arch in ARCHS:
                arch_sub = sub[sub["arch"]==arch]
                if len(arch_sub) == 0:
                    continue
                method_means = arch_sub.groupby("method")["accuracy"].mean()
                best_method_per_arch[arch] = str(method_means.idxmax())
            results[key] = {
                "best_method_per_arch": best_method_per_arch,
                "methods_agree": len(set(best_method_per_arch.values())) == 1,
            }

    n_agree = sum(1 for v in results.values() if v.get("methods_agree"))
    return {
        "per_cell": results,
        "n_cells": len(results),
        "n_method_agreement": n_agree,
        "method_agreement_rate": n_agree / len(results) if results else 0,
    }


def block_d_collapse(df):
    arch_collapse = {}
    for arch in ARCHS:
        sub = df[df["arch"]==arch]
        n = len(sub)
        c = int(sub["collapsed"].sum())
        arch_collapse[arch] = {"collapsed": c, "total": n, "rate": round(c/n, 4) if n else 0}

    method_collapse = {}
    for method in METHODS:
        sub = df[df["method"]==method]
        n = len(sub)
        c = int(sub["collapsed"].sum())
        method_collapse[method] = {"collapsed": c, "total": n, "rate": round(c/n, 4) if n else 0}

    task_collapse = {}
    for task in TASKS:
        sub = df[df["task"]==task]
        n = len(sub)
        c = int(sub["collapsed"].sum())
        task_collapse[task] = {"collapsed": c, "total": n, "rate": round(c/n, 4) if n else 0}

    size_collapse = {}
    for size in SIZES:
        sub = df[df["sample_size"]==size]
        n = len(sub)
        c = int(sub["collapsed"].sum())
        size_collapse[str(size)] = {"collapsed": c, "total": n, "rate": round(c/n, 4) if n else 0}

    fine = {}
    for arch in ARCHS:
        for method in METHODS:
            for task in TASKS:
                for size in SIZES:
                    sub = df[(df["arch"]==arch)&(df["method"]==method)&(df["task"]==task)&(df["sample_size"]==size)]
                    n = len(sub)
                    c = int(sub["collapsed"].sum())
                    key = f"{arch}__{method}__{task}__n{size}"
                    fine[key] = {"collapsed": c, "total": n, "rate": round(c/n,4) if n else 0}

    n_fine_over20 = sum(1 for v in fine.values() if v["rate"] > 0.2)
    n_fine_over50 = sum(1 for v in fine.values() if v["rate"] > 0.5)

    return {
        "by_arch": arch_collapse,
        "by_method": method_collapse,
        "by_task": task_collapse,
        "by_size": size_collapse,
        "fine_grained": fine,
        "summary": {
            "total_collapsed": int(df["collapsed"].sum()),
            "total_runs": len(df),
            "overall_rate": round(int(df["collapsed"].sum()) / len(df), 4),
            "fine_cells_over_20pct": n_fine_over20,
            "fine_cells_over_50pct": n_fine_over50,
            "fine_total_cells": len(fine),
        },
    }


def main():
    df = load_data()
    print(f"Loaded {len(df)} runs from {RESULTS_CSV}")

    print("Block A: Architecture ANOVA...")
    anova = block_a_anova(df)

    print("Block B: Pairwise architecture comparisons...")
    pairwise = block_b_pairwise(df)

    print("Block C: Architecture x method interaction...")
    interaction = block_c_interaction(df)

    print("Block D: Collapse analysis...")
    collapse = block_d_collapse(df)

    output = {
        "block_a_anova": anova,
        "block_b_pairwise": pairwise,
        "block_c_interaction": interaction,
        "block_d_collapse": collapse,
        "metadata": {
            "n_runs": len(df),
            "archs": ARCHS, "methods": METHODS,
            "tasks": TASKS, "sizes": SIZES, "seeds": SEEDS,
        },
    }

    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"\nOutput: {OUTPUT}")

    n_anova_sig = sum(1 for v in anova.values() if isinstance(v, dict) and v.get("significant"))
    print(f"ANOVA: arch effect significant in {n_anova_sig}/{len(anova)} strata")
    print(f"Pairwise: {pairwise['n_significant']}/{pairwise['n_total']} significant after BH")
    if pairwise["significant_direction_counts"]:
        for d, c in sorted(pairwise["significant_direction_counts"].items(), key=lambda x: -x[1]):
            print(f"  {d}: {c}")
    print(f"Interaction: method agrees across archs in {interaction['n_method_agreement']}/{interaction['n_cells']} cells")
    print(f"Collapse: {collapse['summary']['total_collapsed']}/{collapse['summary']['total_runs']} ({collapse['summary']['overall_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
