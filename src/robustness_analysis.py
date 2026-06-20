#!/usr/bin/env python3
"""Cross-Architecture PEFT Phase 4.5: Robustness Analyses

Analysis 1: Friedman test (paired non-parametric) replacing exploratory ANOVA
Analysis 2: Within-cell rank and regret summary
Analysis 3: Collapse sensitivity (re-examine conclusions excluding collapsed cells)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "data" / "results.csv"
OUTPUT = ROOT / "data" / "robustness_analysis.json"

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
        adjusted[i] = sorted_p[i] if i == n - 1 else min(sorted_p[i] * n / (i + 1), adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result.tolist()


# ============================================================
# Analysis 1: Friedman test
# ============================================================
def analysis_1_friedman(df):
    """Per method×task stratum, use size×seed as blocks, compare 3 architectures."""
    results = {}
    all_p = []
    test_keys = []

    for method in METHODS:
        for task in TASKS:
            key = f"{method}__{task}"
            # Build block matrix: rows = (size, seed) blocks, columns = architectures
            blocks = []
            for size in SIZES:
                for seed in SEEDS:
                    row = []
                    valid = True
                    for arch in ARCHS:
                        sub = df[(df["arch"]==arch) & (df["method"]==method) &
                                 (df["task"]==task) & (df["sample_size"]==size) & (df["seed"]==seed)]
                        if len(sub) != 1:
                            valid = False
                            break
                        row.append(float(sub["accuracy"].iloc[0]))
                    if valid:
                        blocks.append(row)

            if len(blocks) < 3:
                results[key] = {"error": "insufficient blocks", "n_blocks": len(blocks)}
                continue

            block_matrix = np.array(blocks)  # shape: (n_blocks, 3)
            try:
                stat, p = stats.friedmanchisquare(*[block_matrix[:, i] for i in range(3)])
                # Post-hoc: Wilcoxon signed-rank for each arch pair
                posthoc = {}
                posthoc_p = []
                for a1_idx, a2_idx in [(0,1), (0,2), (1,2)]:
                    pair_key = f"{ARCHS[a1_idx]}_vs_{ARCHS[a2_idx]}"
                    try:
                        w_stat, w_p = stats.wilcoxon(block_matrix[:, a1_idx], block_matrix[:, a2_idx])
                    except ValueError:
                        w_stat, w_p = 0.0, 1.0
                    mean_diff = float(np.mean(block_matrix[:, a1_idx] - block_matrix[:, a2_idx]))
                    posthoc[pair_key] = {
                        "W": float(w_stat), "p_raw": float(w_p),
                        "mean_diff": mean_diff,
                        "direction": f"{ARCHS[a1_idx]} > {ARCHS[a2_idx]}" if mean_diff > 0 else f"{ARCHS[a2_idx]} > {ARCHS[a1_idx]}",
                    }
                    posthoc_p.append(w_p)

                # BH correct posthoc within this stratum
                posthoc_adj = bh_correction(posthoc_p)
                for i, pair_key in enumerate(posthoc):
                    posthoc[pair_key]["p_adjusted"] = posthoc_adj[i]
                    posthoc[pair_key]["significant"] = bool(posthoc_adj[i] < 0.05)

                # Mean ranks
                ranks = np.zeros_like(block_matrix)
                for r in range(len(blocks)):
                    ranks[r] = stats.rankdata(-block_matrix[r])  # rank 1 = best
                mean_ranks = {ARCHS[i]: float(ranks[:, i].mean()) for i in range(3)}

                results[key] = {
                    "chi2": float(stat), "p": float(p),
                    "significant": bool(p < 0.05),
                    "n_blocks": len(blocks),
                    "mean_ranks": mean_ranks,
                    "posthoc_wilcoxon": posthoc,
                    "arch_means": {ARCHS[i]: float(block_matrix[:, i].mean()) for i in range(3)},
                }
                all_p.append(p)
                test_keys.append(key)

            except Exception as e:
                results[key] = {"error": str(e)}

    # Global BH correction across all Friedman tests
    if all_p:
        global_adj = bh_correction(all_p)
        for i, key in enumerate(test_keys):
            results[key]["p_global_adjusted"] = global_adj[i]
            results[key]["significant_global"] = bool(global_adj[i] < 0.05)

    n_sig = sum(1 for k in test_keys if results[k].get("significant"))
    n_sig_global = sum(1 for k in test_keys if results[k].get("significant_global"))

    return {
        "per_stratum": results,
        "n_strata": len(test_keys),
        "n_significant_raw": n_sig,
        "n_significant_global_bh": n_sig_global,
        "_note": "Friedman test with size×seed blocks; paired non-parametric; replaces exploratory ANOVA.",
    }


# ============================================================
# Analysis 2: Within-cell rank and regret
# ============================================================
def analysis_2_rank_regret(df):
    """For each task×size×seed×method cell, rank the 3 architectures."""
    arch_ranks = {a: [] for a in ARCHS}
    method_ranks_by_arch = {a: {m: [] for m in METHODS} for a in ARCHS}
    regrets = {a: [] for a in ARCHS}

    # Architecture ranking: within each method×task×size×seed cell
    for method in METHODS:
        for task in TASKS:
            for size in SIZES:
                for seed in SEEDS:
                    accs = {}
                    for arch in ARCHS:
                        sub = df[(df["arch"]==arch) & (df["method"]==method) &
                                 (df["task"]==task) & (df["sample_size"]==size) & (df["seed"]==seed)]
                        if len(sub) == 1:
                            accs[arch] = float(sub["accuracy"].iloc[0])
                    if len(accs) == 3:
                        best = max(accs.values())
                        ranked = sorted(accs.items(), key=lambda x: -x[1])
                        for rank_idx, (arch, acc) in enumerate(ranked):
                            arch_ranks[arch].append(rank_idx + 1)  # 1=best
                            regrets[arch].append(best - acc)

    # Method ranking: within each arch×task×size×seed cell
    for arch in ARCHS:
        for task in TASKS:
            for size in SIZES:
                for seed in SEEDS:
                    accs = {}
                    for method in METHODS:
                        sub = df[(df["arch"]==arch) & (df["method"]==method) &
                                 (df["task"]==task) & (df["sample_size"]==size) & (df["seed"]==seed)]
                        if len(sub) == 1:
                            accs[method] = float(sub["accuracy"].iloc[0])
                    if len(accs) == 3:
                        ranked = sorted(accs.items(), key=lambda x: -x[1])
                        for rank_idx, (method, acc) in enumerate(ranked):
                            method_ranks_by_arch[arch][method].append(rank_idx + 1)

    arch_summary = {}
    for arch in ARCHS:
        r = arch_ranks[arch]
        reg = regrets[arch]
        arch_summary[arch] = {
            "mean_rank": float(np.mean(r)) if r else None,
            "rank_1_count": r.count(1),
            "rank_3_count": r.count(3),
            "n_cells": len(r),
            "mean_regret": float(np.mean(reg)) if reg else None,
            "median_regret": float(np.median(reg)) if reg else None,
            "max_regret": float(np.max(reg)) if reg else None,
        }

    method_summary = {}
    for arch in ARCHS:
        method_summary[arch] = {}
        for method in METHODS:
            r = method_ranks_by_arch[arch][method]
            method_summary[arch][method] = {
                "mean_rank": float(np.mean(r)) if r else None,
                "rank_1_count": r.count(1),
                "n_cells": len(r),
            }

    # Best method per arch (by rank-1 frequency)
    best_method_per_arch = {}
    for arch in ARCHS:
        best_m = max(METHODS, key=lambda m: method_summary[arch][m]["rank_1_count"])
        best_method_per_arch[arch] = {
            "method": best_m,
            "rank_1_count": method_summary[arch][best_m]["rank_1_count"],
            "total_cells": method_summary[arch][best_m]["n_cells"],
        }

    return {
        "architecture_ranking": arch_summary,
        "method_ranking_by_arch": method_summary,
        "best_method_per_arch": best_method_per_arch,
        "methods_agree_across_archs": len(set(v["method"] for v in best_method_per_arch.values())) == 1,
        "_note": "Rank 1 = best in cell. Regret = best_in_cell - this_arch. Avoids cross-task raw mean aggregation.",
    }


# ============================================================
# Analysis 3: Collapse sensitivity
# ============================================================
def analysis_3_collapse_sensitivity(df):
    """Re-examine architecture ranking and method agreement after excluding collapsed cells."""
    df_clean = df[~df["collapsed"]].copy()
    n_excluded = len(df) - len(df_clean)

    # Recompute method agreement on clean data
    agreement_full = 0
    agreement_clean = 0
    total_cells = 0
    per_cell = {}

    for task in TASKS:
        for size in SIZES:
            total_cells += 1
            # Full data
            best_full = {}
            best_clean = {}
            for arch in ARCHS:
                sub_full = df[(df["arch"]==arch) & (df["task"]==task) & (df["sample_size"]==size)]
                sub_clean = df_clean[(df_clean["arch"]==arch) & (df_clean["task"]==task) & (df_clean["sample_size"]==size)]
                if len(sub_full) > 0:
                    best_full[arch] = sub_full.groupby("method")["accuracy"].mean().idxmax()
                if len(sub_clean) > 0:
                    best_clean[arch] = sub_clean.groupby("method")["accuracy"].mean().idxmax()

            full_agree = len(set(best_full.values())) == 1 if len(best_full) == 3 else False
            clean_agree = len(set(best_clean.values())) == 1 if len(best_clean) == 3 else False
            if full_agree:
                agreement_full += 1
            if clean_agree:
                agreement_clean += 1

            per_cell[f"{task}__n{size}"] = {
                "best_method_full": best_full,
                "best_method_clean": best_clean,
                "agree_full": full_agree,
                "agree_clean": clean_agree,
                "changed": best_full != best_clean,
            }

    # Recompute arch mean ranks on clean data
    arch_ranks_clean = {a: [] for a in ARCHS}
    for method in METHODS:
        for task in TASKS:
            for size in SIZES:
                for seed in SEEDS:
                    accs = {}
                    for arch in ARCHS:
                        sub = df_clean[(df_clean["arch"]==arch) & (df_clean["method"]==method) &
                                       (df_clean["task"]==task) & (df_clean["sample_size"]==size) & (df_clean["seed"]==seed)]
                        if len(sub) == 1:
                            accs[arch] = float(sub["accuracy"].iloc[0])
                    if len(accs) == 3:
                        ranked = sorted(accs.items(), key=lambda x: -x[1])
                        for rank_idx, (arch, acc) in enumerate(ranked):
                            arch_ranks_clean[arch].append(rank_idx + 1)

    arch_rank_summary_clean = {}
    for arch in ARCHS:
        r = arch_ranks_clean[arch]
        arch_rank_summary_clean[arch] = {
            "mean_rank": float(np.mean(r)) if r else None,
            "rank_1_count": r.count(1) if r else 0,
            "n_cells": len(r),
        }

    return {
        "runs_excluded": n_excluded,
        "runs_remaining": len(df_clean),
        "method_agreement_full": {"agree": agreement_full, "total": total_cells},
        "method_agreement_clean": {"agree": agreement_clean, "total": total_cells},
        "agreement_changed": agreement_full != agreement_clean,
        "arch_ranks_clean": arch_rank_summary_clean,
        "per_cell_detail": per_cell,
        "conclusion_stable": abs(agreement_full - agreement_clean) <= 1,
        "_note": "Sensitivity check: excluding collapsed runs. If conclusions hold, collapse is not driving the findings.",
    }


def main():
    df = load_data()
    print(f"Loaded {len(df)} runs")

    print("\nAnalysis 1: Friedman tests...")
    friedman = analysis_1_friedman(df)

    print("Analysis 2: Rank/regret summary...")
    rank_regret = analysis_2_rank_regret(df)

    print("Analysis 3: Collapse sensitivity...")
    collapse_sens = analysis_3_collapse_sensitivity(df)

    output = {
        "analysis_1_friedman": friedman,
        "analysis_2_rank_regret": rank_regret,
        "analysis_3_collapse_sensitivity": collapse_sens,
    }

    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"\nOutput: {OUTPUT}")

    # Summary
    print(f"\n{'='*60}")
    print("PHASE 4.5 SUMMARY")
    print(f"{'='*60}")

    print(f"\n[Friedman] {friedman['n_significant_raw']}/{friedman['n_strata']} strata significant (raw p<0.05)")
    print(f"[Friedman] {friedman['n_significant_global_bh']}/{friedman['n_strata']} strata significant (global BH)")
    for key, v in friedman["per_stratum"].items():
        if isinstance(v, dict) and v.get("significant"):
            ranks = v.get("mean_ranks", {})
            print(f"  {key}: chi2={v['chi2']:.2f} p={v['p']:.4f} ranks={ranks}")

    print(f"\n[Rank/Regret] Architecture mean ranks:")
    for arch, v in rank_regret["architecture_ranking"].items():
        print(f"  {arch}: mean_rank={v['mean_rank']:.2f} rank1={v['rank_1_count']}/{v['n_cells']} regret={v['mean_regret']:.4f}")
    print(f"  Best method per arch: {rank_regret['best_method_per_arch']}")
    print(f"  Methods agree across archs: {rank_regret['methods_agree_across_archs']}")

    print(f"\n[Collapse Sensitivity]")
    print(f"  Excluded: {collapse_sens['runs_excluded']} collapsed runs")
    print(f"  Method agreement full: {collapse_sens['method_agreement_full']}")
    print(f"  Method agreement clean: {collapse_sens['method_agreement_clean']}")
    print(f"  Conclusion stable: {collapse_sens['conclusion_stable']}")
    for arch, v in collapse_sens["arch_ranks_clean"].items():
        print(f"  {arch} (clean): mean_rank={v['mean_rank']:.2f} rank1={v['rank_1_count']}/{v['n_cells']}")


if __name__ == "__main__":
    main()
