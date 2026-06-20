#!/usr/bin/env python3
"""Cross-Architecture PEFT: Aggregate per-run metrics.json into results.csv + summary.json.

Validation-first design: every per-run artifact is parsed and validated, and the
full grid is checked against the expected Cartesian product BEFORE any output file
is written. If validation fails, no file under data/ is modified and the script
exits non-zero. On success, outputs are written atomically (temp file + replace).
"""
import csv
import json
import os
import sys
import math
import tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "final_runs"
OUTPUT_CSV = ROOT / "data" / "results.csv"
OUTPUT_JSON = ROOT / "data" / "results.json"
SUMMARY_JSON = ROOT / "data" / "summary.json"

# Expected full grid (must match grid_runner.py).
ARCHS = ["encoder-only", "decoder-only", "encoder-decoder"]
METHODS = ["lora", "bitfit", "ia3"]
TASKS = ["sst2", "mrpc", "qnli", "rte"]
SIZES = [48, 96, 192, 384]
SEEDS = [7, 13, 29]
EXPECTED_KEYS = {
    (a, me, t, sz, s)
    for a in ARCHS for me in METHODS for t in TASKS for sz in SIZES for s in SEEDS
}
EXPECTED_RUNS = len(EXPECTED_KEYS)  # 432

# Expected experiment configuration (must match grid_runner.py).
MODEL_ID_BY_ARCH = {
    "encoder-only": "bert-base-uncased",
    "decoder-only": "Qwen/Qwen3-0.6B",
    "encoder-decoder": "google-t5/t5-base",
}
EXPECTED_HP = {  # scalar hyperparameters shared by all runs
    "epochs": 10,
    "lr": 2e-4,
    "batch_size": 16,
    "max_length": 128,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "dtype": "bfloat16",
}
# LoRA-specific: r=8, alpha=16 for lora; both null otherwise.
LORA_R, LORA_ALPHA = 8, 16

# train_time_sec is intentionally excluded: it is a wall-clock diagnostic not used
# in any analysis, and the raw archive contains a non-monotonic negative value.
FIELDS = [
    "arch", "model_id", "method", "task", "sample_size", "seed",
    "accuracy", "majority_baseline", "collapsed", "exploded",
    "max_grad_norm_observed",
    "trainable_params", "total_params", "trainable_ratio",
]


ID_FIELDS = ["arch", "model_id", "method", "task", "sample_size", "seed"]
METRIC_FIELDS = ["accuracy", "majority_baseline", "collapsed", "exploded",
                 "max_grad_norm_observed",
                 "trainable_params", "total_params", "trainable_ratio"]
CONFIG_HP_FIELDS = ["epochs", "lr", "batch_size", "max_length",
                    "max_grad_norm", "warmup_ratio", "dtype", "lora_r", "lora_alpha"]


def validate_config(name, c, errors):
    """Verify model_id and all hyperparameters match the expected configuration."""
    ok = True
    arch = str(c.get("arch"))
    method = str(c.get("method"))
    exp_model = MODEL_ID_BY_ARCH.get(arch)
    if c.get("model_id") != exp_model:
        errors.append(f"{name}: model_id {c.get('model_id')!r} != expected {exp_model!r}")
        ok = False
    for k, exp in EXPECTED_HP.items():
        got = c.get(k)
        if isinstance(exp, float):
            try:
                if not math.isclose(float(got), exp, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{name}: {k}={got!r} != expected {exp}"); ok = False
        else:
            if got != exp:
                errors.append(f"{name}: {k}={got!r} != expected {exp!r}"); ok = False
    exp_r = LORA_R if method == "lora" else None
    exp_a = LORA_ALPHA if method == "lora" else None
    if c.get("lora_r") != exp_r:
        errors.append(f"{name}: lora_r={c.get('lora_r')!r} != expected {exp_r!r}"); ok = False
    if c.get("lora_alpha") != exp_a:
        errors.append(f"{name}: lora_alpha={c.get('lora_alpha')!r} != expected {exp_a!r}"); ok = False
    return ok


def parse_run_name(name):
    """run_id = {arch}__{method}__{task}__n{size}__s{seed}. Returns identity tuple or None."""
    parts = name.split("__")
    if len(parts) != 5:
        return None
    arch, method, task, sz, sd = parts
    if not sz.startswith("n") or not sd.startswith("s"):
        return None
    try:
        return (arch, method, task, int(sz[1:]), int(sd[1:]))
    except ValueError:
        return None


def _is_bool(x):
    return isinstance(x, bool)


def validate_field_values(name, row, errors):
    """Validate types and numeric ranges of all CSV fields. Returns True if OK."""
    ok = True
    a = row["accuracy"]
    try:
        if not (math.isfinite(float(a)) and 0.0 <= float(a) <= 1.0):
            errors.append(f"{name}: accuracy out of [0,1] ({a!r})"); ok = False
    except (TypeError, ValueError):
        errors.append(f"{name}: accuracy not numeric ({a!r})"); ok = False
    mb = row["majority_baseline"]
    try:
        if not (0.0 <= float(mb) <= 1.0):
            errors.append(f"{name}: majority_baseline out of [0,1] ({mb!r})"); ok = False
    except (TypeError, ValueError):
        errors.append(f"{name}: majority_baseline not numeric ({mb!r})"); ok = False
    for b in ("collapsed", "exploded"):
        if not _is_bool(row[b]):
            errors.append(f"{name}: {b} not boolean ({row[b]!r})"); ok = False
    tr = row["trainable_ratio"]
    try:
        if not (0.0 <= float(tr) <= 1.0):
            errors.append(f"{name}: trainable_ratio out of [0,1] ({tr!r})"); ok = False
    except (TypeError, ValueError):
        errors.append(f"{name}: trainable_ratio not numeric ({tr!r})"); ok = False
    for p in ("trainable_params", "total_params"):
        v = row[p]
        if not (isinstance(v, int) and v > 0):
            errors.append(f"{name}: {p} not a positive int ({v!r})"); ok = False
    # max_grad_norm is a (clipped) gradient norm: must be finite and non-negative.
    v = row["max_grad_norm_observed"]
    try:
        if not (math.isfinite(float(v)) and float(v) >= 0.0):
            errors.append(f"{name}: max_grad_norm_observed negative/non-finite ({v!r})"); ok = False
    except (TypeError, ValueError):
        errors.append(f"{name}: max_grad_norm_observed not numeric ({v!r})"); ok = False
    return ok


def load_and_validate():
    """Parse every run dir; return (rows, errors). Does NOT write anything.

    Identity (arch/model_id/method/task/size/seed) is taken from config.json only
    (metrics.json cannot override it). The directory name, the config identity, and
    the expected Cartesian product must all agree.
    """
    rows = []
    errors = []
    seen_keys = defaultdict(list)

    if not ARTIFACTS.is_dir():
        return [], [f"artifacts dir not found: {ARTIFACTS}"]

    for run_dir in sorted(p for p in ARTIFACTS.iterdir() if p.is_dir()):
        name = run_dir.name
        mf = run_dir / "metrics.json"
        cf = run_dir / "config.json"
        if not mf.is_file() or not cf.is_file():
            errors.append(f"{name}: missing metrics.json or config.json")
            continue
        try:
            m = json.loads(mf.read_text())
            c = json.loads(cf.read_text())
        except Exception as e:
            errors.append(f"{name}: JSON parse error ({e})")
            continue

        # Identity must come from config and be complete.
        id_missing = [k for k in ID_FIELDS if c.get(k) is None]
        if id_missing:
            errors.append(f"{name}: config missing identity fields {id_missing}")
            continue
        # Metric fields must come from metrics and be complete.
        metric_missing = [k for k in METRIC_FIELDS if m.get(k) is None]
        if metric_missing:
            errors.append(f"{name}: metrics missing fields {metric_missing}")
            continue

        # Three-way identity agreement: directory name == config == expected combo.
        name_id = parse_run_name(name)
        if name_id is None:
            errors.append(f"{name}: directory name does not parse to a run id")
            continue
        try:
            cfg_id = (str(c["arch"]), str(c["method"]), str(c["task"]),
                      int(c["sample_size"]), int(c["seed"]))
        except (TypeError, ValueError):
            errors.append(f"{name}: config identity not coercible")
            continue
        if name_id != cfg_id:
            errors.append(f"{name}: directory name {name_id} != config identity {cfg_id}")
            continue
        if cfg_id not in EXPECTED_KEYS:
            errors.append(f"{name}: identity {cfg_id} not in expected grid")
            continue

        # Full configuration must match expectation (model_id + hyperparameters).
        if not validate_config(name, c, errors):
            continue

        # Build row: identity from config, metric values from metrics (no override).
        row = {f: c.get(f) for f in ID_FIELDS}
        row.update({f: m.get(f) for f in METRIC_FIELDS})

        if not validate_field_values(name, row, errors):
            continue
        if bool(row["exploded"]):
            errors.append(f"{name}: run exploded=True")
            continue

        seen_keys[cfg_id].append(name)
        rows.append(row)

    # Duplicate combinations.
    for k, v in seen_keys.items():
        if len(v) > 1:
            errors.append(f"duplicate combination {k} in {v}")

    # Cartesian-product completeness.
    present = set(seen_keys.keys())
    missing_combos = EXPECTED_KEYS - present
    extra_combos = present - EXPECTED_KEYS
    if missing_combos:
        errors.append(f"{len(missing_combos)} expected combinations missing "
                      f"(e.g. {sorted(missing_combos)[:3]})")
    if extra_combos:
        errors.append(f"{len(extra_combos)} unexpected combinations present "
                      f"(e.g. {sorted(extra_combos)[:3]})")

    return rows, errors



def atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def build_csv_text(rows):
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def build_summary(rows):
    summary = defaultdict(lambda: {"accuracies": [], "collapsed_count": 0, "n_runs": 0})
    for r in rows:
        key = f"{r['arch']}__{r['method']}__{r['task']}__n{r['sample_size']}"
        summary[key]["accuracies"].append(r["accuracy"])
        summary[key]["collapsed_count"] += int(r["collapsed"]) if r["collapsed"] else 0
        summary[key]["n_runs"] += 1
    out = {}
    for key, v in sorted(summary.items()):
        accs = [a for a in v["accuracies"] if a is not None]
        out[key] = {
            "mean_accuracy": float(np.mean(accs)) if accs else None,
            "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "n_runs": v["n_runs"],
            "collapsed_count": v["collapsed_count"],
            "collapse_rate": v["collapsed_count"] / v["n_runs"] if v["n_runs"] > 0 else 0,
        }
    return out


def main():
    rows, errors = load_and_validate()

    if errors:
        print(f"AGGREGATION FAILED: {len(errors)} problem(s) detected. "
              f"No file under data/ was modified.")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        print(f"Re-run grid_runner.py to complete/repair the grid.")
        return 1

    # All valid. Sort deterministically and write atomically.
    rows.sort(key=lambda r: (str(r["arch"]), str(r["method"]),
                             str(r["task"]), int(r["sample_size"]), int(r["seed"])))
    atomic_write(OUTPUT_CSV, build_csv_text(rows))
    atomic_write(OUTPUT_JSON, json.dumps(rows, indent=2))
    summary_out = build_summary(rows)
    atomic_write(SUMMARY_JSON, json.dumps(summary_out, indent=2))

    print(f"Aggregated {len(rows)} runs (all {EXPECTED_RUNS} combinations present, "
          f"0 exploded, 0 duplicates) -> {OUTPUT_CSV}")
    print(f"Summary: {len(summary_out)} cells -> {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
