"""Cross-Architecture PEFT — Full Grid Runner.

432 runs: 3 architectures x 3 PEFT methods x 4 tasks x 4 sizes x 3 seeds.
Each run is fully reproducible via set_all_seeds().

Usage:
  # Reproducibility smoke (same-seed duplicate check)
  python src/grid_runner.py --repro_smoke

  # Full grid
  python src/grid_runner.py
"""
import os
import sys
import json
import math
import random
import gc
import time
import copy
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    Qwen3ForSequenceClassification, AutoTokenizer,
    T5EncoderModel, T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, IA3Config, get_peft_model

# === Configuration ===
METHODS = ["lora", "bitfit", "ia3"]
TASKS = ["sst2", "mrpc", "qnli", "rte"]
SAMPLE_SIZES = [48, 96, 192, 384]
SEEDS = [7, 13, 29]
EPOCHS = 10
BATCH_SIZE = 16
LR = 2e-4
MAX_LENGTH = 128
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1

TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}

ARCHS = [
    {
        "arch": "encoder-only",
        "model_id": "bert-base-uncased",
        "cls_class": "BertForSequenceClassification",
        "tok_class": "BertTokenizer",
        "lora_targets": ["query", "value"],
        "ia3_targets": ["query", "value", "output.dense"],
        "ia3_ff": ["output.dense"],
    },
    {
        "arch": "decoder-only",
        "model_id": "Qwen/Qwen3-0.6B",
        "cls_class": "Qwen3ForSequenceClassification",
        "tok_class": "AutoTokenizer",
        "lora_targets": ["q_proj", "v_proj"],
        "ia3_targets": ["q_proj", "v_proj", "down_proj"],
        "ia3_ff": ["down_proj"],
    },
    {
        "arch": "encoder-decoder",
        "model_id": "google-t5/t5-base",
        "cls_class": "T5EncoderModel",
        "tok_class": "T5Tokenizer",
        "lora_targets": ["q", "v"],
        "ia3_targets": ["q", "v", "wo"],
        "ia3_ff": ["wo"],
    },
]

ARTIFACT_ROOT = Path(__file__).parent.parent / "artifacts" / "final_runs"

# Set to True via --offline to require pre-cached HuggingFace models/datasets.
# Default False: models and GLUE are downloaded from HuggingFace on first run.
OFFLINE = False


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_task_data(tokenizer, task, size, seed, max_length):
    ds = load_dataset("glue", task, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(size, len(ds))))
    k1, k2 = TASK_TO_KEYS[task]
    if k2 is None:
        texts = list(ds[k1])
        enc = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    else:
        texts1 = list(ds[k1])
        texts2 = list(ds[k2])
        enc = tokenizer(texts1, texts2, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    labels = list(ds["label"])
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)


def load_task_val(tokenizer, task, max_length):
    ds = load_dataset("glue", task, split="validation")
    k1, k2 = TASK_TO_KEYS[task]
    if k2 is None:
        texts = list(ds[k1])
        enc = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    else:
        texts1 = list(ds[k1])
        texts2 = list(ds[k2])
        enc = tokenizer(texts1, texts2, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    labels = list(ds["label"])
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)


def get_majority_baseline(labels):
    counts = np.bincount(labels.numpy())
    return float(counts.max() / len(labels))


def build_model_and_tokenizer(arch_cfg):
    model_id = arch_cfg["model_id"]
    dtype = torch.bfloat16

    if arch_cfg["tok_class"] == "BertTokenizer":
        tokenizer = BertTokenizer.from_pretrained(model_id, local_files_only=OFFLINE)
    elif arch_cfg["tok_class"] == "AutoTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=OFFLINE)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_id, local_files_only=OFFLINE)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if arch_cfg["cls_class"] == "BertForSequenceClassification":
        model = BertForSequenceClassification.from_pretrained(
            model_id, num_labels=2, torch_dtype=dtype, local_files_only=OFFLINE)
    elif arch_cfg["cls_class"] == "Qwen3ForSequenceClassification":
        model = Qwen3ForSequenceClassification.from_pretrained(
            model_id, num_labels=2, torch_dtype=dtype, local_files_only=OFFLINE)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model = T5EncoderModel.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=OFFLINE)

    return model, tokenizer


def apply_peft(model, arch_cfg, method):
    cls_head = None
    device = torch.device("cuda")

    if method == "lora":
        task_type = "SEQ_CLS" if arch_cfg["cls_class"] != "T5EncoderModel" else "FEATURE_EXTRACTION"
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                         target_modules=arch_cfg["lora_targets"], task_type=task_type)
        model = get_peft_model(model, cfg)

    elif method == "bitfit":
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
        if arch_cfg["cls_class"] == "BertForSequenceClassification":
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif arch_cfg["cls_class"] == "Qwen3ForSequenceClassification":
            for p in model.score.parameters():
                p.requires_grad = True

    elif method == "ia3":
        task_type = "SEQ_CLS" if arch_cfg["cls_class"] != "T5EncoderModel" else "FEATURE_EXTRACTION"
        cfg = IA3Config(
            target_modules=arch_cfg["ia3_targets"],
            feedforward_modules=arch_cfg["ia3_ff"],
            task_type=task_type,
        )
        model = get_peft_model(model, cfg)

    if arch_cfg["cls_class"] == "T5EncoderModel":
        cfg = model.config
        if hasattr(cfg, 'd_model'):
            hidden_size = cfg.d_model
        elif hasattr(model, 'base_model'):
            base = model.base_model
            if hasattr(base, 'model'):
                hidden_size = base.model.config.d_model
            else:
                hidden_size = base.config.d_model
        else:
            hidden_size = 768
        cls_head = torch.nn.Linear(hidden_size, 2).to(device=device, dtype=torch.bfloat16)

    model = model.to(device)
    return model, cls_head


def forward_pass(model, cls_head, arch_cfg, input_ids, attn_mask, labels=None):
    if arch_cfg["cls_class"] == "T5EncoderModel":
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        hidden = out.last_hidden_state
        mask_exp = attn_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1)
        logits = cls_head(pooled)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        return logits, loss
    else:
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        return out.logits, out.loss


def run_single(arch_cfg, method, task, size, seed, artifact_root=None):
    if artifact_root is None:
        artifact_root = ARTIFACT_ROOT
    set_all_seeds(seed)
    device = torch.device("cuda")

    model, tokenizer = build_model_and_tokenizer(arch_cfg)
    model, cls_head = apply_peft(model, arch_cfg, method)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if cls_head:
        trainable += sum(p.numel() for p in cls_head.parameters())
        total += sum(p.numel() for p in cls_head.parameters())

    # Data
    input_ids, attn_mask, labels = load_task_data(tokenizer, task, size, seed, MAX_LENGTH)
    val_ids, val_mask, val_labels = load_task_val(tokenizer, task, MAX_LENGTH)
    majority = get_majority_baseline(val_labels)

    input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
    val_ids, val_mask = val_ids.to(device), val_mask.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if cls_head:
        params += list(cls_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

    num_batches = max(1, size // BATCH_SIZE)
    total_steps = EPOCHS * num_batches
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Train
    model.train()
    if cls_head:
        cls_head.train()

    max_grad = 0.0
    exploded = False

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for epoch in range(EPOCHS):
        indices = torch.randperm(len(input_ids), device=device)
        for batch_start in range(0, len(input_ids), BATCH_SIZE):
            batch_idx = indices[batch_start:batch_start+BATCH_SIZE]
            b_ids = input_ids[batch_idx]
            b_mask = attn_mask[batch_idx]
            b_labels = labels[batch_idx]

            optimizer.zero_grad()
            logits, loss = forward_pass(model, cls_head, arch_cfg, b_ids, b_mask, b_labels)

            if not torch.isfinite(loss):
                exploded = True
                break

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            max_grad = max(max_grad, norm.item())
            optimizer.step()
            scheduler.step()

        if exploded:
            break

    torch.cuda.synchronize()
    train_time = time.perf_counter() - t_start

    # Eval
    model.eval()
    if cls_head:
        cls_head.eval()

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(val_ids), 64):
            logits, _ = forward_pass(model, cls_head, arch_cfg,
                                     val_ids[i:i+64], val_mask[i:i+64])
            all_preds.append(logits.float().argmax(-1).cpu())

    preds = torch.cat(all_preds).numpy()
    accuracy = float((preds == val_labels.numpy()).mean())
    collapsed = bool(accuracy < majority)

    # Save — stable run_id (no timestamp for final_runs; timestamp only in repro_smoke)
    run_name = f"{arch_cfg['arch']}__{method}__{task}__n{size}__s{seed}"
    if artifact_root != ARTIFACT_ROOT:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_name}__{ts}"
    run_dir = artifact_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": accuracy,
        "majority_baseline": majority,
        "collapsed": collapsed,
        "exploded": exploded,
        "train_time_sec": train_time,
        "max_grad_norm_observed": max_grad,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total,
    }
    config = {
        "arch": arch_cfg["arch"],
        "model_id": arch_cfg["model_id"],
        "method": method,
        "task": task,
        "sample_size": size,
        "seed": seed,
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "max_grad_norm": MAX_GRAD_NORM,
        "warmup_ratio": WARMUP_RATIO,
        "lora_r": 8 if method == "lora" else None,
        "lora_alpha": 16 if method == "lora" else None,
        "dtype": "bfloat16",
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Cleanup
    del model, optimizer, scheduler, input_ids, attn_mask, labels, val_ids, val_mask
    if cls_head:
        del cls_head
    gc.collect()
    torch.cuda.empty_cache()

    return metrics, config, run_dir


def is_run_complete(run_dir, expected=None, expected_model_id=None):
    """A run counts as complete only if both JSON files parse, carry the required
    fields, the config identity AND full configuration (model_id + hyperparameters)
    match expectation, and the run did not explode. Corrupt/partial/mismatched/
    exploded runs are treated as 'incomplete' so they are re-run on resume rather
    than silently skipped.

    `expected` is (arch, method, task, size, seed); `expected_model_id` is the
    model id for that architecture. When provided, config.json is checked
    field-by-field against them and against the module hyperparameter constants.
    """
    m = run_dir / "metrics.json"
    c = run_dir / "config.json"
    if not m.exists() and not c.exists():
        return "missing"
    if not (m.exists() and c.exists()):
        return "incomplete"
    try:
        md = json.loads(m.read_text())
        cd = json.loads(c.read_text())
    except Exception:
        return "incomplete"
    if md.get("accuracy") is None or md.get("exploded") is None:
        return "incomplete"
    try:
        if not math.isfinite(float(md["accuracy"])):
            return "incomplete"
    except (TypeError, ValueError):
        return "incomplete"
    if bool(md.get("exploded")):
        return "incomplete"
    if expected is not None:
        # Identity check.
        for k in ("arch", "method", "task", "sample_size", "seed"):
            if cd.get(k) is None:
                return "incomplete"
        try:
            cfg_id = (str(cd["arch"]), str(cd["method"]), str(cd["task"]),
                      int(cd["sample_size"]), int(cd["seed"]))
        except (TypeError, ValueError):
            return "incomplete"
        if cfg_id != (str(expected[0]), str(expected[1]), str(expected[2]),
                      int(expected[3]), int(expected[4])):
            return "incomplete"
        method = str(expected[1])
        # Full-config check: model_id + hyperparameters + LoRA params.
        if expected_model_id is not None and cd.get("model_id") != expected_model_id:
            return "incomplete"
        if cd.get("epochs") != EPOCHS or cd.get("batch_size") != BATCH_SIZE:
            return "incomplete"
        if cd.get("max_length") != MAX_LENGTH or cd.get("dtype") != "bfloat16":
            return "incomplete"
        try:
            if not math.isclose(float(cd.get("lr")), LR, rel_tol=1e-9, abs_tol=1e-12):
                return "incomplete"
            if not math.isclose(float(cd.get("max_grad_norm")), MAX_GRAD_NORM, rel_tol=1e-9):
                return "incomplete"
            if not math.isclose(float(cd.get("warmup_ratio")), WARMUP_RATIO, rel_tol=1e-9):
                return "incomplete"
        except (TypeError, ValueError):
            return "incomplete"
        exp_r = 8 if method == "lora" else None
        exp_a = 16 if method == "lora" else None
        if cd.get("lora_r") != exp_r or cd.get("lora_alpha") != exp_a:
            return "incomplete"
    return "complete"


def make_run_id(arch, method, task, size, seed):
    return f"{arch}__{method}__{task}__n{size}__s{seed}"


def main():
    global OFFLINE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--repro_smoke", action="store_true",
                        help="Run 2 identical seed runs and verify metrics match")
    parser.add_argument("--offline", action="store_true",
                        help="Require pre-cached HuggingFace models/datasets (no download). "
                             "Default: models and GLUE are downloaded on first run.")
    parser.add_argument("--archs", nargs="+", default=[a["arch"] for a in ARCHS])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--sizes", nargs="+", type=int, default=SAMPLE_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    OFFLINE = args.offline
    if OFFLINE:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_ETAG_TIMEOUT"] = "0"

    if args.repro_smoke:
        repro_root = Path(__file__).parent.parent / "artifacts" / "repro_smoke"
        print(f"=== Reproducibility Smoke Test (artifacts: {repro_root}) ===")
        arch_cfg = ARCHS[0]  # BERT
        m1, _, _ = run_single(arch_cfg, "lora", "sst2", 48, 7, artifact_root=repro_root)
        m2, _, _ = run_single(arch_cfg, "lora", "sst2", 48, 7, artifact_root=repro_root)
        match = abs(m1["accuracy"] - m2["accuracy"]) < 1e-6
        print(f"Run 1 acc={m1['accuracy']:.6f}, Run 2 acc={m2['accuracy']:.6f}")
        print(f"Reproducibility: {'PASS' if match else 'FAIL'}")
        return 0 if match else 1

    # Filter archs
    arch_map = {a["arch"]: a for a in ARCHS}
    selected_archs = [arch_map[a] for a in args.archs if a in arch_map]

    total_runs = len(selected_archs) * len(args.methods) * len(args.tasks) * len(args.sizes) * len(args.seeds)
    print(f"Cross-Architecture PEFT — Full Grid Runner (PEFT-only)")
    print(f"Matrix: {len(selected_archs)} arch x {len(args.methods)} methods x {len(args.tasks)} tasks x {len(args.sizes)} sizes x {len(args.seeds)} seeds = {total_runs} runs")
    print(f"Artifacts: {ARTIFACT_ROOT}")

    # Progress log
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_log = log_dir / "progress.log"

    grid_start = datetime.now().isoformat()
    n_ok = 0
    n_skip = 0
    n_fail = 0
    n_collapse = 0
    n_exploded = 0
    n_incomplete_rerun = 0

    with open(progress_log, "a") as plog:
        plog.write(f"# Grid started {grid_start}\n")

        run_idx = 0
        for arch_cfg in selected_archs:
            for method in args.methods:
                for task in args.tasks:
                    for size in args.sizes:
                        for seed in args.seeds:
                            run_idx += 1
                            run_id = make_run_id(arch_cfg["arch"], method, task, size, seed)
                            run_dir = ARTIFACT_ROOT / run_id
                            tag = f"{arch_cfg['arch']}/{method}/{task}/n={size}/s={seed}"

                            # Skip/resume check (verify identity + full config)
                            status = is_run_complete(
                                run_dir,
                                expected=(arch_cfg["arch"], method, task, size, seed),
                                expected_model_id=arch_cfg["model_id"])
                            if status == "complete":
                                n_skip += 1
                                plog.write(f"SKIP\t{run_id}\n")
                                plog.flush()
                                print(f"[{run_idx}/{total_runs}] {tag} -> SKIP (already complete)")
                                continue

                            if status == "incomplete":
                                n_incomplete_rerun += 1
                                import shutil
                                shutil.rmtree(run_dir, ignore_errors=True)
                                plog.write(f"INCOMPLETE_RERUN\t{run_id}\n")
                                plog.flush()
                                print(f"[{run_idx}/{total_runs}] {tag} -> INCOMPLETE, rerunning")

                            try:
                                m, c, d = run_single(arch_cfg, method, task, size, seed)
                                if m["exploded"]:
                                    n_exploded += 1
                                    run_status = "EXPLODE"
                                elif m["collapsed"]:
                                    n_collapse += 1
                                    n_ok += 1
                                    run_status = "COLLAPSE"
                                else:
                                    n_ok += 1
                                    run_status = "OK"
                                line = (f"{run_status}\t{arch_cfg['arch']}\t{method}\t{task}\t"
                                        f"{size}\t{seed}\t{m['accuracy']:.6f}\t"
                                        f"{m['collapsed']}\t{m['exploded']}\t"
                                        f"{m['train_time_sec']:.1f}\t{d}\n")
                                plog.write(line)
                                plog.flush()
                                print(f"[{run_idx}/{total_runs}] {tag} -> acc={m['accuracy']:.4f} t={m['train_time_sec']:.1f}s [{run_status}]")
                            except Exception as e:
                                n_fail += 1
                                plog.write(f"FAIL\t{run_id}\t{e}\n")
                                plog.flush()
                                print(f"[{run_idx}/{total_runs}] {tag} -> FAIL: {e}")

                            if run_idx % 50 == 0:
                                gc.collect()
                                torch.cuda.empty_cache()

    grid_end = datetime.now().isoformat()

    # Grid summary
    summary = {
        "expected_runs": total_runs,
        "ok": n_ok,
        "skip": n_skip,
        "fail": n_fail,
        "exploded": n_exploded,
        "incomplete_rerun": n_incomplete_rerun,
        "collapsed": n_collapse,
        "start_time": grid_start,
        "end_time": grid_end,
        "methods": args.methods,
        "tasks": args.tasks,
        "sizes": args.sizes,
        "seeds": args.seeds,
        "archs": args.archs,
    }
    with open(ARTIFACT_ROOT / "grid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GRID COMPLETE: {total_runs} expected | {n_ok} ok | {n_skip} skip | "
          f"{n_fail} fail | {n_exploded} exploded | {n_collapse} collapsed | {n_incomplete_rerun} rerun")
    print(f"Summary: {ARTIFACT_ROOT / 'grid_summary.json'}")
    print(f"Progress: {progress_log}")
    print(f"{'='*60}")

    # Non-zero exit if any run failed or exploded, so an incomplete or unstable
    # grid is never mistaken for a successful full run.
    if n_fail > 0 or n_exploded > 0:
        print(f"WARNING: {n_fail} failed, {n_exploded} exploded. Re-run the same command "
              f"to resume (valid completed runs are skipped; failed/exploded ones re-run).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
