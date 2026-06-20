"""Phase 1B Smoke Test — Pipeline Correctness Verification.

Runs 3 architectures x LoRA x SST-2 x size=48 x seed=7.
Gate: 6 pipeline correctness criteria (NOT accuracy).
"""
import os
import sys
import json
import random
import gc
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    Qwen3ForSequenceClassification, AutoTokenizer,
    T5EncoderModel, T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset

# === Constants ===
SEED = 7
SAMPLE_SIZE = 48
TASK = "sst2"
MAX_LENGTH = 128
BATCH_SIZE = 16
LR = 2e-4
EPOCHS = 2  # short for smoke test
ARTIFACT_ROOT = Path(__file__).parent.parent / "artifacts" / "smoke_test"

# Set to True via --offline to require pre-cached HuggingFace models/datasets.
# Default False: models and GLUE are downloaded on first run.
OFFLINE = False

CONFIGS = [
    {
        "arch": "encoder-only",
        "model_id": "bert-base-uncased",
        "cls_class": "BertForSequenceClassification",
        "tokenizer_class": "BertTokenizer",
        "lora_targets": ["query", "value"],
        "task_type": "SEQ_CLS",
    },
    {
        "arch": "decoder-only",
        "model_id": "Qwen/Qwen3-0.6B",
        "cls_class": "Qwen3ForSequenceClassification",
        "tokenizer_class": "AutoTokenizer",
        "lora_targets": ["q_proj", "v_proj"],
        "task_type": "SEQ_CLS",
    },
    {
        "arch": "encoder-decoder",
        "model_id": "google-t5/t5-base",
        "cls_class": "T5EncoderModel",
        "tokenizer_class": "T5Tokenizer",
        "lora_targets": ["q", "v"],
        "task_type": "FEATURE_EXTRACTION",
    },
]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sst2_subset(tokenizer, size, seed, max_length):
    ds = load_dataset("glue", "sst2", split="train")
    ds = ds.shuffle(seed=seed).select(range(size))
    texts = list(ds["sentence"])
    labels = list(ds["label"])
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt")
    return enc, torch.tensor(labels)


def run_smoke(config):
    arch = config["arch"]
    model_id = config["model_id"]
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {arch} ({model_id})")
    print(f"{'='*60}")

    gate_results = {}
    set_all_seeds(SEED)

    # --- Load tokenizer ---
    if config["tokenizer_class"] == "BertTokenizer":
        tokenizer = BertTokenizer.from_pretrained(model_id, local_files_only=OFFLINE)
    elif config["tokenizer_class"] == "AutoTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=OFFLINE)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_id, local_files_only=OFFLINE)

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    if config["cls_class"] == "BertForSequenceClassification":
        model = BertForSequenceClassification.from_pretrained(
            model_id, num_labels=2, local_files_only=OFFLINE)
    elif config["cls_class"] == "Qwen3ForSequenceClassification":
        model = Qwen3ForSequenceClassification.from_pretrained(
            model_id, num_labels=2, torch_dtype=torch.bfloat16,
            local_files_only=OFFLINE)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # T5 encoder + manual classification head
        model = T5EncoderModel.from_pretrained(model_id, local_files_only=OFFLINE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=config["lora_targets"],
        task_type=config["task_type"],
    )
    model = get_peft_model(model, lora_config)

    # For T5, add a classification head manually
    cls_head = None
    if config["cls_class"] == "T5EncoderModel":
        hidden_size = model.base_model.model.config.d_model
        cls_head = torch.nn.Linear(hidden_size, 2).to(device)

    model = model.to(device)
    if cls_head:
        cls_head = cls_head.to(device)

    # --- Gate 2: Trainable params ---
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if cls_head:
        trainable_params += sum(p.numel() for p in cls_head.parameters())
        total_params += sum(p.numel() for p in cls_head.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.3f}%)")
    gate_results["trainable_params_match"] = trainable_params > 0 and trainable_params < total_params * 0.05

    # --- Load data ---
    enc, labels = load_sst2_subset(tokenizer, SAMPLE_SIZE, SEED, MAX_LENGTH)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = labels.to(device)

    # --- Gate 1: Forward/loss/backward ---
    model.train()
    if cls_head:
        cls_head.train()

    optimizer_params = list(model.parameters())
    if cls_head:
        optimizer_params += list(cls_head.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in optimizer_params if p.requires_grad], lr=LR)

    try:
        if config["cls_class"] == "T5EncoderModel":
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (B, seq_len, hidden)
            # Mean pool with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            logits = cls_head(pooled)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        gate_results["forward_loss_backward"] = True
        print(f"  Forward/loss/backward: PASS (loss={loss.item():.4f})")
    except Exception as e:
        gate_results["forward_loss_backward"] = False
        print(f"  Forward/loss/backward: FAIL ({e})")
        return gate_results

    # --- Gate 3: Gradients only on intended modules ---
    # After one step, check which params have been updated
    lora_grad_ok = True
    base_grad_leak = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora" not in name.lower() and "classifier" not in name.lower() and "score" not in name.lower():
                # This shouldn't have grad unless it's a LoRA param
                base_grad_leak = True
                print(f"    WARNING: unexpected trainable param: {name}")

    gate_results["gradient_position"] = not base_grad_leak
    print(f"  Gradient position (LoRA-only): {'PASS' if not base_grad_leak else 'FAIL'}")

    # --- Gate 4: Finite loss (train a few more steps) ---
    losses = [loss.item()]
    for step in range(5):
        if config["cls_class"] == "T5EncoderModel":
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            logits = cls_head(pooled)
            loss = loss_fn(logits, labels)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    all_finite = all(np.isfinite(l) for l in losses)
    gate_results["finite_loss"] = all_finite
    print(f"  Finite loss (6 steps): {'PASS' if all_finite else 'FAIL'} (losses={[f'{l:.4f}' for l in losses]})")

    # --- Gate 5: Save metrics/config ---
    artifact_dir = ARTIFACT_ROOT / f"{arch}__{TASK}__n{SAMPLE_SIZE}__s{SEED}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    preds = logits.argmax(dim=-1).cpu().numpy()
    accuracy = (preds == labels.cpu().numpy()).mean()

    metrics = {
        "accuracy": float(accuracy),
        "final_loss": float(losses[-1]),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "arch": arch,
        "model_id": model_id,
        "losses": losses,
    }
    config_out = {
        "arch": arch,
        "model_id": model_id,
        "method": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_targets": config["lora_targets"],
        "task": TASK,
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "lr": LR,
        "epochs_run": "smoke(6 steps)",
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "dtype": "bfloat16" if config["cls_class"] == "Qwen3ForSequenceClassification" else "float32",
    }

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(artifact_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    saved_ok = (artifact_dir / "metrics.json").exists() and (artifact_dir / "config.json").exists()
    gate_results["save_artifacts"] = saved_ok
    print(f"  Save metrics/config: {'PASS' if saved_ok else 'FAIL'} ({artifact_dir})")

    # --- Gate 6: No dtype/device/shape errors ---
    # If we got here without exception, this passes
    gate_results["no_dtype_device_shape_error"] = True
    print(f"  No dtype/device/shape errors: PASS")

    # Record accuracy (not a gate)
    print(f"  [INFO] Accuracy (not gated): {accuracy:.4f}")

    # Cleanup
    del model, optimizer, input_ids, attention_mask, labels
    if cls_head:
        del cls_head
    gc.collect()
    torch.cuda.empty_cache()

    return gate_results


def main():
    global OFFLINE
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Require pre-cached HuggingFace models/datasets (no download).")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    OFFLINE = args.offline
    if OFFLINE:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    print("Cross-Architecture PEFT — Phase 1B Smoke Test")
    print(f"Config: 3 architectures x LoRA x SST-2 x size={SAMPLE_SIZE} x seed={SEED}")
    print(f"Artifacts: {ARTIFACT_ROOT}")
    print()

    all_results = {}
    all_pass = True

    for config in CONFIGS:
        results = run_smoke(config)
        all_results[config["arch"]] = results
        arch_pass = all(results.values())
        if not arch_pass:
            all_pass = False
        print(f"\n  >>> {config['arch']}: {'ALL GATES PASS' if arch_pass else 'SOME GATES FAILED'}")

    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL 3 ARCHITECTURES PASS' if all_pass else 'SOME FAILURES'}")
    print(f"{'='*60}")

    # Save summary
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_ROOT / "smoke_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
