# PEFT Across Transformer Backbone Families for Low-Resource Text Classification: An Empirical Comparison

> Yaowen Sun, Ning Wu, Xiaolei Sun

Code and data for a controlled comparison of three parameter-efficient fine-tuning (PEFT) methods — LoRA, BitFit, and (IA)³ — applied to encoder-only, decoder-only, and encoder-decoder Transformer backbones under identical low-resource text classification settings. Provided for reproducibility.

## Overview

PEFT methods are usually evaluated within a single architecture family, leaving open whether a recipe that works on one backbone transfers to another. This repository studies that question with a fully controlled grid: the same three PEFT methods are applied to three architecture families on four GLUE classification tasks, across four training-set sizes and three seeds — **432 runs total**. The focus is empirical guidance for method/backbone selection, not a new method.

## Repository Structure

```
.
├── README.md
├── LICENSE                       # MIT
├── requirements-analysis.txt     # Path A deps (no GPU)
├── requirements-experiment.txt   # Path B deps (GPU)
├── src/
│   ├── grid_runner.py            # full 432-run PEFT grid (resume-safe, validated)
│   ├── smoke_test.py             # pipeline correctness smoke test
│   ├── aggregate_results.py      # per-run artifacts -> data/results.csv + summary.json
│   ├── statistical_analysis.py   # ANOVA, paired tests, method agreement, collapse
│   ├── robustness_analysis.py    # Friedman tests, rank/regret, collapse sensitivity
│   └── generate_figures.py       # publication figures
├── data/
│   ├── results.csv               # 432-run aggregated metrics
│   ├── summary.json              # per-cell summaries
│   ├── statistical_analysis.json # ANOVA / paired / agreement / collapse output
│   └── robustness_analysis.json  # Friedman / rank / sensitivity output
└── figures/
    ├── fig1_heatmap.pdf          # mean accuracy by architecture x method
    ├── fig2_scaling.pdf          # accuracy vs sample size
    ├── fig3_collapse.pdf         # collapse rate by architecture x task
    └── fig4_rank_distribution.pdf# within-cell architecture rank distribution
```

## Experimental Setup

| Dimension | Values |
|---|---|
| Architectures | encoder-only (bert-base-uncased, 110M), decoder-only (Qwen/Qwen3-0.6B, 596M), encoder-decoder (google-t5/t5-base encoder, 110M) |
| PEFT methods | LoRA (r=8, α=16, q+v), BitFit (bias-only), (IA)³ |
| Tasks | SST-2, MRPC, QNLI, RTE (GLUE) |
| Sample sizes | 48, 96, 192, 384 |
| Seeds | 7, 13, 29 |
| Total | 3 × 3 × 4 × 4 × 3 = **432 runs** |

`data/results.csv` contains 432 data rows (433 lines including the header). Classification heads: BERT uses the [CLS] token; Qwen3 uses the last non-padding token (native `Qwen3ForSequenceClassification`); T5 uses attention-mask mean pooling over the encoder last hidden states (the decoder is not used).

## Hardware & Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX PRO 6000 (Blackwell), bf16 |
| Python | 3.11 (tested on 3.11.15) |
| PyTorch | 2.11.0 (CUDA 12.8 build) |
| transformers | 5.4.0 |
| peft | 0.18.1 |
| datasets | 4.8.4 |
| numpy / pandas / scipy / statsmodels / matplotlib | 2.4.4 / 3.0.1 / 1.17.1 / 0.14.6 / 3.10.8 |

Training: AdamW, lr = 2×10⁻⁴, 10 epochs, batch size 16, bfloat16, gradient clipping 1.0, 10% linear warmup, max sequence length 128.

## Key Results

- **Architecture affects PEFT behavior**: Friedman tests are significant in 8 of 12 method–task strata after Benjamini–Hochberg correction.
- **The decoder-only model ranks first in 67% of matched cells**, but this is confounded with its larger parameter count (596M vs ~110M). At matched scale, encoder-only and encoder-decoder backbones are comparable (mean rank 2.10 vs 2.40).
- **PEFT method selection does not transfer cleanly**: only 3 of 16 task–size cells agree on the best method across architectures, and this persists after excluding collapsed runs.
- **Collapse is architecture-dependent**: 78/432 runs (18.1%) fall below the majority baseline, concentrated on RTE and MRPC; the most vulnerable architecture differs by task.

## Requirements

Python 3.11. Two pinned dependency sets:

- **Path A (analysis only, no GPU)**: `pip install -r requirements-analysis.txt`
- **Path B (full experiment, GPU)**: `pip install -r requirements-experiment.txt` (install a CUDA build of PyTorch separately for your GPU).

### Reproducing

Path A — statistics and figures from the provided data:

```bash
python src/statistical_analysis.py
python src/robustness_analysis.py
python src/generate_figures.py
```

Path B — re-run the full grid from scratch (downloads models + GLUE on first run; add `--offline` to require pre-cached assets):

```bash
python src/grid_runner.py
python src/aggregate_results.py
python src/statistical_analysis.py
python src/robustness_analysis.py
python src/generate_figures.py
```

## Citation

```bibtex
@article{sun_cross_architecture_peft,
  title   = {PEFT Across Transformer Backbone Families for Low-Resource Text Classification: An Empirical Comparison},
  author  = {Sun, Yaowen and Wu, Ning and Sun, Xiaolei},
  year    = {2026}
}
```

## License

MIT License (see `LICENSE`).
