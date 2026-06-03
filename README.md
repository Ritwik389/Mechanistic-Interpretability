# Mechanistic Interpretability of Domain-Specific Fine-Tuning in LLMs

> Tracking Sparse Autoencoder Feature Shifts in Pythia-160M Before and After Code Domain Adaptation  
> Delhi Technological University — Ritwik Jain

---

## Overview

This repository contains the full reproducible pipeline for a mechanistic interpretability study examining how domain-specific fine-tuning reshapes the internal feature representations of a large language model.

We train a **TopK Sparse Autoencoder (SAE)** on the residual stream activations of **Layer 8 of Pythia-160M**, fine-tune the base model on a Python code corpus, train a second SAE on the fine-tuned model under identical conditions, and then systematically compare the two learned feature sets to characterize expected and collateral feature drift.

---

## Repository Structure

> The codebase is currently a single notebook and will be refactored into the structure below. Module names are final; file locations may shift slightly during conversion.

```
.
├── notebooks/
│   └── full_pipeline.ipynb          # End-to-end notebook (current state)
│
├── src/                             # (in progress — converted from notebook)
│   ├── collect_activations.py       # Hook into blocks.8.hook_resid_post, stream tokens
│   ├── train_sae.py                 # TopK SAE training via SAELens
│   ├── finetune.py                  # Fine-tune Pythia-160M on Python code corpus
│   ├── compare_features.py          # Cosine similarity matrix + Hungarian matching
│   └── visualize.py                 # Histograms, heatmaps, monosemanticity plots
│
├── configs/
│   └── sae_config.yaml              # SAE hyperparameters (d_sae, k, lr, etc.)
│
├── requirements.txt
└── README.md
```

---

## Method Summary

| Stage | Details |
|---|---|
| **Base model** | `EleutherAI/pythia-160m` |
| **Layer tapped** | Layer 8 — `blocks.8.hook_resid_post` |
| **SAE architecture** | TopK, `k=32`, `d_sae=3072` (4× expansion) |
| **SAE training tokens** | 50M tokens from `Skylion007/openwebtext` |
| **Fine-tuning corpus** | `flytech/python-codes-25k` (5000 samples, 90/10 split) |
| **Fine-tuning objective** | Causal language modelling, 1 epoch |
| **Fine-tuning LR** | `5e-6` (cosine schedule, 200 warmup steps) |
| **Comparison method** | Pairwise cosine similarity + Hungarian algorithm matching |
| **Hardware** | NVIDIA T4 16GB (Google Colab Free Tier) |

---

## Pretrained Weights

Both SAE checkpoints (base and fine-tuned) and the fine-tuned Pythia-160M weights are available on Google Drive. Download them before running the comparison pipeline.

> **[Python Finetuned](https://drive.google.com/drive/folders/1GwD-Bww7ta7GyOzrIOqKRL4wqj8eRt5c?usp=share_link)**
> 
> **[SAE Finetuned](https://drive.google.com/drive/folders/1dG9EmK4SOCfHNfvF9gBaVPd9nFLutTTa?usp=share_link)**
> 
> **[SAE_Base](https://drive.google.com/drive/folders/1nkJoasDfC07ya0bc4Jo-gc_6Qigcclzj?usp=sharing)**

Place downloaded files as follows:

```
weights/
├── sae_base.pt                  # SAE trained on base Pythia-160M
├── sae_finetuned.pt             # SAE trained on fine-tuned Pythia-160M
└── pythia_160m_python_ft/       # Fine-tuned HuggingFace model directory
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

> **Important:** The fine-tuned HuggingFace weights must be loaded via `HookedTransformer.from_pretrained(..., hf_model=raw_hf_model)` to prevent SAELens from silently defaulting to the base model weights. See `src/collect_activations.py` for the correct loading pattern.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

Python 3.10+ recommended. A GPU with at least 12GB VRAM is required for SAE training; the comparison pipeline can run on CPU.

```bash
pip install -r requirements.txt
```

Key dependencies:

```
torch
transformer_lens
saelens
transformers
datasets
wandb
scipy
numpy
matplotlib
seaborn
```

### 3. (Optional) Log in to Weights & Biases

Training runs log to W&B. Skip this if you don't want experiment tracking.

```bash
wandb login
```

---

## Reproducing the Full Pipeline

You can either run the end-to-end notebook or execute the individual stages below once the refactor is complete.

### Stage 1 — Collect base model activations & train base SAE

```bash
python src/collect_activations.py --model base --output_dir activations/base/
python src/train_sae.py --activations_dir activations/base/ --output weights/sae_base.pt
```

### Stage 2 — Fine-tune Pythia-160M

```bash
python src/finetune.py \
  --dataset flytech/python-codes-25k \
  --num_samples 5000 \
  --output_dir weights/pythia_160m_python_ft/
```

### Stage 3 — Collect fine-tuned model activations & train fine-tuned SAE

```bash
python src/collect_activations.py --model finetuned \
  --hf_weights_dir weights/pythia_160m_python_ft/ \
  --output_dir activations/finetuned/
python src/train_sae.py --activations_dir activations/finetuned/ --output weights/sae_finetuned.pt
```

### Stage 4 — Compare feature sets

```bash
python src/compare_features.py \
  --base_sae weights/sae_base.pt \
  --ft_sae weights/sae_finetuned.pt \
  --output_dir results/
```

### Stage 5 — Visualize

```bash
python src/visualize.py --results_dir results/
```

Or to skip straight to analysis using the provided weights:

```bash
python src/compare_features.py \
  --base_sae weights/sae_base.pt \
  --ft_sae weights/sae_finetuned.pt \
  --output_dir results/

python src/visualize.py --results_dir results/
```

---

## Key Implementation Notes

- Both SAEs use **strictly identical hyperparameters, corpus, tokenized sequences, and layer hooks**. The only variable is the model weights. This is essential to attribute observed feature differences to fine-tuning rather than SAE training variance.
- SAE training is **stochastic** — two runs with identical hyperparameters produce rotated feature spaces. Some measured shift is therefore noise. The comparison pipeline accounts for this via the Hungarian matching threshold.
- Fine-tuning stability required: `max_grad_norm=0.5`, cosine LR schedule, 200 warmup steps, and a `NaNWatchdog` callback. These are already set as defaults in `finetune.py`.

---

## References

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — Elhage et al., 2021
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemanticity/index.html) — Bricken et al., 2023
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) — Templeton et al., 2024
- [SAELens](https://github.com/jbloomAus/SAELens) — Bloom, 2024
- [A Comprehensive Mechanistic Interpretability Explainer](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/comprehensive-mechanistic-interpretability-explainer) — Nanda, 2022
