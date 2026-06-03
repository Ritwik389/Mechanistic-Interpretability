# Mechanistic Interpretability of Domain-Specific Fine-Tuning in LLMs

> Tracking Sparse Autoencoder Feature Shifts in Pythia-160M Before and After Code Domain Adaptation  


---

## Overview

This repository contains the full reproducible pipeline for a mechanistic interpretability study examining how domain-specific fine-tuning reshapes the internal feature representations of a large language model.

We train a **TopK Sparse Autoencoder (SAE)** on the residual stream activations of **Layer 8 of Pythia-160M**, fine-tune the base model on a Python code corpus, train a second SAE on the fine-tuned model under identical conditions, and then systematically compare the two learned feature sets to characterize expected and collateral feature drift.

---

## Repository Structure

> The codebase is currently a single notebook and will be refactored 

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
| **Hardware** | NVIDIA T4 16GB (Google Colab Free Tier/Kaggle) |

---

## Pretrained Weights

Both SAE checkpoints (base and fine-tuned) and the fine-tuned Pythia-160M weights are available on Google Drive. Download them before running the comparison pipeline.

> **[Python Finetuned](https://drive.google.com/drive/folders/1GwD-Bww7ta7GyOzrIOqKRL4wqj8eRt5c?usp=share_link)**
> 
> **[SAE Finetuned](https://drive.google.com/drive/folders/1dG9EmK4SOCfHNfvF9gBaVPd9nFLutTTa?usp=share_link)**
> 
> **[SAE_Base](https://drive.google.com/drive/folders/1nkJoasDfC07ya0bc4Jo-gc_6Qigcclzj?usp=sharing)**



> **Important:** The fine-tuned HuggingFace weights must be loaded via `HookedTransformer.from_pretrained(..., hf_model=raw_hf_model)` to prevent SAELens from silently defaulting to the base model weights. See `src/collect_activations.py` for the correct loading pattern.

---

## Setup

### Install dependencies

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
```

### 3. (Optional) Log in to Weights & Biases

Training runs log to W&B. Skip this if you don't want experiment tracking.

```bash
wandb login
```

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
