# Task 1: Long-Tailed Chest X-Ray Classification

Implementation for **Task 1** of the [CXR-LT 2026 Challenge](https://cxr-lt.github.io/CXR-LT-2026/) — long-tailed multi-label classification of 30 chest X-ray findings.

> 🏆 **Top-1** submission using ConvNeXtV2-Base + CSRA + DB-CAS strategy.
> Pre-trained weights: [hieuphamha/cxrlt2026-task1-convnextv2](https://huggingface.co/hieuphamha/cxrlt2026-task1-convnextv2)

---

## File Structure

```
Task1_Long_tailed/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── model.py               # Model definition + timm registration (HF-compatible)
├── train_csra_head.py     # Training: ConvNeXtV2 + CSRA + DB-CAS
├── inference.py           # Inference with TTA
├── losses.py              # Loss functions (AsymmetricLoss, DB Loss, LDAM, ...)
├── tta.py                 # Test-time augmentation
└── utils.py               # Utility functions
```

---

## Model Architecture

```
Stage 1 — Pre-train on MIMIC-CXR (14 classes)  [FC/MLP head]
  ConvNeXtV2-Base (ImageNet-22k)
  ├── Phase 1: head-only warm-up  (LR 1e-3, 3 epochs)
  └── Phase 2: full fine-tune    (backbone LR 1e-5 / head LR 1e-4)
      Loss: AsymmetricLoss + LogitAdjustment, EMA decay 0.9999

Stage 2 — DB-CAS fine-tune on PadChest (30 classes)  [FC → CSRA head]
  Backbone from Stage 1; FC head replaced with CSRA; new head initialized
  ├── Double-Balance (DB) sampling
  └── Class Activation Spatial (CAS) attention training
      Loss: AsymmetricLoss, EMA decay 0.9999
```

**Architecture (final model):**
```
ConvNeXtV2-Base  (global_pool="", drop_path_rate=0.2)  → (B, 1024, H, W)
  └── BatchNorm2d(1024)
  └── CSRA head  (λ=0.1)
       ├── GAP branch : Linear(1024→30)               → logit_gap
       └── Attention  : Conv2d(1024→30, 1×1) → Softmax → logit_csra
       └── output     : logit_gap + 0.1 × logit_csra
```

---

## Installation

```bash
conda create -n cxr_lt python=3.10
conda activate cxr_lt
pip install -r requirements.txt
```

---

## Usage

### Quick start — load pretrained weights from HuggingFace

```python
from huggingface_hub import hf_hub_download
import importlib.util, sys, timm

# Register model (one-time)
path = hf_hub_download("hieuphamha/cxrlt2026-task1-convnextv2", "model.py")
spec = importlib.util.spec_from_file_location("cxrlt", path)
mod  = importlib.util.module_from_spec(spec)
sys.modules["cxrlt"] = mod
spec.loader.exec_module(mod)

model = timm.create_model("cxrlt2026_task1_csra_dbcas", pretrained=True)
model.eval()
```

### Training (Stage 2 — DB-CAS on PadChest)

```bash
# Multi-GPU training with DDP
torchrun --nproc_per_node=4 train_csra_head.py
```

**Key training features:**
- CSRA (Class-Specific Residual Attention) head with spatial attention
- Distribution-Balanced (DB) Loss + Class-Aware Sampling (CAS)
- Two-phase training: head warm-up → full fine-tune with differential LR
- EMA (Exponential Moving Average) for stable training
- Mixed precision (AMP) + gradient clipping

### Inference

```bash
python inference.py \
  --checkpoints /path/to/best_padchest_db_cas.pth \
  --test_csv test.csv \
  --image_dir /path/to/images \
  --use_tta \
  --tta_config medium \
  --tta_merge_mode logit_mean \
  --postprocess normal_gating \
  --normal_alpha 0.5 \
  --batch_size 32 \
  --output submission.csv
```

---

## Loss Functions

| Loss | Description |
|---|---|
| `AsymmetricLoss` | Asymmetric focusing for positive/negative imbalance |
| `DistributionBalancedLoss` | Re-balanced weighting + negative-tolerant regularization |
| `LDAMLoss` | Larger margins for tail classes |
| `FocalLoss` | Hard example mining |

---

## Test-Time Augmentation

| Config | Transforms |
|---|---|
| `flip` | Horizontal flip |
| `medium` | Flip + scale (0.9×, 1.1×) |
| `flip_scale` | Flip + scale combinations |

---

## Post-Processing

**Normal Gating** — suppresses abnormal findings when "Normal" score is high:
```
p_abnormal = p_abnormal × (1 - α × p_normal),   α = 0.5
```

---

## Hardware Requirements

- **Training**: 4× GPU, 16GB+ VRAM each (e.g. A100, V100)
- **Inference**: 1× GPU, 8GB+ VRAM

---

## Citation

```bibtex
@article{Pham2026HandlingSS,
  title   = {Handling Supervision Scarcity in Chest X-ray Classification:
             Long-Tailed and Zero-Shot Learning},
  author  = {Ha-Hieu Pham and Hai-Dang Nguyen and Thanh-Huy Nguyen and
             Min Xu and Ulas Bagci and Trung-Nghia Le and Huy-Hieu Pham},
  journal = {ArXiv},
  year    = {2026},
  volume  = {abs/2602.13430},
  url     = {https://arxiv.org/abs/2602.13430}
}
```

---

## License

Research purposes only.
