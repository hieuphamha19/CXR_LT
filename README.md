# CXR Long-Tailed Recognition Challenge 2026

🔗 Challenge website: https://cxr-lt.github.io/CXR-LT-2026/

This repository contains the **Top-1** solutions for the CXR Long-Tailed Recognition Challenge 2026:

1. **Task 1**: Long-tailed multi-label classification (30 classes)
2. **Task 2**: Zero-shot out-of-distribution detection (6 OOD classes)

📄 Paper: [arXiv 2602.13430](https://arxiv.org/abs/2602.13430)

---

## Repository Structure

```
├── README.md                    # This file
├── Task1_Long_tailed/           # Long-tailed classification (30 classes)  🏆 Top-1
│   ├── README.md                # Detailed documentation
│   ├── requirements.txt         # Dependencies
│   ├── model.py                 # Model definition (timm + HuggingFace compatible)
│   ├── train_csra_head.py       # Training: ConvNeXtV2 + CSRA + DB-CAS
│   ├── inference.py             # Inference with TTA
│   ├── losses.py                # Loss functions for long-tail
│   ├── tta.py                   # Test-time augmentation
│   └── utils.py                 # Utility functions
│
└── Task2_Zero_shot/             # Zero-shot OOD detection (6 classes)  🏆 Top-1
    ├── README.md                # Detailed documentation
    ├── requirements.txt         # Dependencies
    ├── zero_shot_ood_detection.py  # Main inference script
    ├── config.py                # Configuration
    ├── tta.py                   # Test-time augmentation
    └── utils.py                 # Utility functions
```

---

## Quick Start

### Task 1: Long-Tailed Classification

```bash
cd Task1_Long_tailed
pip install -r requirements.txt

# Load pretrained model from HuggingFace (recommended)
python -c "
from huggingface_hub import hf_hub_download
import importlib.util, sys, timm

path = hf_hub_download('hieuphamha/cxrlt2026-task1-convnextv2', 'model.py')
spec = importlib.util.spec_from_file_location('cxrlt', path)
mod  = importlib.util.module_from_spec(spec)
sys.modules['cxrlt'] = mod
spec.loader.exec_module(mod)

model = timm.create_model('cxrlt2026_task1_csra_dbcas', pretrained=True)
print('Model loaded:', sum(p.numel() for p in model.parameters()), 'params')
"

# Train from scratch (Stage 2 DB-CAS fine-tune)
torchrun --nproc_per_node=4 train_csra_head.py

# Inference with TTA
python inference.py \
  --checkpoints /path/to/best_padchest_db_cas.pth \
  --test_csv test.csv \
  --image_dir /path/to/images \
  --use_tta \
  --tta_config medium \
  --postprocess normal_gating \
  --output submission.csv
```

**Key Features:**
- ConvNeXtV2-Base + CSRA (Class-Specific Residual Attention) head
- DB-CAS: Distribution-Balanced Loss + Class-Aware Sampling
- Two-stage training: MIMIC-CXR (14 cls, FC head) → PadChest (30 cls, CSRA head)
- Test-Time Augmentation (TTA) + Normal Gating post-processing

### Task 2: Zero-Shot OOD Detection

```bash
cd Task2_Zero_shot
pip install -r requirements.txt

# Run inference with WhyXrayCLIP
python zero_shot_ood_detection.py \
  --test_csv test.csv \
  --image_dir /path/to/images \
  --output submission.csv
```

**Key Features:**
- Zero-shot learning (no training required)
- WhyXrayCLIP (CXR-specialized CLIP from UPenn)
- Enhanced prompt engineering (13 prompts per class)

---

## Pretrained Models

| Task | Model | HuggingFace |
|---|---|---|
| Task 1 | ConvNeXtV2-Base + CSRA (DB-CAS) | [hieuphamha/cxrlt2026-task1-convnextv2](https://huggingface.co/hieuphamha/cxrlt2026-task1-convnextv2) |

---

## Key Innovations

### Task 1
1. **DB-CAS**: Distribution-Balanced Loss + Class-Aware Sampling — best combination for extreme long-tail
2. **CSRA Head**: Spatial class-specific attention for better localization of rare findings
3. **Two-Stage Training**: MIMIC-CXR pre-train (FC head) → PadChest fine-tune (CSRA head)
4. **TTA + Normal Gating**: Robust inference with post-processing

### Task 2
1. **WhyXrayCLIP**: CXR-specialized CLIP from University of Pennsylvania
2. **Enhanced Prompt Engineering**: 13 diverse prompts per class (clinical, radiological, anatomical)
3. **Zero-Shot**: No training data required for new OOD classes

---

## Hardware Requirements

### Task 1 Training
- **GPU**: 4× GPU, 16GB+ VRAM each (e.g. A100, V100)
- **RAM**: 64GB+ system memory
- **Training Time**: ~12–24 hours per stage

### Task 2 Inference
- **GPU**: 1× GPU, 8GB+ VRAM
- **Model**: WhyXrayCLIP (ViT-L-14)
- **Inference Time**: ~5–10 min / 1000 images

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

This project is for research purposes only.
