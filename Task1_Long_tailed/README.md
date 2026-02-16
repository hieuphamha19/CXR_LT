# Task 1: Long-Tailed Chest X-Ray Classification

This repository contains the implementation for long-tailed multi-label chest X-ray classification using ConvNeXtV2 models with advanced techniques for handling class imbalance.

## Overview

This project addresses the challenge of long-tailed distribution in medical image classification, where some diseases are much more common than others. We implement multiple training strategies and model architectures to improve performance on rare (tail) classes.

## Features

- **Multiple Model Architectures**:
  - Standard classification head with BatchNorm and Dropout
  - CSRA (Class-Specific Residual Attention) head for better spatial attention
  
- **Advanced Loss Functions**:
  - Weighted BCE Loss
  - Focal Loss
  - Class-Balanced Focal Loss
  - Asymmetric Loss
  - LDAM (Label-Distribution-Aware Margin) Loss
  - DRW (Deferred Re-Weighting) Loss
  - Distribution-Balanced Loss with Class-Aware Sampling

- **Test-Time Augmentation (TTA)**:
  - Multiple augmentation strategies (flip, rotation, scaling, brightness)
  - Configurable merge modes (mean, max, geometric mean, median, logit mean)
  
- **Ensemble Methods**:
  - Weighted ensemble of multiple models
  - Combined with TTA for robust predictions

## File Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── utils.py                     # Utility functions (seed setting)
├── losses.py                    # Loss function implementations
├── tta.py                       # Test-time augmentation module
├── train_standard_head.py       # Training with standard classification head
├── train_csra_head.py          # Training with CSRA head + DB Loss + CAS
└── ensemble_inference.py        # Ensemble inference with TTA support
```

## Installation

```bash
# Create a conda environment
conda create -n cxr_lt python=3.10
conda activate cxr_lt

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Training with Standard Head

```bash
# Single GPU training
python train_standard_head.py

# Multi-GPU training with DDP
torchrun --nproc_per_node=4 train_standard_head.py
```

**Key Features**:
- Two-phase training: head warmup → full fine-tuning
- Differential learning rates for backbone and head
- EMA (Exponential Moving Average) for stable training
- Logit Adjustment for class imbalance
- Asymmetric Loss for multi-label classification

### 2. Training with CSRA Head

```bash
# Multi-GPU training
torchrun --nproc_per_node=4 train_csra_head.py
```

**Key Features**:
- CSRA (Class-Specific Residual Attention) head
- Distribution-Balanced Loss
- Class-Aware Sampling (CAS) for tail class boost
- Advanced augmentation (CLAHE, GaussNoise, CoarseDropout)

### 3. Ensemble Inference

```bash
python ensemble_inference.py \
  --checkpoints model1.pth model2.pth \
  --weights 1.0 1.5 \
  --test_csv test.csv \
  --image_dir /path/to/images \
  --use_tta \
  --tta_config flip_scale \
  --tta_merge_mode mean \
  --ensemble_mode weighted_mean \
  --postprocess normal_gating \
  --normal_alpha 0.5 \
  --batch_size 32 \
  --output submission.csv
```

**Arguments**:
- `--checkpoints`: Paths to model checkpoints
- `--weights`: Weights for each model in ensemble
- `--use_tta`: Enable test-time augmentation
- `--tta_config`: TTA strategy (`none`, `flip`, `light`, `medium`, `heavy`, `flip_scale`)
- `--tta_merge_mode`: How to merge TTA predictions (`mean`, `max`, `gmean`, `median`)
- `--postprocess`: Post-processing mode (`none`, `normal_gating`)

## Model Architectures

### Standard Head

```
ConvNeXtV2 Backbone
    ↓
BatchNorm1d
    ↓
Dropout(0.3)
    ↓
Linear(backbone_features → 512)
    ↓
ReLU
    ↓
BatchNorm1d
    ↓
Dropout(0.3)
    ↓
Linear(512 → num_classes)
```

### CSRA Head

```
ConvNeXtV2 Backbone (without global pooling)
    ↓
BatchNorm2d
    ↓
CSRA Module:
  ├─ Global Average Pooling → Linear → logit_gap
  └─ Spatial Attention:
       Conv2d(1×1) → Softmax → Weighted Features → logit_csra
    ↓
Output: logit_gap + λ × logit_csra
```

## Loss Functions

### 1. Asymmetric Loss
Addresses the imbalance between positive and negative samples in multi-label classification.

### 2. Class-Balanced Focal Loss
Combines focal loss with effective number of samples for class balancing.

### 3. LDAM Loss
Assigns larger margins to tail classes to improve their separability.

### 4. Distribution-Balanced Loss
Comprehensive loss with:
- Re-balanced weighting via effective number
- Negative-tolerant regularization
- Logit margin for tail classes

## Test-Time Augmentation

Available TTA configurations:

- **none**: No augmentation (identity only)
- **flip**: Horizontal flip
- **light**: Flip + rotation (±5°)
- **medium**: Light + scaling (0.9×, 1.1×)
- **heavy**: Medium + brightness adjustment
- **flip_scale**: Flip + scaling combinations (recommended)

## Post-Processing

### Normal Gating
Reduces abnormal class predictions based on normal class confidence:

```
p_abnormal = p_abnormal × (1 - p_normal)^α
```

where `α` controls the gating strength (default: 0.5).

## Training Tips

1. **Two-Phase Training**: Always start with head-only warmup to avoid catastrophic forgetting
2. **Differential Learning Rates**: Use lower LR for backbone (1e-5) and higher for head (1e-4)
3. **EMA**: Helps stabilize training and often gives better final performance
4. **Class-Aware Sampling**: Effective for extreme imbalance (use with CSRA model)
5. **Gradient Clipping**: Essential for stable training with complex losses

## Performance Optimization

- **Mixed Precision Training**: Enabled by default with `torch.cuda.amp`
- **Channels Last Memory Format**: Faster GPU operations
- **Persistent Workers**: Reduces dataloader overhead
- **Prefetch Factor**: Optimized for 4 workers per GPU

## Dataset Format

### Training CSV
```csv
ImageID,StudyDate_DICOM,PatientID,Normal,cardiomegaly,...
image1.png,20200101,P001,1,0,...
image2.png,20200102,P002,0,1,...
```

### Test CSV
```csv
ImageID
test_image1.png
test_image2.png
```


## License

This project is for research purposes only.

## Contact

For questions or issues, please open an issue on GitHub.
