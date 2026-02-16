# CXR Long-Tailed Recognition Challenge 2026

🔗 Challenge website: https://cxr-lt.github.io/CXR-LT-2026/

This repository contains solutions for the CXR Long-Tailed Recognition Challenge 2026, addressing two critical problems in medical image analysis:

1. **Task 1**: Long-tailed multi-label classification (30 classes)
2. **Task 2**: Zero-shot out-of-distribution detection (6 OOD classes)

## Repository Structure

```
github/
├── README.md                    # This file
├── Task1_Long_tailed/          # Long-tailed classification (30 classes)
│   ├── README.md               # Detailed documentation
│   ├── requirements.txt        # Dependencies
│   ├── ensemble_inference.py   # Ensemble inference with TTA
│   ├── losses.py               # Loss functions for long-tail
│   ├── train_standard_head.py  # Training with standard head
│   ├── train_csra_head.py      # Training with CSRA head
│   ├── tta.py                  # Test-time augmentation
│   └── utils.py                # Utility functions
│
└── Task2_Zero_shot/            # Zero-shot OOD detection (6 classes)
    ├── README.md               # Detailed documentation
    ├── requirements.txt        # Dependencies
    ├── zero_shot_ood_detection.py  # Main inference script
    ├── config.py               # Configuration
    ├── tta.py                  # Test-time augmentation
    └── utils.py                # Utility functions
```

## Quick Start

### Task 1: Long-Tailed Classification

Train models to classify 30 chest X-ray findings with extreme class imbalance.

```bash
cd Task1_Long_tailed

# Install dependencies
pip install -r requirements.txt

# Train with standard head
torchrun --nproc_per_node=4 train_standard_head.py

# Train with CSRA head (better for long-tail)
torchrun --nproc_per_node=4 train_csra_head.py

# Ensemble inference with TTA
python ensemble_inference.py \
  --checkpoints model1.pth model2.pth \
  --weights 1.0 1.5 \
  --test_csv test.csv \
  --image_dir /path/to/images \
  --use_tta \
  --output submission.csv
```

**Key Features**:
- Multiple loss functions (Focal, LDAM, DRW, Asymmetric, DB Loss)
- Class-Aware Sampling (CAS) for tail classes
- CSRA (Class-Specific Residual Attention) head
- Test-Time Augmentation (TTA)
- Model ensemble

### Task 2: Zero-Shot OOD Detection

Detect 6 rare findings not seen during training using WhyXrayCLIP.

```bash
cd Task2_Zero_shot

# Install dependencies
pip install -r requirements.txt

# Run inference with WhyXrayCLIP
python zero_shot_ood_detection.py \
  --test_csv test.csv \
  --image_dir /path/to/images \
  --output submission.csv

# Or use default paths from config
python zero_shot_ood_detection.py
```

**Key Features**:
- Zero-shot learning (no training required)
- WhyXrayCLIP (CXR-specialized CLIP from UPenn)
- Enhanced prompt engineering (13 prompts per class)
- Medical terminology and radiological findings
- Simple & clean API

## Statistics

### Task 1: Long-Tailed Classification
- **Files**: 8 Python files + README + requirements
- **Lines of Code**: 3,039 lines
- **Models**: ConvNeXtV2-Base
- **Techniques**: LDAM, DRW, CSRA, DB Loss, CAS, TTA

### Task 2: Zero-Shot OOD Detection
- **Files**: 4 Python files + README + requirements + config
- **Lines of Code**: 1,242 lines
- **Model**: WhyXrayCLIP (CXR-specialized CLIP)
- **Techniques**: Prompt ensemble (13 prompts/class), multi-template prompts

**Total**: 4,281 lines of well-documented code

## Key Innovations

### Task 1
1. **Distribution-Balanced Loss + Class-Aware Sampling**: Best combination for extreme long-tail
2. **CSRA Head**: Spatial attention mechanism for better localization
3. **Two-Phase Training**: Head warmup → full fine-tuning prevents catastrophic forgetting
4. **Ensemble + TTA**: Combines multiple models with test-time augmentation

### Task 2
1. **WhyXrayCLIP**: CXR-specialized CLIP model from UPenn for best performance
2. **Enhanced Prompt Engineering**: 13 diverse prompts per class covering clinical, radiological, anatomical, and medical terminology
3. **Zero-Shot Learning**: No training data required for new classes
4. **Clean & Simple**: Single model focus, easy to use and maintain

## Performance Tips

### Task 1
- Use CSRA head with DB Loss + CAS for best results
- Ensemble at least 2 models (standard + CSRA)
- Enable TTA with `flip_scale` configuration
- Use `normal_gating` post-processing

### Task 2
- WhyXrayCLIP is optimized for chest X-rays
- Batch size 32-64 for optimal throughput
- 13 prompts per class provide robust predictions
- Simple API with minimal configuration

## Hardware Requirements

### Task 1 Training
- **GPU**: 4x GPUs with 16GB+ VRAM each (e.g., V100, A100)
- **RAM**: 64GB+ system memory
- **Storage**: Fast SSD for data loading
- **Training Time**: ~12-24 hours per model

### Task 2 Inference
- **GPU**: 1x GPU with 8GB+ VRAM (e.g., RTX 3070, V100)
- **RAM**: 16GB+ system memory
- **Model**: WhyXrayCLIP (ViT-L-14)
- **Inference Time**: ~5-10 minutes for 1000 images

## Dependencies

### Common
- Python 3.10+
- PyTorch 2.0+
- OpenCV
- NumPy, Pandas
- tqdm

### Task 1 Specific
- timm (PyTorch Image Models)
- albumentations (augmentation)
- scikit-learn (metrics)

### Task 2 Specific
- open-clip-torch (for WhyXrayCLIP)



## License

This project is for research purposes only.

## Acknowledgments

- **Task 1**: Inspired by LDAM, DRW, CSRA, and DB Loss papers
- **Task 2**: Built on WhyXrayCLIP from University of Pennsylvania
- Thanks to the medical imaging community for domain knowledge
- Special thanks to the WhyXrayCLIP team for the CXR-specialized model

## Contact

For questions or issues, please open an issue on GitHub.
