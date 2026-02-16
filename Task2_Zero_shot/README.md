# Task 2: Zero-Shot Out-of-Distribution (OOD) Detection

This repository contains the implementation for zero-shot detection of 6 out-of-distribution (OOD) chest X-ray findings using Vision-Language Models (VLMs).

## Overview

This project uses CLIP-based models to detect rare chest X-ray findings that were not present in the training data. By leveraging the power of vision-language pre-training, the model can identify new pathologies using only textual descriptions, without requiring any labeled training examples.

## OOD Classes (6 classes)

1. **Scoliosis** - Abnormal lateral curvature of the spine
2. **Osteopenia** - Decreased bone mineral density
3. **Bulla** - Large air-filled space in the lung
4. **Infarction** - Pulmonary tissue death due to ischemia
5. **Adenopathy** - Enlarged lymph nodes
6. **Goiter** - Enlarged thyroid gland

## Features

- **Zero-Shot Detection**: No training required, uses pre-trained WhyXrayCLIP model
- **WhyXrayCLIP**: CXR-specialized CLIP from UPenn - best performance on chest X-rays
- **Enhanced Prompt Engineering**:
  - Multi-template prompts (13 prompts per class)
  - Clinical, radiological, anatomical, and medical terminology perspectives
  - Prompt ensemble for robust predictions
- **Optimized for Medical Imaging**: Specialized preprocessing for chest X-rays
- **Simple & Clean**: Single model, no complex configurations

## File Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── zero_shot_ood_detection.py    # Main inference script (709 lines)
├── tta.py                         # Test-time augmentation utilities (315 lines)
├── utils.py                       # Helper functions (16 lines)
└── config.py                      # Configuration file (46 lines)
```

**Total: 1,086 lines of code**

## Installation

```bash
# Create conda environment
conda create -n cxr_ood python=3.10
conda activate cxr_ood

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python zero_shot_ood_detection.py \
  --test_csv /path/to/test.csv \
  --image_dir /path/to/images \
  --output submission.csv
```

### With Custom Batch Size

```bash
python zero_shot_ood_detection.py \
  --test_csv /path/to/test.csv \
  --image_dir /path/to/images \
  --batch_size 64 \
  --output submission.csv
```

### Using Default Paths from Config

If you have `config.py` set up with correct paths:

```bash
python zero_shot_ood_detection.py
```

### Command-Line Arguments

```
--test_csv              Path to test CSV file (default: from config)
--image_dir             Path to image directory (default: from config)
--batch_size            Batch size for inference (default: from config)
--output                Output CSV file path (default: submission_task2_whyxrayclip.csv)
```

## Why WhyXrayCLIP?

### WhyXrayCLIP
- **Source**: University of Pennsylvania
- **Specialty**: Trained specifically on chest X-rays with radiology reports
- **Architecture**: ViT-L-14 (Vision Transformer Large)
- **Advantages**: 
  - Best performance on CXR tasks
  - Understands medical terminology
  - Pre-trained on large-scale CXR dataset
  - Robust to various imaging conditions
- **Paper**: "Towards Explainable Zero-Shot Chest X-Ray Classification" (2023)

## Prompt Engineering Strategy

Our prompts are designed using multiple perspectives to maximize detection accuracy:

### 1. Clinical/Descriptive Prompts
Simple, direct descriptions of the finding:
- "a chest x-ray showing scoliosis"
- "chest radiograph demonstrating scoliosis with spinal curvature"

### 2. Radiological Findings Prompts
Specific radiological signs and patterns:
- "abnormal lateral curvature of the spine visible in chest x-ray"
- "rotational deformity and lateral curvature of thoracic spine"

### 3. Anatomical/Structural Prompts
Focus on anatomical structures and changes:
- "vertebral column showing coronal plane deviation"
- "thoracic scoliosis with vertebral rotation on chest imaging"

### 4. Medical Terminology Prompts
Professional medical language:
- "idiopathic thoracic scoliosis visible on posteroanterior chest radiograph"
- "spinal deformity presenting as lateral curvature with vertebral rotation"

### Prompt Ensemble
Each class uses **13 diverse prompts** that are averaged together for robust predictions. This ensemble approach reduces sensitivity to individual prompt variations and improves overall accuracy.

## How It Works

### 1. Text Encoding
All 13 prompts for each of the 6 OOD classes are encoded into text embeddings using the CLIP text encoder. This happens once at initialization.

### 2. Image Encoding
Each chest X-ray is:
- Preprocessed (resize, normalize)
- Encoded into an image embedding using the CLIP vision encoder

### 3. Similarity Computation
For each class:
- Compute cosine similarity between image embedding and all 13 text embeddings
- Average similarities across prompts (ensemble)
- Apply sigmoid with temperature scaling to convert to probabilities

### 4. Output
Generate predictions for all 6 OOD classes with confidence scores [0, 1].

## Output Format

The script generates a CSV file with the following format:

```csv
ImageID,Scoliosis,Osteopenia,Bulla,Infarction,Adenopathy,goiter
image001.png,0.123,0.456,0.089,0.234,0.567,0.012
image002.png,0.891,0.234,0.567,0.123,0.456,0.789
...
```

Each value represents the probability [0, 1] that the corresponding finding is present in the image.

## Performance Tips

1. **Batch Size**: Larger batch sizes (32-64) improve throughput
2. **GPU Memory**: WhyXrayCLIP (ViT-L-14) requires ~8GB VRAM
3. **Preprocessing**: Images are automatically normalized to [0, 1] and resized to 224×224
4. **Prompt Ensemble**: Using 13 prompts per class provides robust predictions

## Technical Details

### Image Preprocessing

**WhyXrayCLIP**:
- Resize to 224×224
- Convert to RGB (3 channels)
- Normalize with CLIP stats: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]

### Similarity to Probability Conversion

```python
# Cosine similarity: [-1, 1]
similarity = image_features @ text_features.T

# Average across prompts
class_score = similarity.mean(dim=1)

# Convert to probability with temperature scaling
probability = sigmoid(class_score * temperature)
```

Default temperature = 5.0 (tuned for optimal calibration)

## Code Design

This implementation is **clean and focused**:
- Single model (WhyXrayCLIP) - no model selection complexity
- Simple API - minimal command-line arguments
- Well-documented prompts - easy to understand and modify
- Efficient inference - optimized for production use

For advanced features (negative prompts, multi-scale, attention weighting), see the research codebase in the parent directory.

## Troubleshooting

### Import Errors

**OpenCLIP not found**:
```bash
pip install open-clip-torch
```

### Memory Issues

If you encounter OOM errors:
1. Reduce batch size: `--batch_size 16`
2. Close other applications to free GPU memory
3. Use a GPU with at least 8GB VRAM

### Poor Performance

1. **Check image quality**: Ensure images are readable chest X-rays
2. **Verify preprocessing**: Images should be properly loaded
3. **Review prompts**: The 13 prompts per class are optimized but can be customized
4. **Check batch size**: Larger batches may improve stability


## License

This project is for research purposes only.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- **WhyXrayCLIP** team at University of Pennsylvania for the CXR-specialized CLIP model
- **Microsoft Research** for CXR-CLIP (BioViL)
- **LAION** and **OpenAI** for OpenCLIP
- Prompt engineering strategies inspired by medical imaging literature and best practices
