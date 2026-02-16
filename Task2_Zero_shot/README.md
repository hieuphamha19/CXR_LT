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

- **Zero-Shot Detection**: No training required, uses pre-trained CLIP models
- **Multiple CLIP Models Supported**:
  - **WhyXrayCLIP** (Recommended): CXR-specialized CLIP from UPenn
  - **Microsoft CXR-CLIP (BioViL)**: Medical imaging specialized model
  - **OpenCLIP**: General-purpose CLIP models (ViT-B-16, ViT-L-14, etc.)
- **Enhanced Prompt Engineering**:
  - Multi-template prompts (13 prompts per class)
  - Clinical, radiological, anatomical, and medical terminology perspectives
  - Prompt ensemble for robust predictions
- **Optimized for Medical Imaging**: Specialized preprocessing for chest X-rays

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

# For WhyXrayCLIP (Recommended - Best Performance)
pip install open-clip-torch

# For Microsoft CXR-CLIP (Optional - Alternative)
pip install hi-ml-multimodal
```

## Usage

### Basic Usage (WhyXrayCLIP - Recommended)

```bash
python zero_shot_ood_detection.py \
  --test_csv /path/to/test.csv \
  --image_dir /path/to/images \
  --output submission.csv
```

### Using Microsoft CXR-CLIP (BioViL)

```bash
python zero_shot_ood_detection.py \
  --use_cxr_clip \
  --test_csv /path/to/test.csv \
  --image_dir /path/to/images \
  --output submission_biovil.csv
```

### Using Different OpenCLIP Models

```bash
# ViT-L-14 with LAION-2B
python zero_shot_ood_detection.py \
  --clip_model ViT-L-14 \
  --clip_pretrained laion2b_s34b_b88k \
  --test_csv /path/to/test.csv \
  --image_dir /path/to/images \
  --output submission_vitl14.csv

# ViT-B-16 with OpenAI weights
python zero_shot_ood_detection.py \
  --clip_model ViT-B-16 \
  --clip_pretrained openai \
  --test_csv /path/to/test.csv \
  --image_dir /path/to/images \
  --output submission_vitb16.csv
```

### Command-Line Arguments

```
--clip_model              CLIP model architecture (default: ViT-L-14)
                         Options: ViT-B-16, ViT-L-14, ViT-H-14, etc.
                         Or use: hf-hub:yyupenn/whyxrayclip for CXR-specialized

--clip_pretrained        Pretrained weights (default: hf-hub:yyupenn/whyxrayclip)
                         Options: openai, laion2b_s34b_b88k, laion400m_e32, etc.

--use_cxr_clip          Use Microsoft CXR-CLIP (BioViL) instead of OpenCLIP
                         Specialized for chest X-rays

--batch_size            Batch size for inference (default: from config)

--output                Output CSV file path (default: auto-generated)

--test_csv              Path to test CSV file

--image_dir             Path to image directory
```

## Model Comparison

### WhyXrayCLIP (Recommended)
- **Source**: University of Pennsylvania
- **Specialty**: Trained specifically on chest X-rays
- **Architecture**: ViT-L-14
- **Advantages**: Best performance on CXR tasks, understands medical terminology
- **Installation**: `pip install open-clip-torch`

### Microsoft CXR-CLIP (BioViL)
- **Source**: Microsoft Research
- **Specialty**: Medical imaging (CXR + radiology reports)
- **Architecture**: ResNet50 + CXR-BERT
- **Advantages**: Strong medical domain knowledge, robust to variations
- **Installation**: `pip install hi-ml-multimodal`

### OpenCLIP (General)
- **Source**: LAION / OpenAI
- **Specialty**: General vision-language understanding
- **Architecture**: Various (ViT-B-16, ViT-L-14, ViT-H-14, etc.)
- **Advantages**: Large-scale pre-training, good generalization
- **Installation**: `pip install open-clip-torch`

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

1. **Use WhyXrayCLIP**: Consistently gives best results on chest X-ray tasks
2. **Batch Size**: Larger batch sizes (32-64) improve throughput
3. **GPU Memory**: 
   - ViT-B-16: ~4GB VRAM
   - ViT-L-14: ~8GB VRAM
   - CXR-CLIP (BioViL): ~6GB VRAM
4. **Preprocessing**: Images are automatically normalized to [0, 1] and resized

## Technical Details

### Image Preprocessing

**For OpenCLIP (WhyXrayCLIP, ViT models)**:
- Resize to 224×224
- Convert to RGB (3 channels)
- Normalize with CLIP stats: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]

**For Microsoft CXR-CLIP (BioViL)**:
- Resize to 480×480
- Convert to RGB (3 channels)
- Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

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

## Advanced Features (Future Work)

The codebase is designed to support advanced features:
- **Negative Prompts**: Use contrastive prompts to improve discrimination
- **Multi-Scale Inference**: Test at multiple resolutions
- **Attention-Based Prompt Weighting**: Learn optimal prompt combinations
- **Learnable Temperature**: Per-class temperature scaling

These features are implemented in `task2_best_v3.py` and can be integrated if needed.

## Troubleshooting

### Import Errors

**OpenCLIP not found**:
```bash
pip install open-clip-torch
```

**CXR-CLIP not found**:
```bash
pip install hi-ml-multimodal
```

### Memory Issues

If you encounter OOM errors:
1. Reduce batch size: `--batch_size 16`
2. Use smaller model: `--clip_model ViT-B-16`
3. Enable CPU offloading (slower but works)

### Poor Performance

1. **Try WhyXrayCLIP**: Best for chest X-rays
2. **Check image quality**: Ensure images are readable
3. **Verify preprocessing**: Images should be grayscale CXRs
4. **Review prompts**: Customize prompts for your specific use case


## License

This project is for research purposes only.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- **WhyXrayCLIP** team at University of Pennsylvania for the CXR-specialized CLIP model
- **Microsoft Research** for CXR-CLIP (BioViL)
- **LAION** and **OpenAI** for OpenCLIP
- Prompt engineering strategies inspired by medical imaging literature and best practices
