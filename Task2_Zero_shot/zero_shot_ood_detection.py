"""
Task 2: Zero-Shot OOD Detection using Vision-Language Model
Predicts 6 OOD classes using CLIP-based models

OOD Classes: Scoliosis, Osteopenia, Bulla, Infarction, Adenopathy, goiter
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import config
import importlib.util
config_path = os.path.join(current_dir, 'config.py')
spec = importlib.util.spec_from_file_location("local_config", config_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from utils import set_seed

# Import OpenCLIP (required for WhyXrayCLIP)
try:
    import open_clip
    logger.info("✓ OpenCLIP available")
except ImportError:
    logger.error("ERROR: OpenCLIP not available. Install with: pip install open-clip-torch")
    raise

# Config
DEVICE = cfg.DEVICE
DATA_DIR = cfg.DATA_DIR
IMAGE_DIR = cfg.IMAGE_DIR
IMAGE_SIZE = cfg.IMAGE_SIZE
BATCH_SIZE = cfg.BATCH_SIZE
NUM_WORKERS = cfg.NUM_WORKERS
CHECKPOINT_DIR = cfg.CHECKPOINT_DIR
OUTPUT_DIR = cfg.OUTPUT_DIR
# Use Task 2 test file (contains OOD classes)
TEST_CSV = os.path.join(DATA_DIR, "CXRLT_2026_TEST_ALL_TASK2.csv")
RANDOM_SEED = cfg.RANDOM_SEED
USE_AMP = getattr(cfg, 'USE_AMP', False)

# Class names - Only 6 OOD classes
OOD_CLASSES = [
    "Scoliosis", "Osteopenia", "Bulla", 
    "Infarction", "Adenopathy", "goiter"
]

print(f"Device: {DEVICE}")
print(f"OOD classes: {len(OOD_CLASSES)}")

set_seed(RANDOM_SEED)


# ============================================================================
# Data Loading
# ============================================================================

def load_removal_list(for_test=True):
    """Load list of images to remove (corrupted images only for test)."""
    removal_images = set()
    
    # Load corrupted images
    corrupted_csv = "dataset/corrupted_images.csv"
    if os.path.exists(corrupted_csv):
        corrupted_df = pd.read_csv(corrupted_csv)
        corrupted_images = set(corrupted_df["ImageID"].tolist())
        removal_images.update(corrupted_images)
        logger.info(f"Loaded {len(corrupted_images)} corrupted images to remove")
    
    if not for_test:
        removal_csv = os.path.join(DATA_DIR, "Removal.csv")
        if os.path.exists(removal_csv):
            removal_df = pd.read_csv(removal_csv)
            removal_from_comp = set(removal_df["ImageID"].tolist())
            removal_images.update(removal_from_comp)
            logger.info(f"Loaded {len(removal_from_comp)} images from Removal.csv")
    
    logger.info(f"Total images to remove: {len(removal_images)}")
    return removal_images


class CXRDataset(Dataset):
    """Dataset for chest X-rays."""
    
    def __init__(self, df, image_dir, image_size=512):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_size = image_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["ImageID"]
        image_path = os.path.join(self.image_dir, image_id)
        
        try:
            # Load image with cv2
            image_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image_array is None:
                raise ValueError(f"Failed to load: {image_path}")
            
            # Convert to float32 and normalize
            image_array = image_array.astype(np.float32)
            
            # Normalize 16-bit to 8-bit if needed
            if image_array.max() > 255:
                image_array = image_array / 65535.0
            else:
                image_array = image_array / 255.0
            
            # Resize
            image_array = cv2.resize(image_array, (self.image_size, self.image_size), 
                                    interpolation=cv2.INTER_LINEAR)
            
            # Convert to RGB (3 channels) for standard ResNet50
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=0)  # (3, H, W)
            elif len(image_array.shape) == 3:
                if image_array.shape[2] == 1:
                    image_array = np.repeat(image_array, 3, axis=2)
                    image_array = np.transpose(image_array, (2, 0, 1))  # (3, H, W)
                elif image_array.shape[2] == 3:
                    image_array = np.transpose(image_array, (2, 0, 1))  # (3, H, W)
                else:
                    image_array = image_array.mean(axis=2)
                    image_array = np.stack([image_array] * 3, axis=0)
            
            image_tensor = torch.tensor(image_array, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            image_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        
        return image_tensor, image_id


# ============================================================================
# Zero-Shot OOD Detector using CLIP
# ============================================================================

class CLIPZeroShotOOD:
    """Zero-shot OOD detection using WhyXrayCLIP (CXR-specialized CLIP)."""
    
    def __init__(self, device='cuda'):
        """
        Initialize WhyXrayCLIP model for zero-shot classification.
        
        Args:
            device: Device to run on
        """
        self.device = device
        
        # Load WhyXrayCLIP from Hugging Face Hub
        model_name = "hf-hub:yyupenn/whyxrayclip"
        logger.info(f"Loading WhyXrayCLIP - CXR-specialized CLIP model")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info("✓ WhyXrayCLIP loaded successfully")
        
        # Define text prompts for OOD classes
        self.ood_classes = OOD_CLASSES
        self.text_prompts = self._create_text_prompts()
        
        # Pre-encode text prompts
        self.text_features = self._encode_text_prompts()
        
        logger.info(f"✓ Model ready for {len(self.ood_classes)} OOD classes")
    
    def _create_text_prompts(self):
        """
        Create enhanced multi-template text prompts for each OOD class.
        Uses multiple prompt strategies: clinical, radiological, anatomical, and descriptive.
        Based on best-performing prompts from task2_best_v3.py
        """
        prompts = {
            "Scoliosis": [
                # Clinical/Descriptive prompts
                "a chest x-ray showing scoliosis",
                "chest radiograph demonstrating scoliosis with spinal curvature",
                "thoracic spine with lateral deviation indicating scoliosis",
                
                # Radiological findings prompts
                "abnormal lateral curvature of the spine visible in chest x-ray",
                "rotational deformity and lateral curvature of thoracic spine",
                "spinal column with abnormal sideways curvature in frontal radiograph",
                "asymmetric rib cage and curved spine consistent with scoliosis",
                
                # Anatomical/Structural prompts
                "vertebral column showing coronal plane deviation",
                "thoracic scoliosis with vertebral rotation on chest imaging",
                "structural spinal deformity with three-dimensional curvature",
                
                # Medical terminology prompts
                "idiopathic thoracic scoliosis visible on posteroanterior chest radiograph",
                "spinal deformity presenting as lateral curvature with vertebral rotation"
            ],
            "Osteopenia": [
                # Clinical/Descriptive prompts
                "a chest x-ray showing osteopenia",
                "chest radiograph with decreased bone density",
                "reduced bone mineralization visible in thoracic skeleton",
                
                # Radiological findings prompts
                "decreased bone mineral density in ribs and spine on chest x-ray",
                "osteopenic changes with increased radiolucency of bones",
                "thin cortices and decreased trabecular density in thoracic bones",
                "diffuse demineralization of thoracic skeleton",
                
                # Anatomical/Structural prompts
                "low bone mass evident in vertebral bodies and ribs",
                "reduced bone opacity throughout thoracic cage",
                "decreased skeletal density in chest imaging",
                
                # Medical terminology prompts
                "osteopenia with generalized decrease in bone radiodensity",
                "pre-osteoporotic bone changes visible on chest radiograph"
            ],
            "Bulla": [
                # Clinical/Descriptive prompts
                "a chest x-ray showing pulmonary bulla",
                "chest radiograph demonstrating large air-filled space in lung",
                "emphysematous bulla visible as hyperlucent area",
                
                # Radiological findings prompts
                "well-defined thin-walled air space exceeding one centimeter",
                "avascular area with hairline wall in lung parenchyma",
                "large radiolucent region with no visible lung markings",
                "giant bulla occupying significant portion of hemithorax",
                
                # Anatomical/Structural prompts
                "bullous emphysema with destroyed alveolar walls",
                "air-filled cystic space within lung parenchyma",
                "emphysematous destruction creating large air pocket",
                
                # Medical terminology prompts
                "bullous lung disease with parenchymal destruction on chest x-ray",
                "emphysematous bulla presenting as localized hyperlucency"
            ],
            "Infarction": [
                # Clinical/Descriptive prompts
                "a chest x-ray showing pulmonary infarction",
                "chest radiograph with lung tissue infarction",
                "areas of infarcted lung parenchyma visible on imaging",
                
                # Radiological findings prompts
                "wedge-shaped peripheral opacity consistent with pulmonary infarct",
                "Hampton's hump indicating pulmonary infarction",
                "pleural-based consolidation from lung tissue necrosis",
                "peripheral triangular opacity with base toward pleura",
                
                # Anatomical/Structural prompts
                "focal region of ischemic lung tissue damage",
                "pulmonary parenchyma with hemorrhagic infarction",
                "segmental or subsegmental lung infarct",
                
                # Medical terminology prompts
                "pulmonary infarction secondary to thromboembolism on chest radiograph",
                "ischemic necrosis of lung tissue visible as peripheral opacity"
            ],
            "Adenopathy": [
                # Clinical/Descriptive prompts
                "a chest x-ray showing adenopathy",
                "chest radiograph with enlarged lymph nodes",
                "lymphadenopathy visible in mediastinum or hilum",
                
                # Radiological findings prompts
                "mediastinal lymphadenopathy causing widening of mediastinum",
                "hilar adenopathy with bilateral lymph node enlargement",
                "enlarged mediastinal and hilar lymph nodes on chest imaging",
                "prominent lymphadenopathy in paratracheal region",
                
                # Anatomical/Structural prompts
                "pathologically enlarged thoracic lymph nodes",
                "mediastinal mass effect from lymph node enlargement",
                "bilateral hilar fullness from adenopathy",
                
                # Medical terminology prompts
                "intrathoracic lymphadenopathy visible on posteroanterior radiograph",
                "mediastinal and bilateral hilar lymph node enlargement"
            ],
            "goiter": [
                # Clinical/Descriptive prompts
                "a chest x-ray showing goiter",
                "chest radiograph with enlarged thyroid gland",
                "thyroid enlargement extending into superior mediastinum",
                
                # Radiological findings prompts
                "superior mediastinal mass from retrosternal goiter",
                "tracheal deviation caused by thyroid goiter",
                "retrosternal thyroid extension visible on chest x-ray",
                "widened superior mediastinum from substernal goiter",
                
                # Anatomical/Structural prompts
                "enlarged thyroid gland displacing trachea and esophagus",
                "cervicothoracic goiter extending below thoracic inlet",
                "substernal thyroid mass in anterior mediastinum",
                
                # Medical terminology prompts
                "retrosternal goiter causing mass effect in superior mediastinum",
                "intrathoracic thyroid enlargement with tracheal compression"
            ]
        }
        return prompts
    
    def _encode_text_prompts(self):
        """Pre-encode all text prompts."""
        all_features = {}
        
        with torch.no_grad():
            for class_name, prompts in self.text_prompts.items():
                # WhyXrayCLIP text encoding
                text_tokens = self.tokenizer(prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                # Normalize
                text_features = F.normalize(text_features, dim=-1)
                
                all_features[class_name] = text_features
        
        return all_features
    
    def preprocess_image(self, image_tensor):
        """
        Preprocess image for WhyXrayCLIP.
        Input: (B, 3, H, W) RGB tensor [0, 1]
        Output: (B, 3, H, W) RGB tensor normalized for CLIP
        """
        # WhyXrayCLIP expects 224x224 with CLIP normalization
        image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(image_tensor.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(image_tensor.device)
        
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor
    
    def predict_batch(self, images):
        """
        Predict OOD class probabilities for a batch of images.
        
        Args:
            images: Tensor of shape (B, 3, H, W) - RGB CXR images [0, 1]
        
        Returns:
            predictions: Tensor of shape (B, num_ood_classes) - probabilities [0, 1]
        """
        with torch.no_grad():
            # Preprocess images for WhyXrayCLIP
            images = self.preprocess_image(images.to(self.device))
            
            # Encode images with WhyXrayCLIP
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            # Compute similarities with text prompts
            predictions = []
            
            for class_name in self.ood_classes:
                text_features = self.text_features[class_name]
                
                # Compute similarity: (B, D) @ (D, num_prompts) = (B, num_prompts)
                similarity = (image_features @ text_features.T)  # Cosine similarity
                
                # Average over prompts (ensemble)
                class_score = similarity.mean(dim=1)  # (B,)
                
                predictions.append(class_score)
            
            # Stack predictions: (B, num_ood_classes)
            predictions = torch.stack(predictions, dim=1)
            
            # Convert to probabilities
            # CLIP similarities are in [-1, 1], need to map to [0, 1]
            # Use sigmoid with scaling for better range
            predictions = torch.sigmoid(predictions * 5.0)  # Scale factor of 5
            
        return predictions


# ============================================================================
# OOD Detector (CLIP only)
# ============================================================================

class OODDetector:
    """
    Zero-shot OOD detection using CLIP for 6 OOD classes only.
    """
    
    def __init__(self, clip_model, device='cuda'):
        self.clip_model = clip_model
        self.device = device
    
    def predict_batch(self, images):
        """
        Predict 6 OOD classes.
        
        Args:
            images: Tensor of shape (B, 3, H, W)
        
        Returns:
            predictions: Tensor of shape (B, 6)
        """
        # OOD predictions (6 classes)
        ood_preds = self.clip_model.predict_batch(images)
        
        return ood_preds


# ============================================================================
# Inference
# ============================================================================

def run_inference(ood_detector, dataloader):
    """Run inference on test set."""
    all_predictions = []
    all_image_ids = []
    
    logger.info("Running inference...")
    
    for images, image_ids in tqdm(dataloader, desc="Predicting"):
        preds = ood_detector.predict_batch(images)
        
        all_predictions.append(preds.cpu().numpy())
        all_image_ids.extend(image_ids)
    
    all_predictions = np.vstack(all_predictions)
    
    return all_predictions, all_image_ids


def create_submission(predictions, image_ids, output_path):
    """Create submission CSV for Task 2 (6 OOD classes only)."""
    submission_data = {'ImageID': image_ids}
    
    for i, class_name in enumerate(OOD_CLASSES):
        submission_data[class_name] = predictions[:, i]
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✓ Submission saved to: {output_path}")
    logger.info(f"  Shape: {submission_df.shape}")
    logger.info(f"  Columns: {submission_df.columns.tolist()}")
    
    return submission_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Task 2: Zero-Shot OOD Detection using WhyXrayCLIP')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output submission file path')
    parser.add_argument('--test_csv', type=str, default=None,
                        help='Path to test CSV file (default: from config)')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to image directory (default: from config)')
    
    args = parser.parse_args()
    
    batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
    test_csv = args.test_csv if args.test_csv is not None else TEST_CSV
    image_dir = args.image_dir if args.image_dir is not None else IMAGE_DIR
    
    print("\n" + "="*60)
    print("TASK 2: ZERO-SHOT OOD DETECTION (6 CLASSES)")
    print("="*60)
    print("Model: WhyXrayCLIP (CXR-specialized CLIP from UPenn)")
    print(f"Batch Size: {batch_size}")
    print(f"Test CSV: {test_csv}")
    print(f"Image Dir: {image_dir}")
    
    # Load test data
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    if not os.path.exists(test_csv):
        logger.error(f"ERROR: Test CSV not found at {test_csv}")
        return
    
    test_df = pd.read_csv(test_csv)
    original_len = len(test_df)
    logger.info(f"Original test samples: {original_len}")
    
    # Remove corrupted images
    removal_images = load_removal_list(for_test=True)
    if removal_images:
        test_df = test_df[~test_df["ImageID"].isin(removal_images)].reset_index(drop=True)
        removed_count = original_len - len(test_df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} corrupted images")
        logger.info(f"Final test samples: {len(test_df)}")
    
    # Create dataset
    test_dataset = CXRDataset(test_df, image_dir, image_size=IMAGE_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=16 if NUM_WORKERS > 0 else None
    )
    
    # Load WhyXrayCLIP model
    print("\n" + "="*60)
    print("LOADING WHYXRAYCLIP MODEL")
    print("="*60)
    
    clip_model = CLIPZeroShotOOD(device=DEVICE)
    
    # Create OOD detector
    print("\n" + "="*60)
    print("CREATING OOD DETECTOR")
    print("="*60)
    
    ood_detector = OODDetector(
        clip_model=clip_model,
        device=DEVICE
    )
    
    logger.info("✓ OOD detector ready")
    logger.info(f"  - WhyXrayCLIP: 6 OOD classes")
    
    # Run inference
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    predictions, image_ids = run_inference(
        ood_detector,
        test_loader
    )
    
    # Create submission
    print("\n" + "="*60)
    print("CREATING SUBMISSION")
    print("="*60)
    
    if args.output is None:
        output_name = "submission_task2_whyxrayclip.csv"
        output_path = os.path.join(OUTPUT_DIR, output_name)
    else:
        output_path = args.output
    
    submission_df = create_submission(predictions, image_ids, output_path)
    
    # Statistics
    print("\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    
    print("\n--- OOD Classes (6 classes) ---")
    for i, class_name in enumerate(OOD_CLASSES):
        pred_mean = predictions[:, i].mean()
        pred_std = predictions[:, i].std()
        positive_count = (predictions[:, i] > 0.5).sum()
        positive_pct = positive_count / len(predictions) * 100
        print(f"  {class_name:30s}: mean={pred_mean:.4f}, std={pred_std:.4f}, >0.5: {positive_count:4d} ({positive_pct:5.2f}%)")
    
    print("\n" + "="*60)
    print("TASK 2 INFERENCE COMPLETE! 🎉")
    print("="*60)
    print(f"✓ Submission file: {output_path}")
    print(f"✓ Total predictions: {len(predictions)}")
    print(f"✓ Classes: 6 OOD classes only")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory Used:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

