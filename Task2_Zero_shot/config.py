import os
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TRAIN_CSV = os.path.join(BASE_DIR, "dataset", "CXRLT_2026_training.csv")
TEST_CSV = os.path.join(BASE_DIR, "dataset", "CXRLT_2026_TEST_ALL_TASK1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Class names (30 classes for Task 1)
CLASS_NAMES = [
    "Normal", "aortic elongation", "cardiomegaly", "pleural effusion", 
    "Nodule", "atelectasis", "pleural thickening", "aortic atheromatosis",
    "Support Devices", "alveolar pattern", "fracture", "Hernia", 
    "Emphysema", "azygos lobe", "Hydropneumothorax", "Kyphosis",
    "Mass", "Pneumothorax", "Subcutaneous Emphysema", "pneumoperitoneo",
    "vascular hilar enlargement", "vertebral degenerative changes", 
    "hyperinflated lung", "interstitial pattern", "central venous catheter",
    "hypoexpansion", "bronchiectasis", "hemidiaphragm elevation", 
    "sternotomy", "calcified densities"
]
NUM_CLASSES = len(CLASS_NAMES)

# Training params
IMAGE_SIZE = 512  # Match TorchXRayVision ResNet50-res512 pretrained resolution
BATCH_SIZE = 64  # Reduced from 64 to 32 - slower GPU = more time for slow disk I/O
NUM_WORKERS = 16  # Increased to 8 for parallel disk reads with smaller batches
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# GPU Optimization
USE_AMP = True  # Automatic Mixed Precision for faster training
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients over 2 steps for effective batch size of 128

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
