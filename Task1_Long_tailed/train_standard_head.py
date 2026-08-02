import sys
import os
import warnings
import logging
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

import timm
from timm.utils import ModelEmaV2 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================
# CONFIGURATION - PADCHEST FINE-TUNING
# =============================================================================
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
RANK = int(os.environ.get("RANK", -1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", -1))

# --- PADCHEST PATHS ---
DATA_DIR = '/mnt/data/PadChest'
IMAGE_BASE_DIR = '/mnt/data/PadChest/images'
LABELS_CSV = '/mnt/data/PadChest/CXRLT_2026_training.csv'
CORRUPTED_CSV = '/mnt/data/PadChest/corrupted_images.csv'
REMOVAL_CSV = '/mnt/data/PadChest/Removal.csv'

# --- PRETRAINED MODEL FROM MIMIC-CXR ---
PRETRAINED_CHECKPOINT = '/home/hieuph2/CXR_LT/outputs/checkpoints/best_convnext_model.pth'

# --- RESUME TRAINING (Set to None to start from scratch) ---
RESUME_CHECKPOINT = '/home/hieuph2/CXR_LT/outputs/checkpoints_padchest/best_padchest_finetune.pth' # e.g., './outputs/checkpoints_padchest/best_padchest_finetune.pth'

# --- DATA SPLIT ---
VAL_SPLIT = 0.1  # 10% validation
RANDOM_SEED = 42

# --- TRAINING CONFIG ---
IMAGE_SIZE = 512
BATCH_SIZE = 64  # Per GPU - Training
VAL_BATCH_SIZE = 32  # Per GPU - Validation (smaller to avoid OOM)
NUM_WORKERS = 4  # Optimal: 4-8 workers per GPU

# Phase 1: Warm-up (Head-only training)
WARMUP_HEAD_EPOCHS = 3
WARMUP_HEAD_LR = 1e-3

# Phase 2: Full fine-tuning with differential LR
FINETUNE_EPOCHS = 60
# Learning rates scaled for batch_size=48 (1.5x from 32)
BACKBONE_LR = 1.5e-5  # Low LR to preserve MIMIC-CXR knowledge (scaled)
HEAD_LR = 1.5e-4      # Higher LR for new PadChest classes (scaled)

MAX_GRAD_NORM = 1.0       
EMA_DECAY = 0.9999
CHECKPOINT_DIR = './outputs/checkpoints_padchest'

# Setup Logging
if RANK == 0:
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Rank 0] - %(message)s')
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())

warnings.filterwarnings('ignore')

# =============================================================================
# DDP UTILS
# =============================================================================
def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)

def cleanup_ddp():
    dist.destroy_process_group()

def set_seed(seed):
    seed += RANK 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= WORLD_SIZE
    return rt

def gather_tensor(tensor):
    output_tensors = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)

# =============================================================================
# LOSS FUNCTION & LOGIT ADJUSTMENT
# =============================================================================
class LogitAdjustment(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        # Tính xác suất P(y) cho từng class
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.logit_adj = torch.from_numpy(np.log(prior + 1e-8)).float()
        self.tau = tau

    def forward(self, logits):
        # Đẩy logit adj lên cùng device với logits
        return logits - self.tau * self.logit_adj.to(logits.device)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        # Clamp logits for numerical stability
        x = torch.clamp(x, min=-80, max=80)

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

# =============================================================================
# DATASET
# =============================================================================
def get_transforms(is_train=False, img_size=512):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),  # LINEAR faster than default
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=10, 
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,  # Faster than reflect
                value=0
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ], additional_targets={})
    else:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ], additional_targets={})

class PadChestDataset(Dataset):  
    def __init__(self, df, class_names, image_base_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.class_names = class_names
        self.image_base_dir = image_base_dir
        self.transform = get_transforms(is_train=augment, img_size=IMAGE_SIZE)
        self.img_size = IMAGE_SIZE
        
        # Pre-convert labels to numpy for faster access
        self.labels = self.df[class_names].values.astype(np.float32)
        # Pre-compute image paths
        self.image_paths = [os.path.join(image_base_dir, fname) for fname in self.df["ImageID"].values]
    
    def __len__(self): 
        return len(self.df)
    
    def _load_and_preprocess_image(self, image_path):
        """Optimized image loading - keeps same preprocessing as training"""
        # Load as UNCHANGED to handle both grayscale and color
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError("Image not found")
        
        # Handle channels
        if len(image.shape) == 2:
            # Grayscale image
            if image.dtype != np.uint8:
                # OPTIMIZED: Deterministic strided sampling (faster than random)
                flat = image.ravel()
                stride = max(1, flat.size // 2000)  # Sample ~2000 pixels
                sampled = flat[::stride]
                p_low, p_high = np.percentile(sampled, (0.5, 99.5))
                
                if p_high > p_low:
                    image = np.clip(image, p_low, p_high)
                    image = ((image - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
                else:
                    image = ((image / image.max()) * 255.0).astype(np.uint8) if image.max() > 0 else image.astype(np.uint8)
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Handle non-uint8 (DICOM)
            if image.dtype != np.uint8:
                # Strided sampling for speed
                h, w, c = image.shape
                flat = image.reshape(-1, c)
                stride = max(1, flat.shape[0] // 2000)
                sampled = flat[::stride]
                
                p_low, p_high = np.percentile(sampled, (0.5, 99.5))
                if p_high > p_low:
                    image = np.clip(image, p_low, p_high)
                    image = ((image - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
                else:
                    image = ((image / image.max()) * 255.0).astype(np.uint8) if image.max() > 0 else image.astype(np.uint8)
        
        return image
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = self._load_and_preprocess_image(image_path)
        except Exception as e:
            # OPTIMIZED FALLBACK: Return black numpy array
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Apply Albumentations (Augmentation + Normalization)
        augmented = self.transform(image=image)
        image_tensor = augmented['image']
        
        # Use pre-converted labels (much faster)
        labels = torch.from_numpy(self.labels[idx])
        
        return image_tensor, labels

def fast_collate_fn(batch):
    """Optimized collate function for faster batching"""
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels

def load_removal_list():
    removal_images = set()
    if os.path.exists(CORRUPTED_CSV):
        corrupted_df = pd.read_csv(CORRUPTED_CSV)
        removal_images.update(corrupted_df["ImageID"].tolist())
    if os.path.exists(REMOVAL_CSV):
        removal_df = pd.read_csv(REMOVAL_CSV)
        removal_images.update(removal_df["ImageID"].tolist())
    if RANK == 0: logger.info(f"Total images to remove: {len(removal_images)}")
    return removal_images

def calculate_class_counts(df, class_names):
    """Tính số lượng mẫu positive cho từng class"""
    cls_counts = []
    for class_name in class_names:
        positive_count = df[class_name].sum()
        # Đảm bảo có ít nhất 1 mẫu để tránh log(0)
        cls_counts.append(max(1, int(positive_count)))
    
    if RANK == 0:
        logger.info(f"Class counts: {cls_counts}")
        logger.info(f"Min count: {min(cls_counts)}, Max count: {max(cls_counts)}")
    
    return cls_counts

def load_and_split_data():
    df = pd.read_csv(LABELS_CSV)
    original_len = len(df)
    
    if RANK == 0: logger.info(f"Loaded PadChest dataset: {original_len} images")
    
    removal_images = load_removal_list()
    if removal_images:
        df = df[~df["ImageID"].isin(removal_images)].reset_index(drop=True)
    
    metadata_cols = ['ImageID', 'StudyDate_DICOM', 'PatientID']
    class_names = [col for col in df.columns if col not in metadata_cols]
    
    if RANK == 0:
        logger.info(f"Number of classes: {len(class_names)}")
        logger.info(f"Samples after cleaning: {len(df)}")
    
    # Patient-level split
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(df, groups=df["PatientID"]))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    if RANK == 0:
        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")
    
    # Tính class counts từ training data
    cls_counts = calculate_class_counts(train_df, class_names)
    
    return train_df, val_df, class_names, cls_counts

# =============================================================================
# MODEL
# =============================================================================
class ConvNeXtV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_base', pretrained=True, num_classes=0, drop_path_rate=0.2)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.backbone.num_features),
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

def load_pretrained_and_surgery(checkpoint_path, num_padchest_classes):
    if RANK == 0: logger.info(f"PERFORMING CLASSIFIER SURGERY...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pretrained_state_dict = checkpoint['model_state_dict']
    
    model = ConvNeXtV2Classifier(num_classes=num_padchest_classes)
    model_state_dict = model.state_dict()
    
    pretrained_filtered = {}
    for key, value in pretrained_state_dict.items():
        if 'classifier.6' in key: continue # Skip final layer
        if key in model_state_dict and value.shape == model_state_dict[key].shape:
            pretrained_filtered[key] = value
    
    model.load_state_dict(pretrained_filtered, strict=False)
    
    # Initialize new head
    final_layer = model.classifier[6]
    nn.init.xavier_uniform_(final_layer.weight)
    if final_layer.bias is not None:
        nn.init.constant_(final_layer.bias, -2.0)
    
    return model

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if 'backbone' in name: param.requires_grad = False
        else: param.requires_grad = True

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def load_checkpoint_for_resume(checkpoint_path):
    """Load checkpoint to resume training"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if RANK == 0:
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"  Best mAP: {checkpoint.get('mAP', 'N/A'):.4f}")
    
    return checkpoint

# =============================================================================
# ENGINE
# =============================================================================
def train_epoch(model, model_ema, loader, criterion, optimizer, scaler, epoch, logit_adjuster=None):
    model.train()
    loader.sampler.set_epoch(epoch)
    
    avg_loss = 0.0
    steps = 0
    iterator = tqdm(loader, desc=f"Ep {epoch}", disable=(RANK != 0))
    
    for images, labels in iterator:
        images = images.cuda(LOCAL_RANK, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.cuda(LOCAL_RANK, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            logits = model(images)
            # Áp dụng LogitAdjustment nếu có
            if logit_adjuster is not None:
                logits = logit_adjuster(logits)
            loss = criterion(logits.float(), labels.float()) # Force FP32 for loss
        
        # NaN Check
        is_nan = torch.tensor([0.0], device=images.device)
        if torch.isnan(loss) or torch.isinf(loss): is_nan = torch.tensor([1.0], device=images.device)
        dist.all_reduce(is_nan, op=dist.ReduceOp.MAX)
        if is_nan.item() > 0:
            if RANK == 0: logger.warning("NaN Loss detected. Skipping batch.")
            optimizer.zero_grad()
            continue 

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        if model_ema: model_ema.update(model)
        
        reduced_loss = reduce_tensor(loss.detach())
        avg_loss += reduced_loss.item()
        steps += 1
        
        if RANK == 0:
            iterator.set_postfix({'loss': f'{reduced_loss.item():.4f}'})
            
    return avg_loss / max(steps, 1)

def evaluate(model, loader, criterion):
    model.eval()
    avg_loss = 0.0
    steps = 0
    local_preds, local_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", disable=(RANK != 0)):
            images = images.cuda(LOCAL_RANK, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.cuda(LOCAL_RANK, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits.float(), labels.float())
            
            reduced_loss = reduce_tensor(loss)
            avg_loss += reduced_loss.item()
            steps += 1
            
            local_preds.append(torch.sigmoid(logits.float()).detach())
            local_labels.append(labels.detach())
            
    local_preds = torch.cat(local_preds).cuda(LOCAL_RANK)
    local_labels = torch.cat(local_labels).cuda(LOCAL_RANK)
    
    # Gather from all GPUs
    global_preds = gather_tensor(local_preds)
    global_labels = gather_tensor(local_labels)
    
    preds_np = global_preds.cpu().numpy()
    labels_np = global_labels.cpu().numpy()
    
    metrics = {'loss': avg_loss / max(steps, 1), 'mAP': 0.0}
    
    if RANK == 0:
        # OPTIMIZATION: Remove DDP Padding (critical for accurate mAP)
        true_val_len = len(loader.dataset)
        preds_np = preds_np[:true_val_len]
        labels_np = labels_np[:true_val_len]

        ap_scores = []
        for i in range(labels_np.shape[1]):
            if labels_np[:, i].sum() > 0:
                ap_scores.append(average_precision_score(labels_np[:, i], preds_np[:, i]))
            else:
                ap_scores.append(0.0)
        metrics['mAP'] = np.mean(ap_scores) if ap_scores else 0.0
        
    return metrics

# =============================================================================
# MAIN
# =============================================================================
def main():
    setup_ddp()
    set_seed(42)
    
    if RANK == 0:
        logger.info(f"PADCHEST FINE-TUNING FROM MIMIC-CXR | GPUs: {WORLD_SIZE}")

    train_df, val_df, class_names, cls_counts = load_and_split_data()
    
    train_ds = PadChestDataset(train_df, class_names, IMAGE_BASE_DIR, augment=True)
    val_ds = PadChestDataset(val_df, class_names, IMAGE_BASE_DIR, augment=False)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,  # Conservative for stability
        collate_fn=fast_collate_fn,  
        drop_last=True  
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=VAL_BATCH_SIZE,  # Use smaller batch size for validation
        sampler=val_sampler,
        num_workers=NUM_WORKERS,  
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=4,
        collate_fn=fast_collate_fn  
    )
    
    # Check if resuming from checkpoint
    resume_checkpoint = None
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        resume_checkpoint = load_checkpoint_for_resume(RESUME_CHECKPOINT)
    
    # Load model
    if resume_checkpoint:
        # Resume: Load model with saved state
        model = ConvNeXtV2Classifier(num_classes=len(class_names))
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        if RANK == 0: logger.info("Loaded model state from checkpoint")
    else:
        # Fresh start: Load pretrained from MIMIC-CXR
        model = load_pretrained_and_surgery(PRETRAINED_CHECKPOINT, len(class_names))
    
    model = model.cuda(LOCAL_RANK)
    
    # Use channels_last memory format for faster GPU operations
    model = model.to(memory_format=torch.channels_last)
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Phase 1: Wrap with find_unused_parameters=True (backbone frozen)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05).cuda(LOCAL_RANK)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Initialize LogitAdjustment for class imbalance
    logit_adjuster = LogitAdjustment(cls_counts, tau=1.0).cuda(LOCAL_RANK)
    if RANK == 0: logger.info(f"LogitAdjustment initialized with tau=1.0")
    
    # Initialize training state
    if resume_checkpoint:
        # Backward compatibility: old checkpoints use 'mAP', new ones use 'best_mAP'
        best_mAP = resume_checkpoint.get('best_mAP', resume_checkpoint.get('mAP', 0.0))
        global_epoch = resume_checkpoint['epoch']
        start_finetune_epoch = resume_checkpoint.get('finetune_epoch', 0)
    else:
        best_mAP = 0.0
        global_epoch = 0
        start_finetune_epoch = 0
    
    # --- PHASE 1: WARM-UP ---
    if not resume_checkpoint:
        if RANK == 0: logger.info("PHASE 1: WARM-UP (Head-Only)")
        freeze_backbone(model.module)
        
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(head_params, lr=WARMUP_HEAD_LR, weight_decay=0.01, eps=1e-5)
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
        
        for epoch in range(1, WARMUP_HEAD_EPOCHS + 1):
            global_epoch += 1
            loss = train_epoch(model, None, train_loader, criterion, optimizer, scaler, global_epoch, logit_adjuster)
            metrics = evaluate(model.module, val_loader, criterion)
            scheduler.step()
            
            if RANK == 0:
                logger.info(f"[WARMUP] Ep {epoch} | Loss: {loss:.4f} | Val Loss: {metrics['loss']:.4f} | mAP: {metrics['mAP']:.4f}")
                if metrics['mAP'] > best_mAP: best_mAP = metrics['mAP']
    else:
        if RANK == 0: logger.info("SKIPPING WARMUP (resuming from checkpoint)")
    
    # --- PHASE 2: FULL FINE-TUNING ---
    if RANK == 0: logger.info("PHASE 2: FULL FINE-TUNING")
    
    # OPTIMIZATION: Unwrap and Re-wrap DDP for performance
    # We switch find_unused_parameters to FALSE since all params are now trained
    model_unwrapped = model.module
    unfreeze_all(model_unwrapped)
    
    # Create EMA before wrapping DDP (and ensure correct device)
    model_ema = ModelEmaV2(model_unwrapped, decay=EMA_DECAY, device=torch.device(f'cuda:{LOCAL_RANK}'))
    
    # Re-wrap DDP
    model = DDP(model_unwrapped, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False)
    
    backbone_params = []
    head_params = []
    for name, param in model.module.named_parameters():
        if 'backbone' in name: backbone_params.append(param)
        else: head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR * WORLD_SIZE},
        {'params': head_params, 'lr': HEAD_LR * WORLD_SIZE}
    ], weight_decay=0.05, eps=1e-5)
    
    scheduler = LambdaLR(optimizer, lambda epoch: max(0.01, 0.5 * (1.0 + np.cos(np.pi * epoch / FINETUNE_EPOCHS))))
    
    # Load optimizer and scheduler states if resuming
    if resume_checkpoint:
        if 'optimizer_state_dict' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            if RANK == 0: logger.info("Loaded optimizer state")
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            if RANK == 0: logger.info("Loaded scheduler state")
        if 'scaler_state_dict' in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
            if RANK == 0: logger.info("Loaded scaler state")
    
    # Start from the next epoch after checkpoint
    start_epoch = start_finetune_epoch + 1 if resume_checkpoint else 1
    if RANK == 0 and resume_checkpoint:
        logger.info(f"Resuming fine-tuning from epoch {start_epoch}/{FINETUNE_EPOCHS}")
    
    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        global_epoch += 1
        loss = train_epoch(model, model_ema, train_loader, criterion, optimizer, scaler, global_epoch, logit_adjuster)
        
        # Clear cache before validation to free memory
        torch.cuda.empty_cache()
        
        metrics = evaluate(model_ema.module, val_loader, criterion) # Eval on EMA
        scheduler.step()
        
        if RANK == 0:
            logger.info(f"[FINETUNE] Ep {epoch} | Train: {loss:.4f} | Val: {metrics['loss']:.4f} | mAP: {metrics['mAP']:.4f}")
            
            if metrics['mAP'] > best_mAP:
                best_mAP = metrics['mAP']
                # Save comprehensive checkpoint
                checkpoint = {
                    'epoch': global_epoch,
                    'finetune_epoch': epoch,
                    'model_state_dict': model_ema.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_mAP': best_mAP,
                    'mAP': best_mAP,
                    'class_names': class_names
                }
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_padchest_finetune.pth"))
                # Also save a backup with epoch number
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{global_epoch}.pth"))
                logger.info(f" >>> Saved Best (mAP: {best_mAP:.4f})")

    cleanup_ddp()

if __name__ == "__main__":
    main()