#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ensemble Inference for Task 1
- Combines predictions from multiple ConvNeXtV2 models:
  1. Standard head (from train_convnext_padchest.py)
  2. CSRA head (from train_convnext_padchest_v3.py)
- Supports TTA and post-processing
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import TTA utilities
from tta import MultiModelTTA, TTAPredictor, TTA_CONFIGS


# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ensemble_infer")


# =============================================================================
# Configuration defaults
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
RANDOM_SEED = 42

# AMP
AMP_ENABLED = torch.cuda.is_available()


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# PadChest class names (30 classes) - MUST match training order!
CLASS_NAMES = [
    "Normal", "aortic elongation", "cardiomegaly", "pleural effusion",
    "Nodule", "atelectasis", "pleural thickening", "aortic atheromatosis",
    "Support Devices", "alveolar pattern", "fracture", "Hernia",
    "Emphysema", "azygos lobe", "Hydropneumothorax", "Kyphosis",
    "Mass", "Pneumothorax", "Subcutaneous Emphysema", "pneumoperitoneo",
    "vascular hilar enlargement", "vertebral degenerative changes",
    "hyperinflated lung", "interstitial pattern", "central venous catheter",
    "hypoexpansion", "bronchiectasis", "hemidiaphragm elevation",
    "sternotomy", "calcified densities",
]
NUM_CLASSES = len(CLASS_NAMES)


# =============================================================================
# Model Architectures
# =============================================================================

# Model 1: Standard Head
class ConvNeXtV2Classifier(nn.Module):
    def __init__(self, num_classes: int, drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_base",
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.backbone.num_features),
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


# Model 2: CSRA Head
class CSRA(nn.Module):
    def __init__(self, input_dim, num_classes, lam=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.classifier = nn.Linear(input_dim, num_classes)
        self.conv_att = nn.Conv2d(input_dim, num_classes, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        gap_feat = torch.mean(x, dim=(2, 3))
        logit_gap = self.classifier(gap_feat)

        att_map = self.conv_att(x).view(b, self.num_classes, h * w)
        att_score = self.softmax(att_map)

        x_flat = x.view(b, c, h * w)
        csra_feat = torch.bmm(att_score, x_flat.permute(0, 2, 1))

        w_cls = self.classifier.weight
        logit_csra = torch.sum(csra_feat * w_cls.unsqueeze(0), dim=2) + self.classifier.bias
        return logit_gap + self.lam * logit_csra


class ConvNeXtV2ClassifierCSRA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_base",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.2,
            global_pool=""
        )
        nf = self.backbone.num_features
        self.bn = nn.BatchNorm2d(nf)
        self.head = CSRA(input_dim=nf, num_classes=num_classes, lam=0.1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        x = self.head(x)
        return x


def load_checkpoint_safely(path: str) -> Dict[str, Any]:
    """torch.load compatibility"""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    return ckpt


def detect_model_type(state_dict: Dict[str, Any]) -> str:
    """Detect if model uses standard head or CSRA head"""
    if any("head.classifier" in k for k in state_dict.keys()):
        return "csra"
    elif any("classifier.6" in k for k in state_dict.keys()):
        return "standard"
    else:
        logger.warning("Could not detect model type from state_dict. Defaulting to 'standard'.")
        return "standard"


def load_model(checkpoint_path: str, num_classes: int) -> nn.Module:
    logger.info(f"Loading model from: {checkpoint_path}")
    ckpt = load_checkpoint_safely(checkpoint_path)

    state_key = "model_state_dict" if "model_state_dict" in ckpt else None
    if state_key is None:
        logger.warning("Checkpoint has no 'model_state_dict'. Trying to load as state_dict directly.")
        state_dict = ckpt
    else:
        state_dict = ckpt[state_key]

    # Detect model type
    model_type = detect_model_type(state_dict)
    logger.info(f"  Detected model type: {model_type}")

    # Load appropriate architecture
    if model_type == "csra":
        model = ConvNeXtV2ClassifierCSRA(num_classes=num_classes)
    else:
        model = ConvNeXtV2Classifier(num_classes=num_classes)

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    if isinstance(ckpt, dict):
        if "mAP" in ckpt or "best_mAP" in ckpt:
            mAP = ckpt.get("best_mAP", ckpt.get("mAP", 0.0))
            logger.info(f"  ✓ Loaded! Checkpoint mAP: {float(mAP):.4f}")
        if "epoch" in ckpt:
            logger.info(f"  Trained for {ckpt['epoch']} epochs")

    return model


# =============================================================================
# Dataset
# =============================================================================
class CXRDataset(Dataset):
    """Dataset for chest X-rays with PadChest-like preprocessing."""

    def __init__(self, df: pd.DataFrame, image_dir: str, image_size: int = 512, image_col: str = "ImageID"):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_size = image_size

        if image_col not in self.df.columns:
            fallback_col = self.df.columns[0]
            logger.warning(f"Column '{image_col}' not found. Using '{fallback_col}' as image id column.")
            self.image_col = fallback_col
        else:
            self.image_col = image_col

        self.image_ids = self.df[self.image_col].astype(str).values.tolist()
        self.image_paths = [os.path.join(image_dir, fname) for fname in self.image_ids]

        # ImageNet stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _percentile_rescale_to_uint8(img: np.ndarray, p1: float = 0.5, p2: float = 99.5) -> np.ndarray:
        """Fast percentile clip + rescale to uint8"""
        if img.size == 0:
            return img.astype(np.uint8)

        flat = img.ravel()
        stride = max(1, flat.size // 2000)
        sampled = flat[::stride]

        p_low, p_high = np.percentile(sampled, (p1, p2))
        if p_high > p_low:
            out = np.clip(img, p_low, p_high)
            out = ((out - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
        else:
            mx = float(img.max()) if img.max() is not None else 0.0
            out = ((img / mx) * 255.0).astype(np.uint8) if mx > 0 else img.astype(np.uint8)
        return out

    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Image not found or unreadable")

        # Handle grayscale vs color
        if len(image.shape) == 2:
            if image.dtype != np.uint8:
                image = self._percentile_rescale_to_uint8(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.dtype != np.uint8:
                h, w, c = image.shape
                flat = image.reshape(-1, c)
                stride = max(1, flat.shape[0] // 2000)
                sampled = flat[::stride]
                p_low, p_high = np.percentile(sampled, (0.5, 99.5))
                if p_high > p_low:
                    image = np.clip(image, p_low, p_high)
                    image = ((image - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
                else:
                    mx = float(image.max()) if image.max() is not None else 0.0
                    image = ((image / mx) * 255.0).astype(np.uint8) if mx > 0 else image.astype(np.uint8)

        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image_id = self.image_ids[idx]

        try:
            image = self._load_and_preprocess_image(image_path)
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Resize to fixed size
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # To tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image, image_id


# TTA is now imported from tta.py


# =============================================================================
# Post-processing
# =============================================================================
def postprocess_none(probs: np.ndarray, eps: float = 1e-6, **kwargs) -> np.ndarray:
    return np.clip(probs, eps, 1.0 - eps)


def postprocess_normal_gating(probs: np.ndarray, alpha: float = 1.0, eps: float = 1e-6, **kwargs) -> np.ndarray:
    """p_c := p_c * (1 - p_normal)^alpha for c != Normal"""
    p = probs.copy()
    p0 = p[:, 0:1]
    abnormal = np.clip(1.0 - p0, 0.0, 1.0)
    p[:, 1:] = p[:, 1:] * (abnormal ** float(alpha))
    return np.clip(p, eps, 1.0 - eps)


def apply_postprocess(
    probs: np.ndarray,
    mode: str,
    eps: float = 1e-6,
    normal_alpha: float = 1.0,
) -> np.ndarray:
    mode = (mode or "none").lower().strip()
    if mode == "none":
        return postprocess_none(probs, eps=eps)
    if mode == "normal_gating":
        return postprocess_normal_gating(probs, alpha=normal_alpha, eps=eps)
    raise ValueError(f"Unknown postprocess mode '{mode}'. Valid: none, normal_gating")


# =============================================================================
# Ensemble Inference using tta.py
# =============================================================================
class DataLoaderWrapper:
    """Wrapper to make our dataloader compatible with tta.py expectations"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.image_ids = []
    
    def __iter__(self):
        self.image_ids = []
        for images, ids in self.dataloader:
            self.image_ids.extend(list(ids))
            yield images
    
    def __len__(self):
        return len(self.dataloader)


@torch.no_grad()
def predict_ensemble(
    models: List[nn.Module],
    model_weights: List[float],
    dataloader: DataLoader,
    use_tta: bool = False,
    tta_config: str = "flip_scale",
    tta_merge_mode: str = "mean",
    ensemble_mode: str = "mean",
) -> Tuple[np.ndarray, List[str]]:
    """
    Ensemble prediction from multiple models using tta.py.
    
    Args:
        models: List of PyTorch models
        model_weights: List of weights for each model (will be normalized)
        dataloader: DataLoader for test data
        use_tta: Whether to use test-time augmentation
        tta_config: TTA configuration (from tta.py)
        tta_merge_mode: How to merge TTA predictions
        ensemble_mode: How to ensemble models (currently uses model_weights)
    """
    for model in models:
        model.eval()
        model.to(DEVICE)

    # Normalize weights
    model_weights_arr = np.array(model_weights)
    model_weights_normalized = (model_weights_arr / model_weights_arr.sum()).tolist()
    logger.info(f"Model weights (normalized): {model_weights_normalized}")

    # Collect image IDs
    all_image_ids: List[str] = []
    
    if use_tta and tta_config != "none":
        # Use MultiModelTTA from tta.py
        logger.info(f"Using MultiModelTTA with config: {tta_config}, merge_mode: {tta_merge_mode}")
        logger.info(f"Number of TTA transforms: {len(TTA_CONFIGS.get(tta_config, []))}")
        
        multi_tta = MultiModelTTA(
            models=models,
            model_weights=model_weights_normalized,
            tta_config=tta_config,
            merge_mode=tta_merge_mode,
            device=DEVICE
        )
        
        # Collect predictions and image IDs
        all_preds = []
        for images, image_ids in tqdm(dataloader, desc=f"Ensemble TTA ({len(models)} models)"):
            images = images.to(DEVICE, non_blocking=True)
            images = images.to(memory_format=torch.channels_last)
            
            # Get ensemble + TTA predictions
            batch_preds = []
            for predictor in multi_tta.predictors:
                batch_pred = predictor.predict_batch(images)
                batch_preds.append(batch_pred.cpu().numpy())
            
            # Weighted average across models
            weighted_pred = np.zeros_like(batch_preds[0])
            for pred, weight in zip(batch_preds, model_weights_normalized):
                weighted_pred += weight * pred
            
            all_preds.append(weighted_pred)
            all_image_ids.extend(list(image_ids))
        
        preds = np.vstack(all_preds) if len(all_preds) else np.zeros((0, NUM_CLASSES), dtype=np.float32)
        
    else:
        # No TTA - simple ensemble
        logger.info("Running ensemble without TTA")
        all_preds_per_model: List[List[np.ndarray]] = [[] for _ in models]
        
        for images, image_ids in tqdm(dataloader, desc=f"Ensemble Inference ({len(models)} models)"):
            images = images.to(DEVICE, non_blocking=True)
            images = images.to(memory_format=torch.channels_last)

            # Get predictions from each model
            for i, model in enumerate(models):
                with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                all_preds_per_model[i].append(probs.detach().cpu().numpy())

            all_image_ids.extend(list(image_ids))

        # Combine predictions from each model
        preds_per_model = [
            np.vstack(preds) if len(preds) else np.zeros((0, NUM_CLASSES), dtype=np.float32)
            for preds in all_preds_per_model
        ]

        # Weighted ensemble
        preds = np.average(preds_per_model, axis=0, weights=model_weights_normalized)

    logger.info(f"Ensemble predictions shape: {preds.shape}")
    if preds.shape[0] > 0:
        logger.info(f"  Normal (col 0) mean: {preds[:, 0].mean():.4f}, >0.5: {(preds[:, 0] > 0.5).sum()}")
        logger.info(f"  All classes mean: {preds.mean():.4f}")

    return preds, all_image_ids


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble Inference for Task 1")

    parser.add_argument("--checkpoints", type=str, nargs="+", required=True, help="Paths to model checkpoints")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="Weights for each model (default: equal weights)")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV filename")

    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE, help=f"Image size (default: {DEFAULT_IMAGE_SIZE})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help=f"Num workers (default: {DEFAULT_NUM_WORKERS})")

    parser.add_argument("--use_tta", action="store_true", help="Use Test-Time Augmentation")
    parser.add_argument("--tta_config", type=str, default="flip_scale", 
                        choices=["none", "flip", "light", "medium", "heavy", "flip_scale"], 
                        help="TTA configuration (from tta.py)")
    parser.add_argument("--tta_merge_mode", type=str, default="mean", 
                        choices=["mean", "max", "gmean", "median"], 
                        help="How to merge TTA predictions")

    parser.add_argument("--ensemble_mode", type=str, default="weighted_mean", choices=["mean", "weighted_mean", "max"], help="How to ensemble models")
    parser.add_argument("--postprocess", type=str, default="normal_gating", choices=["none", "normal_gating"], help="Post-process mode")
    parser.add_argument("--normal_alpha", type=float, default=0.5, help="Alpha for normal gating")
    parser.add_argument("--clip_eps", type=float, default=1e-6, help="Epsilon for clipping probs")
    parser.add_argument("--image_col", type=str, default="ImageID", help="Column name for image id")

    args = parser.parse_args()

    # Validate
    if args.weights is not None:
        if len(args.weights) != len(args.checkpoints):
            logger.error(f"Number of weights ({len(args.weights)}) must match number of checkpoints ({len(args.checkpoints)})")
            sys.exit(1)
    else:
        args.weights = [1.0] * len(args.checkpoints)

    seed_everything(RANDOM_SEED)

    logger.info("=" * 90)
    logger.info("ENSEMBLE INFERENCE - Task 1")
    logger.info("=" * 90)
    logger.info(f"Device: {DEVICE} | AMP: {AMP_ENABLED}")
    logger.info(f"Number of models: {len(args.checkpoints)}")
    for i, ckpt in enumerate(args.checkpoints):
        logger.info(f"  Model {i+1}: {ckpt} (weight: {args.weights[i]:.2f})")
    logger.info(f"Test CSV: {args.test_csv}")
    logger.info(f"Image dir: {args.image_dir}")
    logger.info(f"Batch size: {args.batch_size} | Workers: {args.num_workers}")
    logger.info(f"TTA: {args.use_tta} | config: {args.tta_config} | merge: {args.tta_merge_mode}")
    logger.info(f"Ensemble mode: {args.ensemble_mode}")
    logger.info(f"Postprocess: {args.postprocess} | normal_alpha: {args.normal_alpha}")

    # Validate checkpoints
    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            logger.error(f"Checkpoint not found: {ckpt}")
            sys.exit(1)

    # Speed knobs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load models
    logger.info("Loading models...")
    models = []
    for i, ckpt_path in enumerate(args.checkpoints):
        logger.info(f"\n[Model {i+1}/{len(args.checkpoints)}]")
        model = load_model(ckpt_path, num_classes=NUM_CLASSES)
        models.append(model)

    # Load test data
    logger.info(f"\nLoading test data: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    logger.info(f"Test samples: {len(test_df)}")

    test_dataset = CXRDataset(
        test_df,
        image_dir=args.image_dir,
        image_size=args.image_size,
        image_col=args.image_col,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Run ensemble inference
    logger.info("\nRunning ensemble inference...")
    preds, image_ids = predict_ensemble(
        models=models,
        model_weights=args.weights,
        dataloader=test_loader,
        use_tta=args.use_tta,
        tta_config=args.tta_config,
        tta_merge_mode=args.tta_merge_mode,
        ensemble_mode=args.ensemble_mode,
    )

    if preds.shape[0] == 0:
        logger.error("No predictions generated. Check your test_csv and image_dir.")
        sys.exit(1)

    logger.info(f"Raw ensemble predictions: [{preds.min():.6f}, {preds.max():.6f}] | mean: {preds.mean():.6f}")

    # Post-process
    preds_pp = apply_postprocess(
        preds,
        mode=args.postprocess,
        eps=args.clip_eps,
        normal_alpha=args.normal_alpha,
    )

    logger.info(f"Post-processed predictions: [{preds_pp.min():.6f}, {preds_pp.max():.6f}] | mean: {preds_pp.mean():.6f}")
    logger.info(f"Normal mean (after): {preds_pp[:, 0].mean():.4f} | >0.5: {(preds_pp[:, 0] > 0.5).sum()}")

    # Build submission
    submission_df = pd.DataFrame({"ImageID": image_ids})
    for i, name in enumerate(CLASS_NAMES):
        submission_df[name] = preds_pp[:, i]

    submission_df.to_csv(args.output, index=False)
    logger.info(f"\n✓ Saved ensemble submission: {args.output}")
    logger.info(f"  Shape: {submission_df.shape}")
    logger.info(f"  Columns: {submission_df.columns[:6].tolist()} ...")

    logger.info("\nSample (first 3 rows, first 5 classes):")
    print(submission_df.iloc[:3, :6].to_string(index=False))

    logger.info("=" * 90)
    logger.info("Done.")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()

