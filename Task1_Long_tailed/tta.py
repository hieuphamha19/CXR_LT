import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Callable, Optional, Tuple
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TTATransforms:
    """Collection of TTA transforms for CXR images."""
    
    @staticmethod
    def identity(x: torch.Tensor) -> torch.Tensor:
        """Original image, no transformation."""
        return x
    
    @staticmethod
    def horizontal_flip(x: torch.Tensor) -> torch.Tensor:
        """Flip horizontally."""
        return torch.flip(x, dims=[3])
    
    @staticmethod
    def rotate_5(x: torch.Tensor) -> torch.Tensor:
        """Rotate 5 degrees clockwise."""
        angle = torch.tensor([5.0 * np.pi / 180])
        return TTATransforms._rotate_tensor(x, angle)
    
    @staticmethod
    def rotate_neg5(x: torch.Tensor) -> torch.Tensor:
        """Rotate 5 degrees counter-clockwise."""
        angle = torch.tensor([-5.0 * np.pi / 180])
        return TTATransforms._rotate_tensor(x, angle)
    
    @staticmethod
    def scale_up(x: torch.Tensor, factor: float = 1.1) -> torch.Tensor:
        """Scale up (zoom in) by factor."""
        _, _, h, w = x.shape
        new_h, new_w = int(h * factor), int(w * factor)
        scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # Center crop back to original size
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return scaled[:, :, start_h:start_h+h, start_w:start_w+w]
    
    @staticmethod
    def scale_down(x: torch.Tensor, factor: float = 0.9) -> torch.Tensor:
        """Scale down (zoom out) by factor."""
        _, _, h, w = x.shape
        new_h, new_w = int(h * factor), int(w * factor)
        scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # Pad back to original size
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        return F.pad(scaled, (pad_w, w - new_w - pad_w, pad_h, h - new_h - pad_h), mode='reflect')
    
    @staticmethod
    def brightness_up(x: torch.Tensor, factor: float = 1.1) -> torch.Tensor:
        """Increase brightness."""
        return torch.clamp(x * factor, 0, 1)
    
    @staticmethod
    def brightness_down(x: torch.Tensor, factor: float = 0.9) -> torch.Tensor:
        """Decrease brightness."""
        return x * factor
    
    @staticmethod
    def _rotate_tensor(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rotate tensor by angle (in radians)."""
        # Create rotation matrix
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Affine transformation matrix
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)
        
        theta = theta.expand(x.size(0), -1, -1)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


# Predefined TTA configurations
TTA_CONFIGS = {
    "none": [
        TTATransforms.identity
    ],
    "flip": [
        TTATransforms.identity,
        TTATransforms.horizontal_flip
    ],
    "light": [
        TTATransforms.identity,
        TTATransforms.horizontal_flip,
        TTATransforms.rotate_5,
        TTATransforms.rotate_neg5
    ],
    "medium": [
        TTATransforms.identity,
        TTATransforms.horizontal_flip,
        TTATransforms.rotate_5,
        TTATransforms.rotate_neg5,
        lambda x: TTATransforms.scale_up(x, 1.1),
        lambda x: TTATransforms.scale_down(x, 0.9),
    ],
    "heavy": [
        TTATransforms.identity,
        TTATransforms.horizontal_flip,
        TTATransforms.rotate_5,
        TTATransforms.rotate_neg5,
        lambda x: TTATransforms.scale_up(x, 1.1),
        lambda x: TTATransforms.scale_down(x, 0.9),
        lambda x: TTATransforms.brightness_up(x, 1.1),
        lambda x: TTATransforms.brightness_down(x, 0.9),
    ],
    "flip_scale": [
        TTATransforms.identity,
        TTATransforms.horizontal_flip,
        lambda x: TTATransforms.scale_up(x, 1.05),
        lambda x: TTATransforms.scale_down(x, 0.95),
        lambda x: TTATransforms.horizontal_flip(TTATransforms.scale_up(x, 1.05)),
        lambda x: TTATransforms.horizontal_flip(TTATransforms.scale_down(x, 0.95)),
    ],
}


class TTAPredictor:
    """
    Test-Time Augmentation predictor.
    
    Applies multiple transforms and aggregates predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        transforms: List[Callable] = None,
        tta_config: str = "flip",
        merge_mode: str = "mean",
        device: torch.device = None
    ):
        """
        Args:
            model: Trained model
            transforms: List of transform functions (overrides tta_config)
            tta_config: Predefined config name ("none", "flip", "light", "medium", "heavy")
            merge_mode: How to merge predictions ("mean", "max", "gmean", "median", "logit_mean")
            device: Device to run on
        """
        self.model = model
        self.model.eval()
        
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = TTA_CONFIGS.get(tta_config, TTA_CONFIGS["flip"])
        
        self.merge_mode = merge_mode
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict with TTA for a batch of images.
        
        Args:
            images: Batch of images [B, C, H, W]
        
        Returns:
            Aggregated predictions [B, num_classes]
        """
        images = images.to(self.device)
        all_preds = []
        all_logits = []
        
        with torch.no_grad():
            for transform in self.transforms:
                augmented = transform(images)
                logits = self.model(augmented)
                probs = torch.sigmoid(logits)
                all_logits.append(logits)
                all_preds.append(probs)
        
        # Stack: [num_transforms, B, num_classes]
        stacked = torch.stack(all_preds, dim=0)
        
        # Merge predictions
        if self.merge_mode == "mean":
            return stacked.mean(dim=0)
        elif self.merge_mode == "max":
            return stacked.max(dim=0)[0]
        elif self.merge_mode == "gmean":
            # Geometric mean
            log_preds = torch.log(stacked + 1e-8)
            return torch.exp(log_preds.mean(dim=0))
        elif self.merge_mode == "median":
            return stacked.median(dim=0)[0]
        elif self.merge_mode == "logit_mean":
            stacked_logits = torch.stack(all_logits, dim=0)
            return torch.sigmoid(stacked_logits.mean(dim=0))
        else:
            return stacked.mean(dim=0)
    
    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Run TTA prediction on entire dataset.
        
        Args:
            dataloader: Test dataloader
            show_progress: Whether to show progress bar
        
        Returns:
            Predictions array [N, num_classes]
        """
        all_predictions = []
        
        iterator = tqdm(dataloader, desc=f"TTA ({len(self.transforms)} transforms)") if show_progress else dataloader
        
        for batch in iterator:
            if isinstance(batch, dict):
                images = batch['image']
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            preds = self.predict_batch(images)
            all_predictions.append(preds.cpu().numpy())
        
        return np.concatenate(all_predictions, axis=0)


class MultiModelTTA:
    """
    TTA with multiple models (ensemble + TTA).
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        model_weights: List[float] = None,
        tta_config: str = "flip",
        merge_mode: str = "mean",
        device: torch.device = None
    ):
        self.models = models
        self.model_weights = model_weights or [1.0 / len(models)] * len(models)
        self.tta_config = tta_config
        self.merge_mode = merge_mode
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create TTA predictor for each model
        self.predictors = [
            TTAPredictor(model, tta_config=tta_config, merge_mode=merge_mode, device=self.device)
            for model in models
        ]
    
    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        show_progress: bool = True
    ) -> np.ndarray:
        """Run ensemble + TTA prediction."""
        all_model_preds = []
        
        for i, predictor in enumerate(self.predictors):
            if show_progress:
                print(f"Model {i+1}/{len(self.predictors)}")
            preds = predictor.predict(dataloader, show_progress=show_progress)
            all_model_preds.append(preds)
        
        # Weighted average across models
        weighted_preds = np.zeros_like(all_model_preds[0])
        for preds, weight in zip(all_model_preds, self.model_weights):
            weighted_preds += weight * preds
        
        return weighted_preds


def tta_predict(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    tta_config: str = "flip",
    merge_mode: str = "mean"
) -> np.ndarray:
    """
    Convenience function for TTA prediction.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to use
        tta_config: TTA configuration ("none", "flip", "light", "medium", "heavy")
        merge_mode: How to merge predictions ("mean", "max", "gmean", "median", "logit_mean")
    
    Returns:
        Predictions [N, num_classes]
    """
    predictor = TTAPredictor(
        model=model,
        tta_config=tta_config,
        merge_mode=merge_mode,
        device=device
    )
    return predictor.predict(dataloader)
