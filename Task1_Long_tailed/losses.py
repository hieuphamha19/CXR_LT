import torch
import torch.nn as nn


class BCEWithLogitsLossWeighted(nn.Module):
    """BCEWithLogitsLoss with optional positive weights."""
    
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification."""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean()


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (Cui et al., CVPR 2019)."""
    
    def __init__(self, class_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(beta, torch.tensor(class_counts, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer('weights', weights)
        
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        weights = self.weights.to(logits.device)
        cb_weight = targets * weights + (1 - targets)
        
        loss = cb_weight * focal_weight * bce_loss
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ridnik et al., ICCV 2021)."""
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping
        probs_pos = probs
        probs_neg = probs.clamp(max=1 - self.clip) if self.clip > 0 else probs
        
        # Basic BCE
        loss_pos = targets * torch.log(probs_pos + self.eps)
        loss_neg = (1 - targets) * torch.log(1 - probs_neg + self.eps)
        
        # Asymmetric focusing
        pt_pos = probs_pos
        pt_neg = 1 - probs_neg
        
        weight_pos = (1 - pt_pos) ** self.gamma_pos
        weight_neg = pt_neg ** self.gamma_neg
        
        loss = -weight_pos * loss_pos - weight_neg * loss_neg
        return loss.mean()


class LDAMLoss(nn.Module):
    """
    LDAM Loss for long-tailed recognition (Cao et al., NeurIPS 2019).
    Adapted for multi-label classification.
    Assigns larger margins to tail classes.
    """
    
    def __init__(self, class_counts, max_margin=0.5, scale=30.0):
        super().__init__()
        self.scale = scale
        
        # Calculate margins inversely proportional to class frequency
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        margins = 1.0 / torch.sqrt(torch.sqrt(class_counts))
        margins = margins * (max_margin / margins.max())
        self.register_buffer('margins', margins)
        
    def forward(self, logits, targets):
        # Apply margin to positive class logits
        margins = self.margins.to(logits.device)
        margin_logits = logits - targets * margins.unsqueeze(0)
        
        loss = nn.functional.binary_cross_entropy_with_logits(
            self.scale * margin_logits, targets
        )
        return loss


class DRWLoss(nn.Module):
    """
    Deferred Re-Weighting Loss for two-stage training (Cao et al., NeurIPS 2019).
    First stage: standard loss. Second stage: class-balanced weights.
    """
    
    def __init__(self, class_counts, drw_epoch=10, beta=0.9999):
        super().__init__()
        self.drw_epoch = drw_epoch
        self.current_epoch = 0
        
        # Calculate class-balanced weights
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer('cb_weights', weights)
        self.register_buffer('uniform_weights', torch.ones_like(weights))
        
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        
    def forward(self, logits, targets):
        if self.current_epoch < self.drw_epoch:
            weights = self.uniform_weights
        else:
            weights = self.cb_weights
            
        weights = weights.to(logits.device)
        
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Apply weights to positive samples
        weighted_loss = bce_loss * (targets * weights + (1 - targets))
        return weighted_loss.mean()


class LDAMDRWLoss(nn.Module):
    """
    LDAM + DRW combined loss.
    Best performing loss for long-tailed recognition.
    """
    
    def __init__(self, class_counts, max_margin=0.5, scale=30.0, drw_epoch=10, beta=0.9999):
        super().__init__()
        self.scale = scale
        self.drw_epoch = drw_epoch
        self.current_epoch = 0
        
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        
        # Margins for LDAM
        margins = 1.0 / torch.sqrt(torch.sqrt(class_counts))
        margins = margins * (max_margin / margins.max())
        self.register_buffer('margins', margins)
        
        # Weights for DRW
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer('cb_weights', weights)
        self.register_buffer('uniform_weights', torch.ones_like(weights))
        
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        
    def forward(self, logits, targets):
        margins = self.margins.to(logits.device)
        
        # Apply LDAM margins
        margin_logits = logits - targets * margins.unsqueeze(0)
        
        # DRW weights
        if self.current_epoch < self.drw_epoch:
            weights = self.uniform_weights
        else:
            weights = self.cb_weights
            
        weights = weights.to(logits.device)
        
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            self.scale * margin_logits, targets, reduction='none'
        )
        
        weighted_loss = bce_loss * (targets * weights + (1 - targets))
        return weighted_loss.mean()


