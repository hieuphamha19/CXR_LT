"""
CXR-LT 2026 Task 1 — ConvNeXtV2 + CSRA DB-CAS (timm-compatible).

Architecture:
  ConvNeXtV2-Base (spatial features, global_pool="")
    → BatchNorm2D
    → CSRA head (Class-Specific Residual Attention, λ=0.1)

Usage:
    from huggingface_hub import hf_hub_download
    import importlib.util, sys

    path = hf_hub_download("hieuphamha/cxrlt2026-task1-convnextv2", "model.py")
    spec = importlib.util.spec_from_file_location("cxrlt", path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["cxrlt"] = mod
    spec.loader.exec_module(mod)

    import timm
    model = timm.create_model("cxrlt2026_task1_csra_dbcas", pretrained=True)
"""

import torch
import torch.nn as nn
import timm
from timm.models import register_model

NUM_CLASSES = 30
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


class CSRA(nn.Module):
    """Class-Specific Residual Attention (λ=0.1)."""
    def __init__(self, input_dim: int, num_classes: int, lam: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lam         = lam
        self.classifier  = nn.Linear(input_dim, num_classes)
        self.conv_att    = nn.Conv2d(input_dim, num_classes, kernel_size=1, bias=False)
        self.softmax     = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        logit_gap  = self.classifier(x.mean(dim=(2, 3)))
        att_score  = self.softmax(self.conv_att(x).view(b, self.num_classes, h * w))
        csra_feat  = torch.bmm(att_score, x.view(b, c, h * w).permute(0, 2, 1))
        logit_csra = (csra_feat * self.classifier.weight.unsqueeze(0)).sum(2) + self.classifier.bias
        return logit_gap + self.lam * logit_csra


class ConvNeXtV2CXR(nn.Module):
    """ConvNeXtV2-Base + BatchNorm2D + CSRA head (87.8M params)."""
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained_cfg=None, pretrained_cfg_overlay=None):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_base",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.2,
            global_pool="",        # returns (B, 1024, H, W)
        )
        nf        = self.backbone.num_features   # 1024
        self.bn   = nn.BatchNorm2d(nf)
        self.head = CSRA(input_dim=nf, num_classes=num_classes, lam=0.1)

    def forward(self, x):
        x = self.backbone(x)   # (B, 1024, H, W)
        x = self.bn(x)
        return self.head(x)    # (B, num_classes)

    @property
    def class_names(self):
        return CLASS_NAMES


@register_model
def cxrlt2026_task1_csra_dbcas(pretrained=False, **kwargs):
    """
    ConvNeXtV2-Base + CSRA head. Top-1 CXR-LT 2026 Task 1.

    Training pipeline:
      Stage 1 — MIMIC-CXR (14 classes), FC/MLP head:
        ConvNeXtV2-Base (ImageNet-22k) → head warm-up → full fine-tune
        Loss: AsymmetricLoss + LogitAdjustment
      Stage 2 — PadChest (30 classes), FC head replaced with CSRA + DB-CAS:
        Backbone resumed from Stage 1; new CSRA head initialized
        Double-Balance sampling + Class Activation Spatial strategy
        Loss: AsymmetricLoss, EMA decay 0.9999
    """
    model = ConvNeXtV2CXR(num_classes=NUM_CLASSES, **kwargs)
    if pretrained:
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            "hieuphamha/cxrlt2026-task1-convnextv2",
            "convnextv2_base_mimic-cxr_padchest_csra_dbcas.safetensors",
        )
        model.load_state_dict(load_file(path), strict=True)
    return model
