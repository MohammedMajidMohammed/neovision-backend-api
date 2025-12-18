# src/model.py
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class EmotionEfficientNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # استخدام weights الـ pretrained على ImageNet (3 قنوات RGB)
        weights = EfficientNet_B3_Weights.DEFAULT
        self.backbone = efficientnet_b3(weights=weights)

        # لا نغيّر أول Conv (تظل 3 قنوات)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
