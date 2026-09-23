"""
STAGE 9: PLAIN CNN BASELINE -- no equivariance, no physical typing.

NEW file; code_v2/baseline_cnn.py (Phase 1R's Q-only PlainCNN) is NOT
modified. Reuses the same backbone channel widths (c1=64, c2=64) as
Phase 1R's PlainCNN for a comparable non-equivariant capacity, extended
with a classification head so it can be trained with the same dual-loss
structure as Model 2/Model 3.
"""
import torch
import torch.nn as nn


class PlainCNNDualHead(nn.Module):
    def __init__(self, in_channels=3, c1=64, c2=64, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.q_head = nn.Conv2d(c2, 2, kernel_size=3, padding=1, bias=False)
        self.classifier_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c2, n_classes)

    def forward(self, x):
        f = self.features(x)
        q_dense = self.q_head(f)  # (B, 2, H/2, W/2) -- no equivariance constraint of any kind
        feat = self.classifier_pool(f).flatten(1)
        logits = self.classifier(feat)
        return logits, q_dense
