"""
Step 9: non-equivariant control. An ordinary CNN with the same fiber width
as the p8m/p4m regular-representation models (8 fields x 8 group elements
= 64 channels), same input, same training data/procedure, output (q1,q2)
with NO equivariance constraint of any kind. Used only to check whether
the physical typing + equivariance actually buys anything over a same-
capacity conventional network -- not a general architecture search.
"""
import torch
import torch.nn as nn


class PlainCNN(nn.Module):
    def __init__(self, in_channels=1, c1=64, c2=64):
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
        self.head = nn.Conv2d(c2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        f = self.features(x)
        q_field = self.head(f)
        q = q_field.mean(dim=(2, 3))  # same global-average-pool convention as NematicNet
        return q
