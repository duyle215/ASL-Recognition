import torch
import torch.nn as nn
from torchvision import models


class CustomFCHead(nn.Module):
    def __init__(self, in_features=2048, num_classes=5, dropout=0.5):
        super(CustomFCHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ResNet50 đã bao gồm AdaptiveAvgPool2d, nên x đã là (batch_size, in_features)
        return self.classifier(x)
