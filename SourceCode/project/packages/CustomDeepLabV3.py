import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torchvision.models._utils import IntermediateLayerGetter

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            *[ASPPConv(in_channels, out_channels, rate) for rate in atrous_rates],
            ASPPPooling(in_channels, out_channels)
        ])

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        x = torch.cat(res, dim=1)
        return self.project(x)

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__(
            ASPP(in_channels, 256, atrous_rates=[12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, replace_stride_with_dilation=[False, True, True]):
        super().__init__()
        resnet = resnet50(pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)
        return_layers = {"layer4": "out", "layer3": "aux"}
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)

    def forward(self, x):
        return self.body(x)

class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes=21, pretrained_backbone=True):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        self.classifier = DeepLabHead(2048, num_classes)
        self.aux_classifier = FCNHead(1024, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features['out'])
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        result = {'out': x}

        if self.training:
            aux = self.aux_classifier(features['aux'])
            aux = F.interpolate(aux, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = aux

        return result