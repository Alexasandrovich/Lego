import torch
import torch.nn as nn
import torchvision.models as models


class LegoClassifierModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(LegoClassifierModel, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        old_conv = self.backbone.conv1
        # Теперь 3 канала (по одному каналу на каждый кадр)
        self.backbone.conv1 = nn.Conv2d(
            3, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)