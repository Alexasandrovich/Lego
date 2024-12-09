import torch
import torch.nn as nn
import torchvision.models as models


class LegoClassifierModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(LegoClassifierModel, self).__init__()
        # Загружаем предобученный ResNet
        self.backbone = models.resnet18(pretrained=pretrained)
        # Заменяем первый слой: был Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Сделаем Conv2d(1, 64, ...)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None
        )

        # Заменяем последний слой классификатора
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)