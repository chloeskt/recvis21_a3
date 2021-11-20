import torch
from torch import nn
from torchvision import models

from src.CustomInceptionv3 import CustomInceptionv3
from src.CustomResNet import CustomResNet


class CustomInceptionv3ResNet152(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(CustomInceptionv3ResNet152, self).__init__()

        self.pretrained = pretrained

        self.res = models.resnet152(pretrained=self.pretrained)
        self.inc = models.inception_v3(pretrained=self.pretrained)

        for param in self.inc.parameters():
            param.requires_grad = self.pretrained
        self.inc.aux_logits = False
        num_features = self.inc.fc.in_features
        self.inc.fc = nn.Linear(num_features, 512)

        for param in self.res.conv1.parameters():
            param.requires_grad = self.pretrained
        for param in self.res.bn1.parameters():
            param.requires_grad = self.pretrained
        for param in self.res.layer1.parameters():
            param.requires_grad = self.pretrained
        for param in self.res.layer2.parameters():
            param.requires_grad = self.pretrained
        for param in self.res.layer3.parameters():
            param.requires_grad = self.pretrained

        self.res.avgpool = nn.AvgPool2d(10)
        num_features2 = self.res.fc.in_features
        self.res.fc = nn.Linear(num_features2, 512)

        lin3 = nn.Linear(1024, num_classes)
        self.fc = lin3

    def forward(self, input):
        x1 = self.res(input)
        x2 = self.inc(input)
        x = torch.cat((x1, x2), 1)
        return self.fc(x)


