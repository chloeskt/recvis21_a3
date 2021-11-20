from torch import nn
from torchvision import models


class CustomResNet:
    def __init__(self, transform_input=False):
        self.transform_input = transform_input

        # CUSTOM RESNET152 PART
        self.res = models.resnet152(pretrained=False)

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        self.res = self._slice_model(to_layer=-1)
        return self.res(x)

    def _slice_model(self, from_layer=None, to_layer=None):
        return nn.Sequential(*list(self.res.children())[from_layer:to_layer])
