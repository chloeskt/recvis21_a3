import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 20

# TODO: clear visualisation of the images
# probably need to first detect the birds in the image
# take only cropped bird dataset ?

# TODO: try auto-encoder structure
# Add spatial batch norm
# Use pre-trained model e.g
# alexnet, VGG, ResNet, MobileNet
# https://pytorch.org/vision/stable/models.html#classification


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.efficientnet_b7 = models.efficientnet_b7(pretrained=True)
        self.num_features = self.efficientnet_b7.classifier[1].in_features
        self.efficientnet_b7.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.num_features, nclasses),
        )

        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, nclasses)

        # self.network = models.resnet18(pretrained=True)
        # number_of_features = self.network.fc.in_features
        # self.network.fc = nn.Linear(number_of_features, nclasses)

    def forward(self, x):
        # x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.leaky_relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.leaky_relu(F.max_pool2d(self.conv3(x), 2))
        # x = x.view(-1, 320)
        # x = F.leaky_relu(self.fc1(x))
        # return self.fc2(x)
        return self.efficientnet_b7(x)
