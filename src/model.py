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
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = True
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(
            in_features=num_features, out_features=nclasses, bias=True
        )

    def forward(self, x):
        return self.model(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.efficientnet_b7(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = True

        self.num_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.num_features, nclasses),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, nclasses)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.cnn = models.vgg16(pretrained=True)
#         self.classifier = nn.Sequential(
#             nn.Linear(32000, 4096, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4),
#             nn.Linear(4096, 2048, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4),
#             nn.Linear(2048, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.4),
#             nn.Linear(1024, 20),
#         )
#
#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(-1, 32000)
#         x = self.classifier(x)
#         return x

# Model
class Classifier(nn.Module):
    def __init__(self, embedding_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes=20):
        super(Net, self).__init__()

        self.res = models.resnet152(pretrained=True)
        self.inc = models.inception_v3(pretrained=True)

        for param in self.inc.parameters():
            param.requires_grad = False
        self.inc.aux_logits = False
        num_features = self.inc.fc.in_features
        self.inc.fc = nn.Linear(num_features, 512)

        for param in self.res.conv1.parameters():
            param.requires_grad = False
        for param in self.res.bn1.parameters():
            param.requires_grad = False
        for param in self.res.layer1.parameters():
            param.requires_grad = False
        for param in self.res.layer2.parameters():
            param.requires_grad = True
        for param in self.res.layer3.parameters():
            param.requires_grad = True

        self.res.avgpool = nn.AvgPool2d(10)
        num_features2 = self.res.fc.in_features
        self.res.fc = nn.Linear(num_features2, 512)

        lin3 = nn.Linear(1024, nclasses)
        self.fc = lin3

    def forward(self, input):
        x1 = self.res(input)
        x2 = self.inc(input)
        x = torch.cat((x1, x2), 1)
        return self.fc(x)
