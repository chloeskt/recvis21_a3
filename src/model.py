import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 20


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = True
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(
            in_features=num_features, out_features=nclasses, bias=True
        )

    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
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

        lin3 = nn.Linear(1024, num_classes)
        self.fc = lin3

    def forward(self, input):
        x1 = self.res(input)
        x2 = self.inc(input)
        x = torch.cat((x1, x2), 1)
        return self.fc(x)


class Inceptionv3(nn.Module):
    def __init__(self, num_classes=20):
        super(Inceptionv3, self).__init__()

        self.inc = models.inception_v3(pretrained=True)

        for param in self.inc.parameters():
            param.requires_grad = True

        self.inc.aux_logits = False
        num_features = self.inc.fc.in_features
        self.inc.fc = nn.Linear(num_features, 1024)
        lin3 = nn.Linear(1024, num_classes)
        self.fc = lin3

    def forward(self, input):
        x = self.inc(input)
        return self.fc(x)
