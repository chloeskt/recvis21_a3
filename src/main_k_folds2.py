import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
from torchvision import datasets, transforms
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#### PARAMETERS ####
batch_size = 12
epochs = 50
seed = 0
lr = 0.001
momentum = 0.9
torch.manual_seed(seed)

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        "../299_cropped_bird_dataset/test_images", transform=data_transforms["test"]
    ),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


class Net(nn.Module):
    def __init__(self, num_classes=20):
        super(Net, self).__init__()

        self.res = models.resnet152(pretrained=True)
        self.inc = models.inception_v3(pretrained=True)

        for param in self.inc.parameters():
            param.requires_grad = True
        self.inc.aux_logits = False
        num_features = self.inc.fc.in_features
        self.inc.fc = nn.Linear(num_features, 512)

        for param in self.res.conv1.parameters():
            param.requires_grad = True
        for param in self.res.bn1.parameters():
            param.requires_grad = True
        for param in self.res.layer1.parameters():
            param.requires_grad = True
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


# Configuration options
k_folds = 5

# For fold results
results = {}

train_dataset = datasets.ImageFolder(
    "../299_cropped_bird_dataset/train_images", transform=data_transforms["train"]
)
val_dataset = datasets.ImageFolder(
    "../299_cropped_bird_dataset/val_images", transform=data_transforms["val"]
)

dataset = ConcatDataset([train_dataset, val_dataset])

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

foldperf = {}

criterion = torch.nn.CrossEntropyLoss()


def train_epoch(model, device, dataloader, loss_fn, optimizer, lr_scheduler):
    train_loss, train_correct = 0.0, 0
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss, train_correct


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct


for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):

    print("Fold {}".format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss, train_correct = train_epoch(
            model, device, train_loader, criterion, optimizer, lr_scheduler
        )
        test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print(
            "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                epoch + 1, epochs, train_loss, test_loss, train_acc, test_acc
            )
        )
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

    foldperf["fold{}".format(fold + 1)] = history

torch.save(model, "../experiment/k_cross_CNN.pt")
