import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#### PARAMETERS ####
batch_size = 8
epochs = 100
seed = 0
lr = 0.001
momentum = 0.9
torch.manual_seed(seed)

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            # transforms.transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            # transforms.transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(degrees=(-90, 90)),
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

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        "../299_cropped_bird_dataset/train_images", transform=data_transforms["train"]
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        "../299_cropped_bird_dataset/val_images", transform=data_transforms["val"]
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        "../299_cropped_bird_dataset/test_images", transform=data_transforms["test"]
    ),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)


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

        # self.eff = models.efficientnet_b7(pretrained=True)

        # for param in self.eff.parameters():
        #    param.requires_grad = True

        # num_features2 = self.eff.classifier[1].in_features

        # self.eff.classifier = nn.Sequential(
        #    nn.Dropout(p=0.4, inplace=True),
        #    nn.Linear(num_features2, 512),
        # )

        lin3 = nn.Linear(1024, num_classes)
        self.fc = lin3

    def forward(self, input):
        x1 = self.res(input)
        x2 = self.inc(input)
        x = torch.cat((x1, x2), 1)
        return self.fc(x)


model = Net()
model.to(device)

optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=momentum,
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
criterion = torch.nn.CrossEntropyLoss()


def train(model, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward
        preds = model(data)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if batch_idx % 10 == 0:
            print(
                "[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )


def validation(model):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            preds = model(data)
            # sum up batch loss
            validation_loss += criterion(preds, labels).data.item()
            m = nn.Softmax(dim=1)
            probs = m(preds)
            preds_classes = probs.max(1, keepdim=True)[1]
            correct += preds_classes.eq(labels.data.view_as(preds_classes)).sum()
        validation_loss /= len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return 100.0 * correct / len(val_loader.dataset)


for epoch in range(1, epochs + 1):
    print("################################################# EPOCH", epoch)
    train(model, epoch)
    val_acc = validation(model)
    if val_acc >= 93:
        model_file = "../experiments" + "/model_" + str(val_acc) + ".pth"
        torch.save(model.state_dict(), model_file)
