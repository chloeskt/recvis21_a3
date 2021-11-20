import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="RecVis A3 training script")
parser.add_argument(
    "--data",
    type=str,
    default="bird_dataset",
    metavar="D",
    help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,  # 64
    metavar="B",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.1)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--scheduler_step",
    type=int,
    default=2,
    metavar="S",
    help="Scheduler step (default: 10)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.1,
    metavar="S",
    help="Scheduler gamma (default: 0.1)",
)
parser.add_argument(
    "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--experiment",
    type=str,
    default="experiment",
    metavar="E",
    help="folder where experiment outputs are located.",
)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from src import data_transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        args.data + "/train_images", transform=data_transforms["train"]
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + "/val_images", transform=data_transforms["val"]),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from src import Net

model = Net()

if use_cuda:
    print("Using GPU")
    model.cuda()
else:
    print("Using CPU")

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

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


for epoch in range(1, args.epochs + 1):
    print("################################################# EPOCH", epoch)
    train(model, epoch)
    val_acc = validation(model)
    if val_acc >= 93:
        model_file = "../experiments" + "/model_" + str(val_acc) + ".pth"
        torch.save(model.state_dict(), model_file)
