import argparse
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

# TODO: setup Tensorboard

# Training settings
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
    "--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
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
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
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
from src.data import data_transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from src.model import Net

model = Net()
if use_cuda:
    print("Using GPU")
    model.cuda()
else:
    print("Using CPU")

# TODO: use another optimizer such as Adam and changes the parameters
# Add weight decay, early stopping, LRScheduler
# Add gradient clipping

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=args.lr,
#             weight_decay=0.2,
#         )

# Decay LR by a factor of 0.1 every 10 epochs
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=args.scheduler_step, gamma=args.gamma
)


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        # sum up batch loss
        train_loss += loss.data.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
            # print(
            #     "Train current overall loss: ",
            #     train_loss / len(train_loader.dataset),
            #     "\n",
            #     "Train current accuracy: ",
            #     100.0 * correct / len(train_loader.dataset),
            # )


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = args.experiment + "/model_" + str(epoch) + ".pth"
    torch.save(model.state_dict(), model_file)
    print(
        "Saved model to "
        + model_file
        + ". You can run `python evaluate.py --model "
        + model_file
        + "` to generate the Kaggle formatted csv file\n"
    )
