import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from src.data import data_transforms
from src.model import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#### PARAMETERS ####
batch_size = 8
epochs = 25
seed = 0
lr = 0.001
momentum = 0.9
torch.manual_seed(seed)

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

model = Net()
model.to(device)

optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=momentum,
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
criterion = torch.nn.CrossEntropyLoss()


def train(model):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
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


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        print(f"###################### EPOCH {epoch} ###########################")
        train(model)
        val_acc = validation(model)
        if val_acc >= 93:
            model_file = "../experiments" + "/model_" + str(val_acc) + ".pth"
            torch.save(model.state_dict(), model_file)
