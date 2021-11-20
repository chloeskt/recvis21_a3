import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
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
            # transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(degrees=(-45, 45)),
            # transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
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

# Start print
print("--------------------------------")

# K-fold Cross Validation model evaluation
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    # Print
    print(f"FOLD {fold}")
    print("--------------------------------")

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_subsampler
    )
    valloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=val_subsampler
    )

    # Init the neural network
    model = Net()
    model.to(device)
    model.apply(reset_weights)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = torch.nn.CrossEntropyLoss()
    # Run the training loop for defined number of epochs

    model.train()
    for epoch in range(0, epochs):

        # Print epoch
        print(f"Starting epoch {epoch + 1}")

        # Iterate over the DataLoader for training data
        for batch_idx, (data, labels) in enumerate(trainloader, 0):

            # Get inputs
            data, labels = data.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(data)

            # Compute loss
            loss = criterion(outputs, labels)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            lr_scheduler.step()

            # Print statistics
            current_loss = loss.item()

            if batch_idx % 10 == 0:
                print(
                    "[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        batch_idx * len(data),
                        len(trainloader.dataset),
                        100.0 * batch_idx / len(trainloader),
                        loss.data.item(),
                    )
                )

    # Process is complete.
    print("Training process has finished. Saving trained model.")

    # Print about validation
    print("Starting validating")

    # Saving the model
    save_path = f"../experiment/model-fold-{fold}.pth"
    torch.save(model.state_dict(), save_path)

    # Evaluation for this fold
    correct, total = 0, 0
    validation_loss = 0
    model.eval()
    with torch.no_grad():

        # Iterate over the test data and generate predictions
        for i, (data, labels) in enumerate(valloader):
            # Get inputs
            data, labels = data.to(device), labels.to(device)

            # Generate outputs
            outputs = model(data)

            validation_loss += criterion(outputs, labels).data.item()

            # Set total and correct
            m = nn.Softmax(dim=1)
            probs = m(outputs)
            preds_classes = probs.max(1, keepdim=True)[1]
            correct += preds_classes.eq(labels.data.view_as(preds_classes)).sum()
            total += labels.size(0)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(valloader.dataset),
            100.0 * correct / len(valloader.dataset),
        )
    )
    # Print accuracy
    print("Accuracy for fold %d: %d %%" % (fold, 100.0 * correct / total))
    print("--------------------------------")
    results[fold] = 100.0 * (correct / total)

    # Print fold results
print(f"K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS")
print("--------------------------------")
sum = 0.0
for key, value in results.items():
    print(f"Fold {key}: {value} %")
    sum += value
print(f"Average: {sum / len(results.items())} %")
