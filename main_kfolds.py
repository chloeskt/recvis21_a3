import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from src.data import data_transforms
from src.model import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#### PARAMETERS ####
batch_size = 16
epochs = 50
seed = 0
lr = 0.001
momentum = 0.9
weight_decay = 3e-4
k_folds = 5
torch.manual_seed(seed)


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def train_epoch(model, device, dataloader, loss_fn, optimizer, lr_scheduler):
    train_loss, train_correct = 0.0, 0
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_scheduler.step()
        train_loss += loss.item() * images.size(0)

        m = nn.Softmax(dim=1)
        probs = m(output)
        preds_classes = probs.max(1, keepdim=True)[1]
        train_correct += preds_classes.eq(labels.data.view_as(preds_classes)).sum()

        # scores, predictions = torch.max(output.data, 1)
        # train_correct += (predictions == labels).sum().item()

    return train_loss, train_correct


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)

        m = nn.Softmax(dim=1)
        probs = m(output)
        preds_classes = probs.max(1, keepdim=True)[1]
        val_correct += preds_classes.eq(labels.data.view_as(preds_classes)).sum()

    return valid_loss, val_correct


if __name__ == "__main__":
    model_name = "Net: Inceptionv3 and ResNet152"

    results = {}

    train_dataset = datasets.ImageFolder(
        "../299_cropped_bird_dataset/train_images", transform=data_transforms["train"]
    )
    val_dataset = datasets.ImageFolder(
        "../299_cropped_bird_dataset/val_images", transform=data_transforms["val"]
    )

    dataset = ConcatDataset([train_dataset, val_dataset])

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    foldperf = {}
    criterion = torch.nn.CrossEntropyLoss()

    print("##############################################")
    print("Start for model ", model_name)
    print("##############################################")
    print("\n")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        print("##############################################")
        print("Fold {}".format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        model = Net()
        reset_weights(model)
        model.to(device)

        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, verbose=True)

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

            lr_scheduler.step()

            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                    epoch + 1, epochs, train_loss, test_loss, train_acc, test_acc
                )
            )
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            if test_acc >= 91:
                torch.save(
                    model, f"../experiment/{model_name}_fold_{fold}_epoch_{epoch}.pt"
                )
                print(
                    f"save model at ../experiment/{model_name}_fold_{fold}_epoch_{epoch}.pt"
                )

        foldperf["fold{}".format(fold + 1)] = history

        torch.save(model, f"../experiment/{model_name}_fold_{fold}.pt")
        print(f"save model at ../experiment/{model_name}_fold_{fold}.pt")

    testl_f, tl_f, testa_f, ta_f = [], [], [], []
    k = k_folds

    for fold, inner_dict in foldperf.items():
        for key, value in inner_dict.items():
            if key == "train_acc" or key == "test_acc":
                new_list = [x.cpu() for x in inner_dict[key]]
                inner_dict[key] = new_list

    for f in range(1, k + 1):
        tl_f.append(np.mean(foldperf["fold{}".format(f)]["train_loss"]))
        testl_f.append(np.mean(foldperf["fold{}".format(f)]["test_loss"]))

        ta_f.append(np.mean(foldperf["fold{}".format(f)]["train_acc"]))
        testa_f.append(np.mean(foldperf["fold{}".format(f)]["test_acc"]))

    print("Performance of {} fold cross validation".format(k))
    print(
        "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(
            np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f)
        )
    )

    diz_ep = {
        "train_loss_ep": [],
        "test_loss_ep": [],
        "train_acc_ep": [],
        "test_acc_ep": [],
    }

    for i in range(epochs):
        diz_ep["train_loss_ep"].append(
            np.mean(
                [foldperf["fold{}".format(f + 1)]["train_loss"][i] for f in range(k)]
            )
        )
        diz_ep["test_loss_ep"].append(
            np.mean(
                [foldperf["fold{}".format(f + 1)]["test_loss"][i] for f in range(k)]
            )
        )
        diz_ep["train_acc_ep"].append(
            np.mean(
                [foldperf["fold{}".format(f + 1)]["train_acc"][i] for f in range(k)]
            )
        )
        diz_ep["test_acc_ep"].append(
            np.mean([foldperf["fold{}".format(f + 1)]["test_acc"][i] for f in range(k)])
        )

    # Plot losses
    plt.figure(figsize=(10, 8))
    plt.semilogy(diz_ep["train_loss_ep"], label="Train")
    plt.semilogy(diz_ep["test_loss_ep"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.grid()
    plt.legend()
    plt.title("CNN loss")
    plt.show()

    # Plot accuracies
    plt.figure(figsize=(10, 8))
    plt.semilogy(diz_ep["train_acc_ep"], label="Train")
    plt.semilogy(diz_ep["test_acc_ep"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.grid()
    plt.legend()
    plt.title("CNN accuracy")
    plt.show()
