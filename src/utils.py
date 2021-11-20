import cv2
import matplotlib.pyplot as plt
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from torchvision.utils import make_grid


def show_images(train_dataloader):
    for image, label in train_dataloader:
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(
            make_grid(image[: train_dataloader.batch_size], nrow=8).permute(1, 2, 0)
        )
        break


def show_images_with_labels(train_dataloader):
    fig = plt.figure(figsize=(15, 15))
    for images, labels in train_dataloader:
        for i, (image, label) in enumerate(zip(images, labels)):
            ax = fig.add_subplot(8, 4, i + 1, xticks=[], yticks=[])
            plt.imshow(image.permute(1, 2, 0))
            ax.set_title(f"Label: {label}")
            if i == 20:
                return
        break


def visualize_model(model, val_dataloader, device, num_images=12, figsize=(15, 15)):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=figsize)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 4, images_so_far)
                ax.axis("off")
                ax.set_title("predicted: {} | true: {}".format(preds[j], labels[j]))
                plt.imshow(inputs.cpu().data[j].permute(1, 2, 0))

                if images_so_far == num_images:
                    return


def visualize_bounding_boxes(model, images_path, cfg, figsize=(15, 15)):
    num_images = len(images_path)
    fig = plt.figure(figsize=figsize)
    for i, path in enumerate(images_path):
        im = cv2.imread(path)
        outputs = model(im)
        v = Visualizer(
            im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        ax = plt.subplot(4, num_images // 2, i + 1)
        ax.axis("off")
        ax.set_title("Bounding box detected")
        plt.imshow(out.get_image()[:, :, ::-1])
