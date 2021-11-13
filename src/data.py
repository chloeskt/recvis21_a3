import pickle
import zipfile
import os

import torch
import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
from PIL.Image import Image
from torchvision import datasets

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomGrayscale(p=0.4),
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [
                        transforms.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.1, hue=0.4
                        ),
                        # transforms.RandomCrop(size=(32, 32)),
                        # # transforms.RandomVerticalFlip(p=0.5),
                        # transforms.Grayscale(num_output_channels=3),
                    ]
                ),
                p=0.5,
            ),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # TODO: add data augmentation
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
