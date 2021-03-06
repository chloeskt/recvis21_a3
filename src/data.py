import torch
import torchvision.transforms as transforms
from torchvision import datasets

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [
                        transforms.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.1, hue=0.4
                        ),
                    ]
                ),
                p=0.2,
            ),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [
                        transforms.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.1, hue=0.4
                        ),
                    ]
                ),
                p=0.2,
            ),
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


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path
