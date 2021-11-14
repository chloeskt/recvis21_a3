import torch
import torchvision.transforms as transforms
from torchvision import datasets

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            # transforms.transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.1),
            transforms.RandomRotation(45),
            # transforms.transforms.Resize((64, 64)),
            # transforms.transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(p=0.3),
            # # transforms.RandomGrayscale(p=0.4),
            # transforms.RandomApply(
            #     torch.nn.ModuleList(
            #         [
            #             transforms.ColorJitter(
            #                 brightness=0.3, contrast=0.3, saturation=0.1, hue=0.4
            #             ),
            #             # transforms.RandomCrop(size=(32, 32)),
            #             # # transforms.RandomVerticalFlip(p=0.5),
            #             # transforms.Grayscale(num_output_channels=3),
            #         ]
            #     ),
            #     p=0.5,
            # ),
            # transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.1),
            transforms.RandomRotation(45),
            # transforms.transforms.Resize((224, 224)),
            # transforms.transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.transforms.Resize((224, 224)),
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
