import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
# Some basic setup:
# Setup detectron2 logger
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.data import ImageFolderWithPaths

setup_logger()

import numpy as np
import os
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

BATCH_SIZE = 32
SOURCE_DATA_PATH = "../bird_dataset"
DEST_DATA_PATH = "../cropped_bird_dataset"


class BirdImageCropper:
    def __init__(self, source_data_path, dest_data_path, batch_size):
        self.source_data_path = source_data_path
        self.dest_data_path = dest_data_path
        self.batch_size = batch_size

    def _initiate_detector(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Detection Threshold
        cfg.MODEL.ROI_HEADS.NMS = 0.4  # Non Maximum Suppression Threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        return DefaultPredictor(cfg)

    def _check_if_directory_is_available(self):
        if not os.path.isdir(self.dest_data_path):
            print("Creating the folder for the cropped bird dataset")
            os.system("mkdir ../cropped_bird_dataset")

    def _check_reformat_image(self, img_source_path, img_dest_path):
        i = plt.imread(img_source_path)
        if len(i.shape) == 2 or i.shape[2] != 3:
            i = Image.fromarray(i)
            i = i.convert("RGB")
            i.save(img_dest_path)

    def get_birds_images_cropped(self):
        self._check_if_directory_is_available()
        model = self._initiate_detector()

        for state in ["train", "val", "test"]:

            non_cropped_img = 0
            non_cropped_paths = []
            number_img = 0

            if state != "test":
                dataloader = DataLoader(
                    ImageFolderWithPaths(
                        os.path.join(self.source_data_path, state + "_images"),
                        transform=transforms.Compose(
                            [
                                transforms.Resize((299, 299)),
                                transforms.RandomHorizontalFlip(0.3),
                                transforms.RandomRotation(degrees=(-45, 45)),
                                transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                ),
                            ]
                        ),
                    ),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
            else:
                dataloader = DataLoader(
                    ImageFolderWithPaths(
                        os.path.join(self.source_data_path, state + "_images"),
                        transform=transforms.Compose(
                            [
                                transforms.Resize((299, 299)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                ),
                            ]
                        ),
                    ),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

            for images, labels, paths in dataloader:
                for img_path in paths:

                    dest_path = img_path.replace(
                        self.source_data_path, self.dest_data_path
                    )
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                    if os.path.isfile(dest_path):
                        continue

                    self._check_reformat_image(img_path, dest_path)

                    img = cv2.imread(img_path)
                    with torch.no_grad():
                        detections = model(img)["instances"]

                    number_img += 1
                    img = np.array(Image.open(img_path))
                    if len(detections.scores) > 0:
                        index_birds = np.where(
                            detections.pred_classes.cpu().numpy() == 14
                        )[
                            0
                        ]  # 14 = class number for bird
                        if len(index_birds) == 0:
                            non_cropped_img += 1
                            non_cropped_paths.append(img_path)

                            print("SAVING NON-CROPPED image at ", dest_path)
                            plt.imsave(
                                dest_path,
                                np.array(Image.fromarray(img)),
                                dpi=1000,
                            )
                            plt.close()
                            continue

                        bird = int(
                            torch.max(detections.scores[index_birds], 0)[1]
                                .cpu()
                                .numpy()
                        )

                        [x1, y1, x2, y2] = (
                            detections.pred_boxes[index_birds][bird]
                                .tensor[0]
                                .cpu()
                                .numpy()
                        )

                        x1, y1 = np.maximum(0, int(x1) - 20), np.maximum(
                            0, int(y1) - 20
                        )
                        x2, y2 = np.minimum(x2 + 40, img.shape[1]), np.minimum(
                            y2 + 40, img.shape[0]
                        )

                        try:
                            img = img[
                                  int(np.ceil(y1)): int(y2),
                                  int(np.ceil(x1)): int(x2),
                                  :,
                                  ]

                            print("SAVING image at ", dest_path)
                            plt.imsave(
                                dest_path,
                                img,
                                dpi=1000,
                            )
                            plt.close()

                        except IndexError:
                            print("ERROR FOR IMAGE", img_path, img)
                            plt.imsave(
                                dest_path,
                                img,
                                dpi=1000,
                            )
                            plt.close()

                    else:
                        non_cropped_paths.append(img_path)
                        non_cropped_img += 1

                        print("SAVING NON-CROPPED image at ", dest_path)
                        plt.imsave(
                            dest_path,
                            np.array(Image.fromarray(img)),
                            dpi=1000,
                        )
                        plt.close()

            try:
                print(
                    "\t{}% of {} images non cropped".format(
                        np.round(100 * non_cropped_img / number_img, 2), state
                    )
                )
            except ZeroDivisionError:
                continue

        return non_cropped_paths
