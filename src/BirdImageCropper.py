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

# import some common libraries
import numpy as np
import os
import cv2

# import some common detectron2 utilities
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
        # Set up for model
        # Define a Mask-R-CNN model in Detectron2
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

            dataloader = DataLoader(
                ImageFolderWithPaths(
                    os.path.join(self.source_data_path, state + "_images"),
                    transform=transforms.Compose(
                        [
                            # (256,256)
                            transforms.Resize((299, 299)),
                            transforms.RandomApply(
                                torch.nn.ModuleList(
                                    [
                                        transforms.ColorJitter(
                                            brightness=0.3, contrast=0.3, saturation=0.1, hue=0.4
                                        ),
                                        #transforms.RandomCrop(size=()),
                                        #transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomPerspective(distortion_scale=0.4, p=0.9),
                                        # transforms.Grayscale(num_output_channels=3),
                                    ]
                                ),
                                p=0.5,
                            ),
                            transforms.ToTensor(),
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
                    # Bounding boxes and labels of detections
                    if len(detections.scores) > 0:
                        # Get the most probable bird prediction bounding box
                        index_birds = np.where(
                            detections.pred_classes.cpu().numpy() == 14
                        )[
                            0
                        ]  # 14 is the default class number for bird
                        if len(index_birds) == 0:
                            # Flip the image if we are not able to detect the bird
                            non_cropped_img += 1
                            non_cropped_paths.append(img_path)

                            print("SAVING NON-CROPPED image at ", dest_path)
                            plt.imsave(
                                dest_path,
                                np.array(ImageOps.mirror(Image.fromarray(img))),
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

                        # If we are able to detect the bird, enlarge the bounding box and generate a new image
                        x1, y1 = np.maximum(0, int(x1) - 20), np.maximum(
                            0, int(y1) - 20
                        )
                        x2, y2 = np.minimum(x2 + 40, img.shape[1]), np.minimum(
                            y2 + 40, img.shape[0]
                        )

                        try:
                            img = img[
                                int(np.ceil(y1)) : int(y2),
                                int(np.ceil(x1)) : int(x2),
                                :,
                            ]

                            # Save generated image with detections
                            print("SAVING image at ", dest_path)
                            plt.imsave(
                                dest_path,
                                img,
                                dpi=1000,
                            )
                            plt.close()

                        except IndexError:
                            print("ERROR FOR IMAGE", img_path, img)

                    else:
                        # Flip the image if we are not able to detect the bird
                        non_cropped_paths.append(img_path)
                        non_cropped_img += 1

                        print("SAVING NON-CROPPED image at ", dest_path)
                        plt.imsave(
                            dest_path,
                            np.array(ImageOps.mirror(Image.fromarray(img))),
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
