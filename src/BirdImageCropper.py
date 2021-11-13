import matplotlib.pyplot as plt
import torch
from PIL import Image

# Some basic setup:
# Setup detectron2 logger
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from src.data import ImageFolderWithPaths

setup_logger()

# import some common libraries
import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

BATCH_SIZE = 32
DATA_PATH = "../cropped_bird_dataset"


class BirdImageCropper:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
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
        if not os.path.isdir(self.data_path):
            print("copying the bird dataset to create the cropped dataset")
            os.system("cp -r ../bird_dataset/ ../cropped_bird_dataset")

    def _check_reformat_image(self):
        # Reformat weird images
        for path, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".jpg"):
                    i = plt.imread(os.path.join(path, file))
                    if len(i.shape) == 2 or i.shape[2] != 3:
                        print(file)
                        i = Image.fromarray(i)
                        i = i.convert("RGB")
                        i.save(os.path.join(path, file))

    def get_birds_images_cropped(self):
        self._check_if_directory_is_available()
        model = self._initiate_detector()
        self._check_reformat_image()

        non_cropped_img = 0
        non_cropped_paths = []
        number_img = 0

        for state in ["train", "val", "test"]:

            dataloader = DataLoader(
                ImageFolderWithPaths(
                    os.path.join(self.data_path, state + "_images"),
                    transform=transforms.Compose(
                        [
                            transforms.Resize((128, 128)),
                            transforms.ToTensor(),
                        ]
                    ),
                ),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

            for images, labels, paths in dataloader:
                img_detections = []  # Stores detections for each image index

                for img_path in paths:
                    img = cv2.imread(img_path)
                    with torch.no_grad():
                        detections = model(img)["instances"]
                    img_detections.append(detections)

                # Save cropped images
                for (path, detections) in zip(paths, img_detections):
                    print(path)
                    print("-----")
                    number_img += 1
                    img = np.array(Image.open(path))

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
                            non_cropped_paths.append(path)

                            # path = path.split("/")[-1]
                            # plt.imsave(
                            #     output_folder + "/" + data_folder + "/" + folder + "/" + path,
                            #     np.array(ImageOps.mirror(Image.fromarray(img))),
                            #     dpi=1000,
                            # )
                            # plt.close()

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
                                int(np.ceil(y1)): int(y2),
                                int(np.ceil(x1)): int(x2),
                                :,
                            ]

                            # Save generated image with detections
                            # path = path.split("/")[-1]
                            plt.imsave(
                                path,
                                img,
                                dpi=1000,
                            )
                            plt.close()

                        except IndexError:
                            print("ERROR FOR IMAGE", path, img)

                    else:
                        # Flip the image if we are not able to detect the bird
                        non_cropped_paths.append(path)
                        non_cropped_img += 1
                        # path = path.split("/")[-1]
                        # Flip the image if we are not able to detect it
                        # plt.imsave(
                        #     output_folder + "/" + data_folder + "/" + folder + "/" + path,
                        #     np.array(ImageOps.mirror(Image.fromarray(img))),
                        #     dpi=1000,
                        # )
                        # plt.close()
            print(
                "\t{}% of {} images non cropped".format(
                    np.round(100 * non_cropped_img / number_img, 2), self.data_path
                )
            )

        return non_cropped_paths
