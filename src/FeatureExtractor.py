import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.CustomInceptionv3 import CustomInceptionv3
from src.data import ImageFolderWithPaths

torch.manual_seed(0)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name,
        model_path,
        data_path,
        dest_path,
        batch_size,
        data_transforms,
        device,
    ):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.dest_path = dest_path
        self.batch_size = batch_size
        self.data_transforms = data_transforms
        self.device = device

    def _get_model(self):
        if self.model_name == "ResNext":
            self.model = torch.hub.load(
                "facebookresearch/WSL-Images", "resnext101_32x48d_wsl"
            )
            self.model.eval()
            self.model = self._slice_model(to_layer=-1).to(self.device)
            self.data_path = "../299_cropped_bird_dataset"
            self.dest_path = "../ResNext_embeddings"
        elif self.model_name == "Inceptionv3":
            print("choosing Inception v3 model")
            self.model = CustomInceptionv3(final_pooling=1)
            try:
                inception_weights = "/home/kaliayev/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth"
                self.model.load_state_dict(torch.load(inception_weights))
            except FileNotFoundError:
                print("You must change the path to the `inception_weights`")
                raise

            self.model.eval()
            self.data_path = "../299_cropped_bird_dataset"
            self.dest_path = "../Inceptionv3_embeddings"

    def _slice_model(self, from_layer=None, to_layer=None):
        return nn.Sequential(*list(self.model.children())[from_layer:to_layer])

    def _get_dataloader(self, state):
        dataloader = torch.utils.data.DataLoader(
            ImageFolderWithPaths(
                self.data_path + f"/{state}_images", transform=self.data_transforms
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        return dataloader

    def _get_embeddings(self, image_batch, feature_model):
        with torch.no_grad():
            features_batch = feature_model(image_batch).flatten(start_dim=1)
        return features_batch

    def extract_features(self):
        self._get_model()
        self.extract_state_embeddings(self.model, "train")
        self.extract_state_embeddings(self.model, "val")
        test_paths = self.extract_state_embeddings(self.model, "test")
        if self.model_name == "Inceptionv3":
            test_paths = [
                path.replace(".jpg", "").replace(
                    "../299_cropped_bird_dataset/test_images/mistery_category/", ""
                )
                for path in test_paths
            ]
        elif self.model_name == "ResNext":
            test_paths = [
                path.replace(".jpg", "").replace(
                    "../256_cropped_bird_dataset/test_images/mistery_category/", ""
                )
                for path in test_paths
            ]
        else:
            test_paths = [
                path.replace(".jpg", "").replace(
                    "../cropped_bird_dataset/test_images/mistery_category/", ""
                )
                for path in test_paths
            ]

        test_path_filepath = os.path.join(self.dest_path, "test_paths.txt")
        with open(test_path_filepath, "w", encoding="utf-8") as file:
            file.write("\n".join(test_paths))

    def extract_state_embeddings(self, feature_model, state):
        print("Start working on: ", state)
        dataloader = self._get_dataloader(state)
        features_list = []
        labels_list = []
        img_paths = []
        for images, labels, paths in tqdm(dataloader):
            with torch.no_grad():
                features_batch = self._get_embeddings(images, feature_model)
            features_list.extend(features_batch)
            labels_list.extend(labels)
            img_paths.extend(paths)
        if not os.path.isdir(self.dest_path):
            print("Creating embeddings folder")
            os.system(f"mkdir {self.dest_path}")
        print("Saving all features and labels for", state)
        torch.save(
            features_list,
            os.path.join(self.dest_path, f"birds_features_{state}.pt"),
        )
        torch.save(
            labels_list, os.path.join(self.dest_path, f"birds_labels_{state}.pt")
        )
        print(f"Saved {len(features_list)} features map and {len(labels_list)} labels")

        return img_paths
