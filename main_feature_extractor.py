import torch
from torchvision import transforms
from src.FeatureExtractor import FeatureExtractor


BATCH_SIZE = 32
DATA_PATH = "299_cropped_bird_dataset"
DATA_TRANSFORMS = transforms.Compose(
    [
        transforms.transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
EMBEDDINGS_PATH = "Inceptionv3_embeddings"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    feature_extractor = FeatureExtractor(
        model_name="Inceptionv3",
        data_path=DATA_PATH,
        dest_path=EMBEDDINGS_PATH,
        batch_size=BATCH_SIZE,
        data_transforms=DATA_TRANSFORMS,
        device=device,
    )
    feature_extractor.extract_features()
