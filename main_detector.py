from BirdImageCropper import BirdImageCropper

BATCH_SIZE = 32
DATA_PATH = "bird_dataset"
DEST_PATH = "cropped_bird_dataset"

cropper = BirdImageCropper(DATA_PATH, DEST_PATH, BATCH_SIZE)
cropper.get_birds_images_cropped()
