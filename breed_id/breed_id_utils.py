import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Torch imports
import torch
from torchvision import transforms


#####################################################################
# HELPER VARIABLES
#####################################################################
BREED_ID_DIR = os.path.abspath(os.path.dirname(__file__))
LABELS_CSV_PATH = os.path.join(BREED_ID_DIR, 'labels.csv')
SAVED_MODELS_DIR = os.path.join(BREED_ID_DIR, 'saved_models')
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]
TRANSFORM_NORM = transforms.Normalize(
    mean=TRANSFORM_MEAN,
    std=TRANSFORM_STD
)
TRANSFORM_IMG_SIZE = 224
GENERIC_DATA_TRANSFORMS = transforms.Compose([
        transforms.CenterCrop(size=TRANSFORM_IMG_SIZE),
        transforms.ToTensor(),
        TRANSFORM_NORM
    ])


#####################################################################
# HELPER FUNCTIONS
#####################################################################
def tensor_img_show(img, transformed=False):
    """Image show for tensor images."""
    img = img.nump().transpose((1, 2, 0))
    if not transformed:
        img = TRANSFORM_STD * img + TRANSFORM_MEAN
    img = np.clip(img, 0, 1)
    plt.imshow(img)


def predict(img_files: list, model):
    # Process images
    images = [Image.open(img_file) for img_file in img_files]
    batch = torch.stack([GENERIC_DATA_TRANSFORMS(img) for img in images])

    # Use CPU for predicting
    model.to('cpu')
    probs = torch.softmax(model.forward(batch), dim=1).data.numpy()
    return probs

