import numpy as np
from matplotlib import pyplot as plt

# Torch imports
from torchvision import transforms


#####################################################################
# HELPER VARIABLES
#####################################################################
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]
TRANSFORM_NORM = transforms.Normalize(
    mean=TRANSFORM_MEAN,
    std=TRANSFORM_STD
)
TRANSFORM_IMG_SIZE = 224
GENERIC_DATA_TRANSFORMS = transforms.Compose([
        transforms.Resize((TRANSFORM_IMG_SIZE, TRANSFORM_IMG_SIZE)),
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


