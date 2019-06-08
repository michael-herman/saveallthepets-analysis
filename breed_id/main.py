import os
import sys

# torch modules
from torch import nn, optim
from torchvision import transforms, models

# Add to pythonpath so package imports work when run as module
BREED_ID_DIR = os.path.abspath(os.path.dirname(__file__))
PROJ_DIR = os.path.join(BREED_ID_DIR, os.pardir)
if PROJ_DIR not in sys.path:
    sys.path.append(PROJ_DIR)

# Package modules
from breed_id.breed_id_utils import LABELS_CSV_PATH, SAVED_MODELS_DIR
from breed_id.breed_id_utils import TRANSFORM_NORM, TRANSFORM_IMG_SIZE
from breed_id.data_loader import DataLoaderGenerator
from breed_id.train import Trainer


# Module constants
# Map to local directory of images
IMG_DIR = '/media/wdblack-1/saveallthepets/dog-breed-dataset/kaggle/train'
BATCH_SIZE = 32
# Imagenet models require 224x224 sized images except Inception which used 299
INPUT_SIZE = TRANSFORM_IMG_SIZE
VALIDATION_SPLIT = 0.20  # Train/validation subset ratio
RAND_STATE = 42  # Random state to help ensure reproducibility when an option
NUM_CLASSES = 120  # There are 120 breeds in this kaggle dataset

# 1. Define train and validation data transformers
train_transforms = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            TRANSFORM_NORM
        ])
valid_transforms = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    TRANSFORM_NORM
])

# 2. Define loader configurations and generate dataloaders
loader_generator = DataLoaderGenerator(labels_path=LABELS_CSV_PATH,
                                       img_dir=IMG_DIR,
                                       split_size=VALIDATION_SPLIT,
                                       rand_state=RAND_STATE)
dataloaders = loader_generator.get_data_loaders(batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=4,
                                                pin_memory=True,
                                                train_transform=train_transforms,
                                                valid_transform=valid_transforms)
train_loader = dataloaders['train']
valid_loader = dataloaders['validation']

# 3. Define network model and freeze layers for use as feature extractor
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.require_grad = False

# 4. Reshape fully-connected layer
# Note: the name or structure is unique to each Imagenet model
# Useful reference link:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
model.fc = nn.Sequential(
    nn.Linear(2048, NUM_CLASSES)
)

# 5. Define configurations for training: learning rate, loss model (criterion),
# optimizer, and scheduler (optional)
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)
# Allows for decreasing the learning rate per epoch rate
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                      step_size=50, gamma=0.1)

# 6. Train model
# Base name for saving model checkpoints
base_name = 'R50_1node'
epochs = 20
trainer = Trainer(train_loader=train_loader, validation_loader=valid_loader,
                  criterion=criterion, optimizer=optimizer,
                  scheduler=None, save_dir=SAVED_MODELS_DIR)
model_checkpoint = trainer.train_model(arch=base_name, model=model,
                                       epochs=epochs)

# TODO: Visualize result of training
