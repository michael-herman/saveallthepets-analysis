"""
Starter module for quickly configuring, building, and training models.

Can be directly modified and ran as python module or useful for copying
into and experimenting in jupyter notebook .
"""

import os
import sys

# torch modules
from torch import nn, optim, load
from torchvision import transforms, models

# Add to pythonpath so package imports work when run as module
BREED_ID_DIR = os.path.abspath(os.path.dirname(__file__))
PROJ_DIR = os.path.join(BREED_ID_DIR, os.pardir)
# Uncomment below and comment above two lines for use in jupyter notebook
# PROJ_DIR = os.path.abspath(os.path.join(os.pardir, os.pardir))
print('proj dir:', PROJ_DIR)
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
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=25),
    transforms.ToTensor(),
    TRANSFORM_NORM
])
valid_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
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

# Visualize a handful to ensure transformations make sense (OPTIONAL)
show_sample = False
if show_sample:
    loader_generator.show_sample_images(transform=train_transforms)


# 3. Define network model and freeze layers for use as feature extractor
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# 4. Reshape fully-connected layer
# Note: the name or structure is unique to each Imagenet model
# Useful reference link:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, NUM_CLASSES)
)
# model.fc = nn.Sequential(
#     nn.Linear(2048, 1024),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5),
#     nn.Linear(1024, 512),
#     nn.ReLU(inplace=True),
#     nn.Linear(512, NUM_CLASSES)
# )

# OPTIONAL: load saved model
load_model = False
if load_model:
    model_checkpoint = load(os.path.join(SAVED_MODELS_DIR, 'R50_2nodes_model.pt'))
    print(model_checkpoint.keys())
    model.load_state_dict(model_checkpoint['state_dict'])

# 5. Define configurations for training: learning rate, loss model (criterion),
# optimizer, and scheduler (optional)
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)
# Uncomment below and comment above is using unfrozen feature extractor layer
# optimizer = optim.Adam(model.parameters(), lr=lr)
# Allows for decreasing the learning rate per epoch rate
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                      step_size=100, gamma=0.1)

# 6. Train model
# Base name for saving model checkpoints
base_name = 'R50_2nodes'
epochs = 5
CUDA_INT = 0
trainer = Trainer(train_loader=train_loader, validation_loader=valid_loader,
                  criterion=criterion, optimizer=optimizer,
                  scheduler=scheduler, save_dir=SAVED_MODELS_DIR)
model_checkpoint = trainer.train_model(arch=base_name, model=model,
                                       epochs=epochs,
                                       device_int=CUDA_INT, unfreeze=False,
                                       checkpoint={})

# 7. Visualize result of training
train_loss = model_checkpoint['train_loss']
valid_loss = model_checkpoint['valid_loss']
train_acc = model_checkpoint['train_acc']
valid_acc = model_checkpoint['valid_acc']
# train_mcc = model_checkpoint['train_mcc']
# valid_mcc = model_checkpoint['valid_mcc']
# train_y_true = model_checkpoint['train_y_true']
# train_y_pred = model_checkpoint['train_y_pred']
# valid_y_true = model_checkpoint['valid_y_true']
# valid_y_pred = model_checkpoint['valid_y_pred']

# Plot loss vs accuracy
trainer.plot_loss(arch=base_name, train_loss=train_loss,
                  valid_loss=valid_loss, train_metric=train_acc,
                  valid_metric=valid_acc, metric_title='Accuracy',
                  train_metric_label='Train Accuracy',
                  valid_metric_label='Validation Accuracy')
# # Plot loss vs mcc
# trainer.plot_loss(arch=base_name, train_loss=train_loss,
#                   valid_loss=valid_loss, train_metric=train_mcc,
#                   valid_metric=valid_mcc, metric_title='MCC',
#                   train_metric_label='Train MCC',
#                   valid_metric_label='Validation MCC')
# # Plot train confusion matrix
# trainer.plot_confusion_matrix(y_true=train_y_true, y_pred=train_y_pred,
#                               title=f'{base_name} Train Confusion Matrix')
# # Plot validation confusion matrix
# trainer.plot_confusion_matrix(y_true=valid_y_true, y_pred=valid_y_pred,
#                               title=f'{base_name} Valid Confusion Matrix')
