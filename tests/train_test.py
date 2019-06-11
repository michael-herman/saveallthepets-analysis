import unittest
import os
import json
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from breed_id.train import Trainer
from breed_id.breed_id_utils import TRANSFORM_NORM

# NOTE: Update path per your local directory
FLOWER_DATA_DIR = '/media/wdblack-1/saveallthepets/flower_data/flower_data'
# Data can be found from Kaggle:
# https://www.kaggle.com/nunenuh/pytorch-challange-flower-dataset


class TrainerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Load label mapping
        label_mapping_path = os.path.join(FLOWER_DATA_DIR, 'cat_to_name.json')
        with open(label_mapping_path, mode='r') as f:
            self._cat_to_name = json.load(f)

        # Define data transformations
        input_size = 224
        train_transforms = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            TRANSFORM_NORM
        ])
        valid_transforms = transforms.Compose([
            transforms.Resize(400),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            TRANSFORM_NORM
        ])

        # Load the datasets with ImageFolder
        train_data = ImageFolder(os.path.join(FLOWER_DATA_DIR, 'train'),
                                 transform=train_transforms)
        valid_data = ImageFolder(os.path.join(FLOWER_DATA_DIR, 'valid'),
                                 transform=valid_transforms)

        # Configure dataloaders
        self._batch_size = 24
        self._train_loader = DataLoader(train_data, batch_size=self._batch_size,
                                        shuffle=True, num_workers=4,
                                        pin_memory=True)
        self._valid_loader = DataLoader(valid_data, batch_size=self._batch_size,
                                        shuffle=True, num_workers=4,
                                        pin_memory=True)

        # Transfer learning model configurations
        self._model = models.densenet161(pretrained=True)
        self._classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2208, 102))
        ]))

        # Freeze parameters and reshape pretrained model
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.classifier = self._classifier

        # Configure training parameters
        self._lr = 0.01
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self._model.classifier.parameters(),
                                    lr=self._lr)
        self._scheduler = optim.lr_scheduler.StepLR(self._optimizer,
                                                    step_size=50, gamma=0.1)

    def test_train_model(self):
        arch = 'trainer_unit_test'
        trainer = Trainer(self._train_loader, self._valid_loader,
                          self._criterion, self._optimizer,
                          self._scheduler, save_dir='saved_models')

        checkpoint = trainer.train_model(arch, self._model,
                                         epochs=20)
        self.assertTrue(checkpoint['train_acc'][-1] > 90)
        self.assertTrue(checkpoint['valid_acc'][-1] > 90)

    def test_early_stopping(self):
        arch = 'early_stop_unit_test'
        trainer = Trainer(self._train_loader, self._valid_loader,
                          self._criterion, self._optimizer,
                          self._scheduler, save_dir='saved_models')

        checkpoint = trainer.train_model(arch, self._model,
                                         epochs=100, patience=1)
        self.assertTrue(checkpoint['train_acc'][-1] > 90)
        self.assertTrue(checkpoint['valid_acc'][-1] > 90)


if __name__ == '__main__':
    unittest.main()
