import unittest
import os
import torch
import pandas as pd
from torch import nn
from torchvision import models
from math import ceil
from breed_id.data_loader import DataLoaderGenerator
from breed_id.breed_id_utils import predict, SAVED_MODELS_DIR

# NOTE: must map to local directory of images
IMG_DIR = '/media/wdblack-1/saveallthepets/dog-breed-dataset/kaggle/train'
TEST_IMG_DIR = '/media/wdblack-1/saveallthepets/dog-breed-dataset/kaggle/test'
# Kaggle dataset can be found:
# https://www.kaggle.com/c/dog-breed-identification/data


class DataLoaderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._loader = DataLoaderGenerator(labels_path='labels.csv', img_dir=IMG_DIR)

    def test_generate_splits(self):
        # Case 1: default 80/20 ratio
        train_idx, valid_idx = self._loader._splits_indices
        total = len(self._loader._df)
        expected_train = int(0.8 * total)
        expected_valid = ceil(0.2 * total)
        self.assertEqual(expected_train, len(train_idx),
                         msg=f'Expected: {expected_train} train count; result: {len(train_idx)}')
        self.assertEqual(expected_valid, len(valid_idx),
                         msg=f'Expected: {expected_valid} valid count; result: {len(valid_idx)}')
        # Case 2: 50/50 ratio
        train_idx, valid_idx = self._loader._generate_splits(split_size=0.50)
        total = len(self._loader._df)
        expected_train = int(0.5 * total)
        expected_valid = ceil(0.5 * total)
        self.assertEqual(expected_train, len(train_idx),
                         msg=f'Expected: {expected_train} train count; result: {len(train_idx)}')
        self.assertEqual(expected_valid, len(valid_idx),
                         msg=f'Expected: {expected_valid} valid count; result: {len(valid_idx)}')

    def test_get_data_loaders(self):
        loaders = self._loader.get_data_loaders()
        # Verify return dict with two keys: train & validation
        self.assertEqual(2, len(loaders),
                         msg=f'Expected 2; result: {len(loaders)}')
        for key in ['train', 'validation']:
            self.assertTrue(key in loaders,
                            msg=f'{key} not in {loaders.keys()}')

        # Assess size of each loader: more than or equal to data subset size
        expected = len(self._loader._train_df)
        max_loader_count = len(loaders['train']) * 32
        self.assertLessEqual(expected, max_loader_count,
                             msg=f'Expected: {expected}; result: {max_loader_count}')
        expected = len(self._loader._validation_df)
        max_loader_count = len(loaders['validation']) * 32
        self.assertLessEqual(expected, max_loader_count,
                            msg=f'Expected: {expected}; result: {max_loader_count}')

    def test_show_sample_images(self):
        """No unit cases, just visual check."""
        self._loader.show_sample_images()

    def test_predict(self):
        # load model checkpoint and reconfigure model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(2048, 120)
        )
        checkpoint = torch.load(os.path.join(SAVED_MODELS_DIR, 'R50_1node_model.pt'))
        model.load_state_dict(checkpoint['state_dict'])

        # Load sample of validation images
        valid_df = self._loader._validation_df
        img_ids = valid_df.id[:3].tolist()
        breeds = valid_df.breed[:3].tolist()
        targets = valid_df.target[:3].tolist()
        img_files = [os.path.join(IMG_DIR, f'{img}.jpg') for img in img_ids]
        probs = predict(img_files, model)

        # verify model results
        for idx, prob in enumerate(probs):
            print(img_ids[idx], breeds[idx], prob[targets[idx]] * 100)
        self.assertTrue(len(probs) > 0)


if __name__ == '__main__':
    unittest.main()
