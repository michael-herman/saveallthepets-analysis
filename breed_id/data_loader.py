import os
import pandas as pd
import scipy as sp
from PIL import Image
from matplotlib import pyplot as plt

# sklearn imports
from sklearn.model_selection import StratifiedShuffleSplit

# torch imports
from torch.utils.data import Dataset, DataLoader

# project imports
from breed_id.breed_id_utils import GENERIC_DATA_TRANSFORMS


class DataLoaderGenerator(object):
    def __init__(self, labels_path, img_dir, split_size=0.20, rand_state=42):
        self._labels_path = labels_path
        self._img_dir = img_dir
        self._split_size = split_size
        self._rand_state = rand_state
        self._train_transform = None
        self._valid_transform = None

        # Create labels df, map breed to target, and add target column to df
        self._df = pd.read_csv(labels_path)
        self._breeds = self._df.breed.sort_values().unique()
        self._breed_to_target = {
            self._breeds[i]: i for i in range(len(self._breeds))
        }
        self._df['target'] = self._df.breed.apply(
            lambda x: self._breed_to_target[x]
        )

        # Generate train and validation dataframe subsets
        self._splits_indices = self._generate_splits(split_size, rand_state)
        self._train_df = self._df.iloc[self._splits_indices[0]]
        self._validation_df = self._df.iloc[self._splits_indices[1]]

    def _generate_splits(self, split_size, rand_state=42) -> tuple:
        """Split data into train and validaton data subset.

        Returns:
            (train_idx, valid_idx) tuple representing indices allocated as
                train data and validation data.
        """
        y = self._df.target.to_list()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=split_size,
                                     random_state=rand_state)
        return next(sss.split(X=y, y=y))

    def get_data_loaders(self, batch_size=32, shuffle=False, num_workers=0,
                         pin_memory=False, train_transform=None,
                         valid_transform=None) -> dict:
        if train_transform:
            self._train_transform = train_transform
        if valid_transform:
            self._valid_transform = valid_transform

        train_dataset = DogBreedDataset(self._train_df, self._img_dir,
                                        self._train_transform)
        train_loader = DataLoader(train_dataset, batch_size, shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory)
        valid_dataset = DogBreedDataset(self._validation_df, self._img_dir,
                                        self._valid_transform)
        valid_loader = DataLoader(valid_dataset, batch_size, shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory)
        return {'train': train_loader, 'validation': valid_loader}

    def show_sample_images(self, im_scale_x=64, im_scale_y=64,
                           transform=None) -> None:
        """
        Visualize a sample of 36 dataset images including breed label.
        """
        # Use Generic transform if None provided
        if transform is None and not self._train_transform:
            transform = GENERIC_DATA_TRANSFORMS

        dataset = DogBreedDataset(data_df=self._df, img_dir=self._img_dir,
                                  transform=transform)
        loader = DataLoader(dataset=dataset, batch_size=36, shuffle=False)
        images, targets = next(iter(loader))
        images_n = images.numpy()

        # Plot the images in the batch, along with the corresponding labels
        grid_width = 6
        grid_height = 6
        f, ax = plt.subplots(grid_width, grid_height)
        f.set_size_inches(12, 12)

        img_idx = 0
        for i in range(0, grid_width):
            for j in range(0, grid_height):
                target = targets[img_idx].item()
                name = self._breeds[target]
                ax[i][j].axis('off')
                ax[i][j].set_title(f'{name} ({target})')
                ax[i][j].imshow(images_n[img_idx].transpose(1, 2, 0))
                img_idx += 1

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.25)
        plt.show()


class DogBreedDataset(Dataset):
    """Custom Dataset object for Dog Breed Identification."""
    def __init__(self, data_df, img_dir, transform=None):
        """
        Args:
            data_df (pd.DataFrame): Dataframe object with id, breed, and
                target columns.
            img_dir (str): Directory for all the train images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._labels_df = data_df
        self._img_dir = img_dir
        if transform:
            self._transform = transform
        else:  # Otherwise use default transform
            self._transform = GENERIC_DATA_TRANSFORMS

    def __len__(self):
        return len(self._labels_df)

    def __getitem__(self, idx):
        img_id, breed = self._labels_df.iloc[idx, [0, 2]]
        img_file = os.path.join(self._img_dir, f'{img_id}.jpg')
        image = Image.open(img_file)

        if self._transform:
            image = self._transform(image)

        return image, breed
