"""Dataloading for The Caltech-UCSD Birds-200-2011 Dataset."""
import os
import glob

import google_drive_downloader as gdd
import imageio
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader

# Overall we have 200 classes
NUM_TRAIN_CLASSES = 130
NUM_VAL_CLASSES = 10
NUM_TEST_CLASSES = 60
NUM_SAMPLES_PER_CLASS = 20

class BirdsDataset(dataset.Dataset):
    """Caltech-UCSD Birds-200-2011 Dataset dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """
    _BASE_PATH = './data/birds'

    def __init__(self, num_support, num_query):
        """Inits Caltech-UCSD Birds-200-2011 Dataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()


        # download the data 
        if not os.path.isdir(self._BASE_PATH):
            print(
                "PLEASE DOWNLOAD THE BIRDS DATASET FROM https://drive.google.com/file/d/190Q9nRXyfF1efyI5zLFjWcr-mZRd5WEV/view?usp=drive_link AND PLACE IT IN data/birds"
            )
            raise FileNotFoundError


        # get all birds species folders
        self._birds_folders = glob.glob(
            os.path.join(self._BASE_PATH, 'CUB_200_2011/CUB_200_2011/images/'))
        assert len(self._birds_folders) == (
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )

        # shuffle birds classes
        np.random.default_rng(0).shuffle(self._birds_folders)

        # check problem arguments
        assert num_support + num_query <= NUM_SAMPLES_PER_CLASS
        self._num_support = num_support
        self._num_query = num_query

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            all_file_paths = glob.glob(
                os.path.join(self._character_folders[class_idx], '*.png')
            )
            sampled_file_paths = np.random.default_rng().choice(
                all_file_paths,
                size=self._num_support + self._num_query,
                replace=False
            )
            images = [load_image(file_path) for file_path in sampled_file_paths]

            # split sampled examples into support and query
            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend([label] * self._num_support)
            labels_query.extend([label] * self._num_query)

        # aggregate into tensors
        images_support = torch.stack(images_support)  # shape (N*S, C, H, W)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query