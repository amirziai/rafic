"""Dataloading for The Caltech-UCSD Birds-200-2011 Dataset."""
import functools
import os
import glob
import pickle
import typing as t

import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader

from torchvision import transforms
from PIL import Image

from . import config, search

# Overall we have 200 classes
# max pictures for a bird is 60
# min pictures for a bird is 41
NUM_TRAIN_CLASSES = 130
NUM_VAL_CLASSES = 10
NUM_TEST_CLASSES = 60
NUM_SAMPLES_PER_CLASS = 41


def get_rng(seed):
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def load_image(file_path):
    """Loads and transforms an Caltech-UCSD Birds-200-2011 image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (3, 224, 224)
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(file_path).convert("RGB")  # handle greyscale images
    tensor = transform(image)
    return tensor


def load_embedding(path: str) -> torch.Tensor:
    """
    Assumes that we're storing images and embeddings in this way:
    - {path_parent}/images/{class_name}/{image_name}.jpg
    - {path_parent}/embeddings/{class_name}/{image_name}.np
    """
    pieces = path.split("/")
    base = "/".join(pieces[:-3])
    cls = pieces[-2]
    fn = pieces[-1].split(".")[0]
    path_np = f"{base}/embeddings/{cls}/{fn}.np"
    return torch.tensor(np.load(path_np))


@functools.lru_cache()
def _get_laion_db():
    print('loading laion db...')
    db = pickle.load(open(config.PATH_SEARCH, "rb"))
    print('laion db loaded!')
    keys, embs = db["idx_to_key_lookup"], db["embs"]
    return {k: torch.tensor(e) for k, e in zip(keys, embs)}


def load_embedding_aug_by_key(key: str) -> torch.Tensor:
    return _get_laion_db()[key]


class BirdsDataset(dataset.Dataset):
    """Caltech-UCSD Birds-200-2011 Dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    def __init__(
        self,
        num_support,
        num_query,
        num_aug: int = 0,
        seed=None,
        search_index_big=True,
        faiss_index_path=None,
        keep_original_label_idx: bool = False,
    ):
        """Inits Caltech-UCSD Birds-200-2011 Dataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()
        self._num_aug = num_aug
        self._seed = seed
        self._search = search.CLIPSearch(
            path=config.PATH_SEARCH.replace(
                "-1m", "-1m" if search_index_big else "-1k"
            ),
            faiss_index_path=faiss_index_path,
        )
        self._keep_original_label_idx = keep_original_label_idx

        # download the data
        if not os.path.isdir(config.BASE_PATH):
            print(
                "PLEASE DOWNLOAD THE BIRDS DATASET FROM"
                " https://drive.google.com/file/d/190Q9nRXyfF1efyI5zLFjWcr-mZRd5WEV/view?usp=drive_link"
                " AND PLACE IT IN data/birds"
            )
            raise FileNotFoundError

        # get all birds species folders
        self._birds_folders = sorted(
            glob.glob(os.path.join(config.BASE_PATH, "images/*"))
        )
        assert len(self._birds_folders) == (
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )

        # shuffle birds classes
        np.random.default_rng(config.SEED).shuffle(self._birds_folders)

        # check problem arguments
        assert num_support + num_query <= NUM_SAMPLES_PER_CLASS
        self._num_support = num_support
        self._num_query = num_query

    @property
    def num_supp_aug(self) -> int:
        return self._num_support + self._num_aug

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

        for idx, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            all_file_paths = sorted(
                glob.glob(os.path.join(self._birds_folders[class_idx], "*.jpg"))
            )
            rng = get_rng(seed=self._seed)
            sampled_file_paths = rng.choice(
                all_file_paths, size=self._num_support + self._num_query, replace=False
            )
            embs = [load_embedding(file_path) for file_path in sampled_file_paths]
            label = (
                int(self._birds_folders[class_idx].split("/")[-1].split(".")[0])
                if self._keep_original_label_idx
                else idx
            )
            # split sampled examples into support and query
            embs_supp = embs[: self._num_support]
            embs_supp_aug = self._augment(embs_supp)
            images_support.extend(embs_supp_aug)
            images_query.extend(embs[self._num_support :])
            labels_support.extend([label] * self.num_supp_aug)
            labels_query.extend([label] * self._num_query)

        # aggregate into tensors
        images_support = torch.stack(
            images_support
        ).float()  # shape (N*(S+A), D) where D is the size of CLIP embeddings (e.g. 768)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query).float()
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query

    def _augment(self, embs_supp):
        if self._num_aug == 0:
            return embs_supp
        emb = torch.stack(embs_supp).mean(axis=0).numpy()
        keys = self._search.search_given_emb(emb=emb, n=self._num_aug)
        embs_aug = list(map(load_embedding_aug_by_key, keys))
        comb = embs_supp + embs_aug
        return comb


class BirdsSampler(sampler.Sampler):
    """Samples task specification keys for an Caltech-UCSD Birds-200-2011 Dataset."""

    def __init__(self, split_idxs, num_way, num_tasks, seed=None):
        """Inits BirdsSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks
        self._seed = seed

    def __iter__(self):
        rng = get_rng(seed=self._seed)
        return (
            rng.choice(self._split_idxs, size=self._num_way, replace=False)
            for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


def get_birds_dataloader(
    split,
    batch_size,
    num_way,
    num_support,
    num_query,
    num_tasks_per_epoch,
    num_aug=0,
    num_workers=config.NUM_WORKERS,
    search_index_big=True,
    seed=None,
    faiss_index_path=config.PATH_FAISS_INDEX,
    keep_original_label_idx: bool = False,
):
    """Returns a dataloader.DataLoader for Caltech-UCSD Birds-200-2011.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
        num_aug (int): number of additional items to retrieve.
        num_workers (int): number of workers for data loading.
        search_index_big (bool): True: 1M images, False: 1K images.
        seed (int): for reproducibility
        faiss_index_path (str): path to the faiss index
    """
    assert num_aug >= 0

    if split == "train":
        split_idxs = range(NUM_TRAIN_CLASSES)
    elif split == "val":
        split_idxs = range(NUM_TRAIN_CLASSES, NUM_TRAIN_CLASSES + NUM_VAL_CLASSES)
    elif split == "test":
        split_idxs = range(
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES,
        )
    else:
        raise ValueError

    deterministic = split in {"val", "test"}
    sampler_obj = BirdsSampler(
        split_idxs,
        num_way,
        num_tasks_per_epoch,
        seed=seed if deterministic else None,
    )
    return dataloader.DataLoader(
        dataset=BirdsDataset(
            num_support=num_support,
            num_query=num_query,
            seed=seed if deterministic else None,
            num_aug=num_aug,
            search_index_big=search_index_big,
            faiss_index_path=faiss_index_path,
            keep_original_label_idx=keep_original_label_idx,
        ),
        batch_size=batch_size,
        sampler=sampler_obj,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def get_class_index_to_label() -> t.Dict[int, str]:
    i2l = dict()
    for p in glob.glob(f"{config.BASE_PATH}/images/*"):
        idx, name = p.split("/")[-1].split(".")
        i2l[int(idx)] = name
    return i2l
