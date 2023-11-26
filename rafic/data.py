"""Dataloading for The Caltech-UCSD Birds-200-2011 Dataset."""
import collections
import functools
import json
import pickle
import typing as t

import numpy as np
import pandas as pd
import torch
from torch.utils.data import dataset, sampler, dataloader

from torchvision import transforms
from PIL import Image

from . import config, search

DatasetGetItemOutput = t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
Key = str

torch.multiprocessing.set_sharing_strategy("file_descriptor")


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


# def load_embedding(path: str) -> torch.Tensor:
#     """
#     Assumes that we're storing images and embeddings in this way:
#     - {path_parent}/images/{class_name}/{image_name}.jpg
#     - {path_parent}/embeddings/{class_name}/{image_name}.np
#     """
#     pieces = path.split("/")
#     base = "/".join(pieces[:-3])
#     cls = pieces[-2]
#     fn = pieces[-1].split(".")[0]
#     path_np = f"{base}/embeddings/{cls}/{fn}.np"
#     return torch.tensor(np.load(path_np))


@functools.lru_cache()
def _get_laion_db():
    print("loading laion db...")
    db = pickle.load(open(config.PATH_SEARCH, "rb"))
    print("laion db loaded!")
    keys, embs = db["idx_to_key_lookup"], db["embs"]
    return {k: torch.tensor(e) for k, e in zip(keys, embs)}


def load_embedding_aug_by_key(key: str) -> torch.Tensor:
    return _get_laion_db()[key]


def _shuffle(sth: t.Sequence, seed: int = config.SEED) -> None:
    np.random.default_rng(seed).shuffle(sth)


class _Dataset(dataset.Dataset):
    def __init__(
        self,
        path_base: str,
        num_support: int,
        num_query: int,
        num_aug: int = 0,
        seed: int = None,
        search_index_big: bool = True,
        faiss_index_path: str = None,
        keep_original_label_idx: bool = False,
    ):
        super().__init__()
        self._path_base = path_base
        self._num_support = num_support
        self._num_query = num_query
        self._num_aug = num_aug
        self._seed = seed
        self._search = search.CLIPSearch(
            path=config.PATH_SEARCH.replace(
                "-1m", "-1m" if search_index_big else "-1k"
            ),
            faiss_index_path=faiss_index_path,
        )
        self._keep_original_label_idx = keep_original_label_idx
        self._metadata = self._get_metadata()
        self._keys = dict()
        self._key_to_split = dict()
        for split in self._metadata:
            self._keys[split] = sorted(self._metadata[split].keys())
            _shuffle(sth=self._keys[split])
            for k in self._keys[split]:
                self._key_to_split[k] = split
        classes = sorted(self._key_to_split)
        self._class_key_to_global_index = {cls: idx for idx, cls in enumerate(classes)}
        self._class_global_index_to_key = {
            idx: cls for cls, idx in self._class_key_to_global_index.items()
        }
        assert num_support + num_query <= self.min_num_samples_per_class

    def _get_metadata(self) -> dict:
        raise NotImplementedError

    def get_text_query(self, class_global_idx: int) -> str:
        class_key = self._get_class_key(class_global_idx=class_global_idx)
        s = class_key.strip().replace(".", " ").replace("_", " ").replace("-", " ")
        return f"photo of a {s}"

    def _get_class_key(self, class_global_idx: int) -> str:
        return self._class_global_index_to_key[class_global_idx]

    def __getitem__(self, keys: t.List[Key]) -> DatasetGetItemOutput:
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        if self._seed is not None:
            np.random.seed(self._seed)
        for idx, key in enumerate(keys):
            embs = self._get_data(key=key, n=self._num_support + self._num_query)
            label = (
                self._class_key_to_global_index[key]
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
        # D is the size of CLIP embeddings (e.g. 768)
        # N is the number of "ways" or classes in the few-shot classification task
        images_support = torch.stack(images_support).float()  # shape (N*(S+A), D)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query).float()
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query

    @property
    def num_supp_aug(self) -> int:
        return self._num_support + self._num_aug

    @property
    @functools.lru_cache()
    def min_num_samples_per_class(self) -> int:
        return min(
            len(items)
            for classes in self._metadata.values()
            for items in classes.values()
        )

    def _get_data(self, key: str, n: int) -> t.List[torch.Tensor]:
        split = self._key_to_split[key]
        options = self._metadata[split][key]
        selected = np.random.choice(options, n, replace=False)
        return list(map(self._load_embedding, selected))

    def get_class_keys_by_split(self, split: str) -> t.List[str]:
        return self._keys[split]

    def _get_embedding_path(self, key: str) -> str:
        raise NotImplementedError

    def _load_embedding(self, key: str) -> torch.Tensor:
        path = self._get_embedding_path(key=key)
        return torch.tensor(np.load(path))

    def _augment(self, embs_supp):
        if self._num_aug == 0:
            return embs_supp
        emb = torch.stack(embs_supp).mean(axis=0).numpy()
        keys = self._search.search_given_emb(emb=emb, n=self._num_aug)
        embs_aug = list(map(load_embedding_aug_by_key, keys))
        comb = embs_supp + embs_aug
        return comb


class AircraftDataset(_Dataset):
    def __init__(self, path_base: str = config.PATH_BASE_AIRCRAFT, **kwargs):
        super().__init__(path_base=path_base, **kwargs)

    def _get_metadata(self) -> dict:
        return json.load(open(f"{self._path_base}/metadata.json"))

    def _get_embedding_path(self, key: str) -> str:
        return f"{self._path_base}/data/embeddings/{key:07d}.np"

    @functools.lru_cache()
    def _get_variant_to_mfg(self) -> dict:
        df = pd.read_csv(f"{self._path_base}/metadata.csv")
        return df.set_index("variant").manufacturer.to_dict()

    def _get_class_key(self, class_global_idx: int) -> str:
        variant = self._class_global_index_to_key[class_global_idx]
        mfg = self._get_variant_to_mfg()[variant]
        return f"{mfg} {variant}"


class BirdsDataset(_Dataset):
    _FRACTION_TRAIN = config.DATASET_BIRDS_FRACTION_TRAIN
    _FRACTION_VAL = config.DATASET_BIRDS_FRACTION_VAL

    def __init__(self, path_base: str = config.PATH_BASE_BIRDS, **kwargs):
        super().__init__(path_base=path_base, **kwargs)
        for f in (self._FRACTION_VAL, self._FRACTION_TRAIN):
            assert 0 < f < 1
        assert 0 < self._FRACTION_TRAIN + self._FRACTION_VAL < 1

    def _get_metadata(self) -> dict:
        metadata = collections.defaultdict(lambda: collections.defaultdict(list))
        classes = [
            x.strip().split(" ")[1].split(".")[1]
            for x in open(f"{self._path_base}/classes.txt").readlines()
        ]
        _shuffle(sth=classes, seed=0)  # hardcoded seed to keep consistent partitions
        c2s = {
            cls: (
                "train"
                if idx < int(len(classes) * self._FRACTION_TRAIN)
                else (
                    "validation"
                    if idx
                    < int(len(classes) * (self._FRACTION_TRAIN + self._FRACTION_VAL))
                    else "test"
                )
            )
            for idx, cls in enumerate(classes)
        }
        for img in open(f"{self._path_base}/images.txt").readlines():
            img = img.strip()
            cls = img.split("/")[0].split(".")[1]
            metadata[c2s[cls]][cls].append(img)
        return metadata

    def _get_embedding_path(self, key: str) -> str:
        return f"{self._path_base}/embeddings/{key}.np"


# class BirdsDataset(dataset.Dataset):
#     """Caltech-UCSD Birds-200-2011 Dataset for meta-learning.
#
#     Each element of the dataset is a task. A task is specified with a key,
#     which is a tuple of class indices (no particular order). The corresponding
#     value is the instantiated task, which consists of sampled (image, label)
#     pairs.
#     """
#     # Overall we have 200 classes
#     # max pictures for a bird is 60
#     # min pictures for a bird is 41
#     NUM_TRAIN_CLASSES = 140
#     NUM_VAL_CLASSES = 30
#     NUM_TEST_CLASSES = 30
#     NUM_SAMPLES_PER_CLASS = 41
#
#     def __init__(
#         self,
#         num_support,
#         num_query,
#         num_aug: int = 0,
#         seed=None,
#         search_index_big=True,
#         faiss_index_path=None,
#         keep_original_label_idx: bool = False,
#     ):
#         """Inits Caltech-UCSD Birds-200-2011 Dataset.
#
#         Args:
#             num_support (int): number of support examples per class
#             num_query (int): number of query examples per class
#         """
#
#         super().__init__()
#         self._num_aug = num_aug
#         self._seed = seed
#         self._search = search.CLIPSearch(
#             path=config.PATH_SEARCH.replace(
#                 "-1m", "-1m" if search_index_big else "-1k"
#             ),
#             faiss_index_path=faiss_index_path,
#         )
#         self._keep_original_label_idx = keep_original_label_idx
#
#         # download the data
#         if not os.path.isdir(config.BASE_PATH):
#             print(
#                 "PLEASE DOWNLOAD THE BIRDS DATASET FROM"
#                 " https://drive.google.com/file/d/190Q9nRXyfF1efyI5zLFjWcr-mZRd5WEV/view?usp=drive_link"
#                 " AND PLACE IT IN data/birds"
#             )
#             raise FileNotFoundError
#
#         # get all birds species folders
#         self._birds_folders = sorted(
#             glob.glob(os.path.join(config.BASE_PATH, "images/*"))
#         )
#         assert len(self._birds_folders) == (
#             self.NUM_TRAIN_CLASSES + self.NUM_VAL_CLASSES + self.NUM_TEST_CLASSES
#         )
#
#         # shuffle birds classes
#         _shuffle(sth=self._birds_folders)
#
#         # check problem arguments
#         assert num_support + num_query <= self.NUM_SAMPLES_PER_CLASS
#         self._num_support = num_support
#         self._num_query = num_query
#
#     @property
#     def num_supp_aug(self) -> int:
#         return self._num_support + self._num_aug
#
#     def __getitem__(self, class_idxs) -> DatasetGetItemOutput:
#         """Constructs a task.
#
#         Data for each class is sampled uniformly at random without replacement.
#
#         Args:
#             class_idxs (tuple[int]): class indices that comprise the task
#
#         Returns:
#             images_support (Tensor): task support images
#                 shape (num_way * num_support, channels, height, width)
#             labels_support (Tensor): task support labels
#                 shape (num_way * num_support,)
#             images_query (Tensor): task query images
#                 shape (num_way * num_query, channels, height, width)
#             labels_query (Tensor): task query labels
#                 shape (num_way * num_query,)
#         """
#         images_support, images_query = [], []
#         labels_support, labels_query = [], []
#
#         for idx, class_idx in enumerate(class_idxs):
#             # get a class's examples and sample from them
#             all_file_paths = sorted(
#                 glob.glob(os.path.join(self._birds_folders[class_idx], "*.jpg"))
#             )
#             rng = get_rng(seed=self._seed)
#             sampled_file_paths = rng.choice(
#                 all_file_paths, size=self._num_support + self._num_query, replace=False
#             )
#             embs = [load_embedding(file_path) for file_path in sampled_file_paths]
#             label = (
#                 int(self._birds_folders[class_idx].split("/")[-1].split(".")[0])
#                 if self._keep_original_label_idx
#                 else idx
#             )
#             # split sampled examples into support and query
#             embs_supp = embs[: self._num_support]
#             embs_supp_aug = self._augment(embs_supp)
#             images_support.extend(embs_supp_aug)
#             images_query.extend(embs[self._num_support :])
#             labels_support.extend([label] * self.num_supp_aug)
#             labels_query.extend([label] * self._num_query)
#
#         # aggregate into tensors
#         images_support = torch.stack(
#             images_support
#         ).float()  # shape (N*(S+A), D) where D is the size of CLIP embeddings (e.g. 768)
#         labels_support = torch.tensor(labels_support)  # shape (N*S)
#         images_query = torch.stack(images_query).float()
#         labels_query = torch.tensor(labels_query)
#
#         return images_support, labels_support, images_query, labels_query
#
#     def _augment(self, embs_supp):
#         if self._num_aug == 0:
#             return embs_supp
#         emb = torch.stack(embs_supp).mean(axis=0).numpy()
#         keys = self._search.search_given_emb(emb=emb, n=self._num_aug)
#         embs_aug = list(map(load_embedding_aug_by_key, keys))
#         comb = embs_supp + embs_aug
#         return comb


class Sampler(sampler.Sampler):
    """Samples task indices."""

    def __init__(
        self, choices: t.List[str], num_way: int, num_tasks: int, seed: int = None
    ):
        """Inits BirdsSampler.

        Args:
            choices (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._choices = choices
        self._num_way = num_way
        self._num_tasks = num_tasks
        self._seed = seed

    def __iter__(self) -> t.Iterable[t.List[str]]:
        rng = get_rng(seed=self._seed)
        return (
            rng.choice(self._choices, size=self._num_way, replace=False)
            for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


def get_dataloader(
    dataset_name,
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
        dataset_name (str): name of the dataset to use
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
        keep_original_label_idx (bool): if false, will re-index labels to [0, 1, ..., num_classes - 1]
    """
    assert num_aug >= 0

    dataset_options = dict(
        birds=BirdsDataset,
        aircraft=AircraftDataset,
    )
    if dataset_name not in dataset_options:
        raise ValueError(f"dataset {dataset_name} is not supported.")

    dataset_cls = dataset_options[dataset_name]
    deterministic = split in {"val", "test"}
    ds = dataset_cls(
        num_support=num_support,
        num_query=num_query,
        seed=seed if deterministic else None,
        num_aug=num_aug,
        search_index_big=search_index_big,
        faiss_index_path=faiss_index_path,
        keep_original_label_idx=keep_original_label_idx,
    )
    choices = ds.get_class_keys_by_split(split=split)

    sampler_obj = Sampler(
        choices=choices,
        num_way=num_way,
        num_tasks=num_tasks_per_epoch,
        seed=seed if deterministic else None,
    )
    return dataloader.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler_obj,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
