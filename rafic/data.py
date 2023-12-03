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
    _DATASET_NAME = None

    def __init__(
        self,
        path_base: str,
        num_support: int,
        num_query: int,
        num_aug: int = 0,
        seed: int = None,
        search_index_big: bool = True,
        faiss_index_path: str = config.PATH_FAISS_INDEX,
        use_global_labels: bool = False,
        aug_combine: bool = False,
        aug_thr: t.Optional[float] = None,
        aug_by_text: float = 0,
        append_cos_sim: bool = False,
        clip_class_text_embs_pattern: str = config.CLIP_CLASS_TEXT_EMBS,
    ):
        assert 0 <= aug_by_text <= 1
        super().__init__()
        self._path_base = path_base
        self._num_support = num_support
        self._num_query = num_query
        self._num_aug = num_aug
        self._aug_combine = aug_combine
        if aug_thr is not None:
            assert 0 < aug_thr < 1
        self._aug_thr = aug_thr
        self._seed = seed
        self._search = search.CLIPSearch(
            path=config.PATH_SEARCH.replace(
                "-1m", "-1m" if search_index_big else "-1k"
            ),
            faiss_index_path=faiss_index_path,
        )
        self.use_global_labels = use_global_labels
        self._metadata = self._get_metadata()
        self._keys = dict()
        self._key_to_split = dict()
        self._aug_by_text = aug_by_text
        self._append_cos_sim = append_cos_sim
        self._clip_class_text_embs_pattern = clip_class_text_embs_pattern
        for split in self._metadata:
            self._keys[split] = sorted(self._metadata[split].keys())
            _shuffle(sth=self._keys[split])
            for k in self._keys[split]:
                self._key_to_split[k] = split
        self._classes = sorted(self._key_to_split)
        self._class_key_to_global_index = {
            cls: idx for idx, cls in enumerate(self._classes)
        }
        self._class_global_index_to_key = {
            idx: cls for cls, idx in self._class_key_to_global_index.items()
        }
        assert num_support + num_query <= self.min_num_samples_per_class

    def _get_metadata(self) -> dict:
        raise NotImplementedError

    def get_text_query(self, class_global_idx: int) -> str:
        class_key = self._get_class_key(class_global_idx=class_global_idx)
        s = class_key.strip().replace(".", " ").replace("_", " ").replace("-", " ")
        return f"photo of a {s} {self._DATASET_NAME}"

    def get_class_text_emb(self, class_global_idx: int) -> np.ndarray:
        text = self.get_text_query(class_global_idx=class_global_idx)
        return self._search.get_text_emb(text=text).cpu().numpy()

    def _get_class_key(self, class_global_idx: int) -> str:
        return self._class_global_index_to_key[class_global_idx]

    def __getitem__(self, keys: t.List[Key]) -> DatasetGetItemOutput:
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        if self._seed is not None:
            np.random.seed(self._seed)
        for idx, key in enumerate(keys):
            embs = self._get_data(key=key, n=self._num_support + self._num_query)
            class_global_idx = self._class_key_to_global_index[key]
            label = class_global_idx if self.use_global_labels else idx
            # split sampled examples into support and query
            embs_supp = embs[: self._num_support]
            embs_supp_aug = self._augment(
                embs_supp=embs_supp,
                class_global_idx=class_global_idx,
            )
            images_support.extend(embs_supp_aug)
            embs_query = self._append1_cond(embs[self._num_support :])
            images_query.extend(embs_query)
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

    @staticmethod
    def _append_vals(
        embs: t.List[torch.Tensor],
        vals: t.Union[float, t.List[float]],
    ) -> t.List[torch.Tensor]:
        if isinstance(vals, (int, float)):
            vals = [float(vals)] * len(embs)
        return [torch.cat((e, torch.tensor([v]))) for e, v in zip(embs, vals)]

    def _append1_cond(self, embs: t.List[torch.Tensor]) -> t.List[torch.Tensor]:
        return self._append_vals(embs=embs, vals=1.0) if self._append_cos_sim else embs

    def _append_vals_cond(
        self,
        embs: t.List[torch.Tensor],
        vals: t.Union[float, t.List[float]],
    ) -> t.List[torch.Tensor]:
        return self._append_vals(embs=embs, vals=vals) if self._append_cos_sim else embs

    @property
    def num_supp_aug(self) -> int:
        if self._num_aug == 0:
            return self._num_support
        elif self._aug_combine:
            return self._num_support + 1
        else:
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

    @functools.lru_cache()
    def _get_class_text_embs(self) -> np.ndarray:
        path = self._clip_class_text_embs_pattern.format(
            dataset_name=self._DATASET_NAME
        )
        return pickle.load(open(path, "rb"))

    def _augment(self, embs_supp, class_global_idx: int) -> t.List[torch.Tensor]:
        if self._num_aug == 0:
            return self._append1_cond(embs=embs_supp)
        zero = np.zeros(len(embs_supp[0]))
        emb_txt = (
            self._get_class_text_embs()[class_global_idx]
            if self._aug_by_text > 0
            else zero
        )
        emb_img = (
            torch.stack(embs_supp).mean(axis=0).numpy()
            if self._aug_by_text < 1
            else zero
        )
        emb = self._aug_by_text * emb_txt + (1 - self._aug_by_text) * emb_img
        embs_supp = self._append1_cond(embs=embs_supp)
        res = self._search.search_given_emb(emb=emb, n=self._num_aug)
        _rem = []
        if self._aug_thr is not None:
            # what to use for imputation
            _impute_with = torch.tensor(emb)
            # _impute_with = embs_supp[0]
            res = [x for x in res if x.score >= self._aug_thr]
            if len(res) == 0:
                _aug = [_impute_with] * (1 if self._aug_combine else self._num_aug)
                return embs_supp + self._append1_cond(embs=_aug)
            if len(res) != self._num_aug:
                # if going to combine => no action needed
                if not self._aug_combine:
                    # if going to preserve individual embs => impute the missing ones
                    _rem = self._append1_cond(
                        [_impute_with] * (self._num_aug - len(res))
                    )

        keys = [x.key for x in res]
        scores = [x.score for x in res]
        embs_aug = list(map(load_embedding_aug_by_key, keys))
        if not self._aug_combine:
            embs_aug = self._append_vals_cond(embs=embs_aug, vals=scores)
            return embs_supp + embs_aug + _rem
        else:
            _aug = [torch.stack(embs_aug).mean(axis=0)]
            _aug = self._append_vals_cond(embs=_aug, vals=np.mean(scores).item())
            return embs_supp + _aug


class AircraftDataset(_Dataset):
    _DATASET_NAME = "aircraft"

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
    _DATASET_NAME = "bird"
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
                    "val"
                    if idx
                    < int(len(classes) * (self._FRACTION_TRAIN + self._FRACTION_VAL))
                    else "test"
                )
            )
            for idx, cls in enumerate(classes)
        }
        for img in open(f"{self._path_base}/images.txt").readlines():
            _, img = img.strip().split(" ")
            cls = img.split("/")[0].split(".")[1]
            key = img.replace(".jpg", "")
            metadata[c2s[cls]][cls].append(key)
        return metadata

    def _get_embedding_path(self, key: str) -> str:
        return f"{self._path_base}/embeddings/{key}.np"


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
    num_aug=0,
    num_workers=config.NUM_WORKERS,
    search_index_big=True,
    seed=None,
    faiss_index_path=config.PATH_FAISS_INDEX,
    use_global_labels: bool = False,
    aug_combine: bool = False,
    aug_thr: t.Optional[float] = None,
    aug_by_text: float = 0,
    append_cos_sim: bool = False,
):
    """Returns a dataloader.DataLoader for Caltech-UCSD Birds-200-2011.

    Args:
        dataset_name (str): name of the dataset to use
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_aug (int): number of additional items to retrieve.
        num_workers (int): number of workers for data loading.
        search_index_big (bool): True: 1M images, False: 1K images.
        seed (int): for reproducibility
        faiss_index_path (str): path to the faiss index
        use_global_labels (bool): if false, will re-index labels to [0, 1, ..., num_classes - 1]
        aug_thr (float): if provided, will be used for filtering retrieved images.
        aug_combine (bool): if true, will average all retrieved images into a single vector.
    """
    assert num_aug >= 0

    dataset_options = dict(
        birds=BirdsDataset,
        aircraft=AircraftDataset,
    )
    if dataset_name not in dataset_options:
        raise ValueError(f"dataset {dataset_name} is not supported.")

    dataset_cls = dataset_options[dataset_name]
    ds = dataset_cls(
        num_support=num_support,
        num_query=num_query,
        seed=seed,
        num_aug=num_aug,
        search_index_big=search_index_big,
        faiss_index_path=faiss_index_path,
        use_global_labels=use_global_labels,
        aug_combine=aug_combine,
        aug_thr=aug_thr,
        append_cos_sim=append_cos_sim,
        aug_by_text=aug_by_text,
    )
    choices = ds.get_class_keys_by_split(split=split)

    sampler_obj = Sampler(
        choices=choices,
        num_way=num_way,
        num_tasks=len(choices),
        seed=seed,
    )
    return dataloader.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler_obj,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
