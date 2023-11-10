import functools
import logging
import pickle
import typing as t
from dataclasses import dataclass

import faiss
import numpy as np
import torch
from sklearn.preprocessing import normalize


try:
    import clip
except ImportError:
    raise ImportError(
        """
    You don't have `clip` installed.
    See https://github.com/openai/CLIP
    """
    )

CLIP_MODEL_TYPE = "ViT-L/14@336px"
MAX_N = 10
IMAGE_PATH_BASE = "/root/data/laion/images"
PATH_FAISS = "/root/data/laion/faiss/clip-laion-400m-1m-faiss.index"
PATH_EMBS = "/root/data/laion/faiss/clip-laion-400m-1m.pkl"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CLIPSearch:
    """
    Path to the embeddings file.
    Once the object is instantiated, we won't be able to search for more.
    """

    path: str = PATH_EMBS
    faiss_index_path: str = PATH_FAISS
    image_path_base: str = IMAGE_PATH_BASE

    def search_given_emb(self, emb: np.ndarray, n: int) -> t.List[str]:
        """
        Nearest neighbor search given an input embedding.
        Won't filter out the exact match (i.e. if the image is already in the index).
        :param emb: Single-dimensional embedding to search.
        :param n: number of results to return.
        :return: list of LAION image keys.
        """
        emb_dim = emb.shape
        assert len(emb_dim) == 1, "emb must be 1 dimensional"
        assert emb_dim[0] == self._embs.shape[1], "emb must be the same dim as index"
        emb = np.expand_dims(emb, axis=0)
        emb = normalize(emb, axis=1)
        _, idxs = self._faiss_index.search(emb, k=n)
        idxs = idxs.squeeze() if n >= 2 else [idxs.item()]
        return [self._idx_to_key_lookup[idx] for idx in idxs]

    def search_given_text(self, text: str, n: int) -> t.List[str]:
        """
        Nearest neighbor search given an input text.
        Will encode the text first and then run `search_given_emb`.
        """
        emb_text = self._get_text_emb(text=text)
        return self.search_given_emb(emb=emb_text, n=n)

    def show_images_by_key(self, keys: t.List[str]) -> None:
        from IPython.display import Image as JImage, display

        for key in keys:
            path = f"{self.image_path_base}/{key}.png"
            display(JImage(path))

    @property
    def _embs(self) -> np.ndarray:
        return self._index["embs"]

    @property
    def _idx_to_key_lookup(self) -> t.List[str]:
        return self._index["idx_to_key_lookup"]

    @property
    @functools.lru_cache()
    def _index(self) -> dict:
        logger.info("Loading the index...")
        obj = pickle.load(open(self.path, "rb"))
        logger.info("Loading done!")
        return obj

    def _get_text_emb(self, text: str) -> np.ndarray:
        logger.info("Encoding input text query...")
        with torch.no_grad():
            text = clip.tokenize(text).to(self._device)
            enc = self._clip_model.encode_text(text).squeeze()
        logger.info("Text encoded!")
        return enc

    @property
    @functools.lru_cache()
    def _clip_model(self):
        logger.info("Loading CLIP text encoder...")
        model, _ = clip.load(CLIP_MODEL_TYPE, device=self._device)
        logger.info("CLIP loaded!")
        return model

    @property
    @functools.lru_cache()
    def _device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    @functools.lru_cache()
    def _faiss_index(self):
        p = self.faiss_index_path
        logger.info(f"Loading faiss index from {p}...")
        nn = faiss.read_index(p)
        logger.info(f"faiss index loaded!")
        return nn
