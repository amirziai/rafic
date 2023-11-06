import functools
import logging
import pickle
import typing as t
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


try:
    import clip
except ImportError:
    raise ImportError(
        """
    You don't have `clip` installed.
    See https://github.com/openai/CLIP.
    """
    )

CLIP_MODEL_TYPE = "ViT-L/14@336px"
MAX_N = 10


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CLIPSearch:
    """
    Path to the embeddings file.
    `max_n` is the maximum number of items we want to return.
    Once the object is instantiated, we won't be able to search for more.
    """
    path: str
    max_n: int = MAX_N

    def search_given_emb(self, emb: np.ndarray, n: int) -> t.List[str]:
        """
        Nearest neighbor search given an input embedding.
        Won't filter out the exact match (i.e. if the image is already in the index).
        :param emb: Single-dimensional embedding to search.
        :param n: number of results () to return. Can't be more than `max_n`.
        :return: list of LAION image keys.
        """
        assert n <= self.max_n
        emb_dim = emb.shape
        assert len(emb_dim) == 1, "emb must be 1 dimensional"
        assert emb_dim[0] == self._embs.shape[1], "emb must be the same dim as index"
        emb = np.expand_dims(
            emb,
            axis=0,
        )
        idxs = self._nn.kneighbors(emb)[1].squeeze()[:n]
        return [self._idx_to_key_lookup[idx] for idx in idxs]

    def search_given_text(self, text: str, n: int) -> t.List[str]:
        """
        Nearest neighbor search given an input text.
        Will encode the text first and then run `search_given_emb`.
        """
        emb_text = self._get_text_emb(text=text)
        return self.search_given_emb(emb=emb_text, n=n)

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
    def _nn(self):
        logger.info("Fitting NN object...")
        nn = NearestNeighbors(
            n_neighbors=self.max_n,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(self._embs)
        logger.info("NN object fitted!")
        return nn
