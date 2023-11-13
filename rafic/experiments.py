import typing as t

import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .birds_data import get_birds_dataloader
from .evaluation import Evaluation


K_VALS = (1, 2, 5)
A_VALS = (0, 1, 2, 5)
N = 10
Q = 10
BATCH_SIZE = 16
NUM_TASKS_PER_EPOCH = 200
NUM_WORKERS = 8


def _get_val_dataloader(k, a, n: int = N, q: int = Q, seed: int = config.SEED) -> float:
    return get_birds_dataloader(
        split="val",
        batch_size=BATCH_SIZE,
        num_way=n,
        num_support=k,
        num_query=q,
        num_tasks_per_epoch=NUM_TASKS_PER_EPOCH,
        num_workers=NUM_WORKERS,
        seed=seed,
        num_aug=a,
    )


def random(n: int = N, q: int = Q, seed: int = config.SEED) -> float:
    dl = _get_val_dataloader(k=1, a=0, n=n, q=q, seed=seed)
    return Evaluation.eval_random(dl=dl, seed=seed)


def zero_shot_text_label(
    n: int = N,
    q: int = Q,
) -> float:
    dl = get_birds_dataloader(
        split="val",
        batch_size=BATCH_SIZE,
        num_way=n,
        num_support=1,
        num_query=q,
        num_tasks_per_epoch=NUM_TASKS_PER_EPOCH,
        num_workers=NUM_WORKERS,
        seed=config.SEED,
        num_aug=0,
        keep_original_label_idx=True,
    )
    return Evaluation.eval_text_encoder(dl)


def non_parametric_image_embeddings(
    k_vals: t.Sequence[int] = K_VALS,
    a_vals: t.Sequence[int] = A_VALS,
    n: int = N,
    q: int = Q,
) -> pd.DataFrame:
    def _fn(k, a):
        return Evaluation.eval_non_parametric_nn(
            dl=_get_val_dataloader(k=k, a=a, n=n, q=q)
        )

    return _run(k_vals=k_vals, a_vals=a_vals, acc_fn=_fn)


def logistic_regression(
    k_vals: t.Sequence[int] = K_VALS,
    a_vals: t.Sequence[int] = A_VALS,
    seed: int = config.SEED,
    n: int = N,
    q: int = Q,
) -> pd.DataFrame:
    def _fn(k, a):
        return Evaluation.eval_clf(
            dl=_get_val_dataloader(k=k, a=a, n=n, q=q), seed=seed
        )

    return _run(k_vals=k_vals, a_vals=a_vals, acc_fn=_fn)


def _run(
    k_vals: t.Sequence[int],
    a_vals: t.Sequence[int],
    acc_fn: t.Callable,
    n: int = N,
    q: int = Q,
) -> pd.DataFrame:
    return pd.DataFrame(
        dict(num_support=k, num_aug=a, acc=acc_fn(k=k, a=a, n=n, q=q))
        for k in k_vals
        for a in a_vals
    )


def plot_val_accuracy_vs_support(df: pd.DataFrame) -> None:
    df.pivot_table(index="num_support", columns="num_aug", values="acc").plot()
    plt.ylabel("Validation accuracy")
    title = "Validation accuracy vs. number of support images with different number of augmented images"
    _ = plt.title(title)
