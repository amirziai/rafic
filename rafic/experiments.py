import typing as t

import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .birds_data import get_birds_dataloader
from .evaluation import Evaluation


KS = [1, 2, 5]
AS = [0, 1, 2, 5]


def _get_val_dataloader(k, a):
    return get_birds_dataloader(
        split="val",
        batch_size=16,
        num_way=5,
        num_support=k,
        num_query=1,
        num_tasks_per_epoch=200,
        num_workers=8,
        seed=0,
        num_aug=a,
    )


def zero_shot_text_label():
    dl = get_birds_dataloader(
        split="val",
        batch_size=16,
        num_way=5,
        num_support=1,
        num_query=1,
        num_tasks_per_epoch=200,
        num_workers=8,
        seed=0,
        num_aug=0,
        keep_original_label_idx=True,
    )
    return Evaluation.eval_text_encoder(dl)


def non_parametric_image_embeddings(k_vals: KS, a_vals: AS) -> pd.DataFrame:
    def _fn(k, a):
        return Evaluation.eval_non_parametric_nn(dl=_get_val_dataloader(k=k, a=a))

    return _run(k_vals=k_vals, a_vals=a_vals, acc_fn=_fn)


def logistic_regression(
    k_vals: KS, a_vals: AS, seed: int = config.SEED
) -> pd.DataFrame:
    def _fn(k, a):
        return Evaluation.eval_clf(dl=_get_val_dataloader(k=k, a=a), seed=seed)

    return _run(k_vals=k_vals, a_vals=a_vals, acc_fn=_fn)


def _run(k_vals: KS, a_vals: AS, acc_fn: t.Callable) -> pd.DataFrame:
    return pd.DataFrame(
        dict(num_support=k, num_aug=a, acc=acc_fn(k=k, a=a))
        for k in k_vals
        for a in a_vals
    )


def plot_val_accuracy_vs_support(df: pd.DataFrame):
    df.pivot_table(index="num_support", columns="num_aug", values="acc").plot()
    plt.ylabel("Validation accuracy")
    title = "Validation accuracy vs. number of support images with different number of augmented images"
    _ = plt.title(title)
