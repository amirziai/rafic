"""Evaluation component for the model."""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from . import search


class Evaluation:
    @staticmethod
    def score(logits, labels):
        """Returns the mean accuracy of a model's predictions on a set of examples.

        Args:
            logits (torch.Tensor): model predicted logits
                shape (examples, classes)
            labels (torch.Tensor): classification labels from 0 to num_classes - 1
                shape (examples,)
        Return:
            score of scalar value
        """

        assert logits.dim() == 2
        assert labels.dim() == 1
        assert logits.shape[0] == labels.shape[0]
        y = torch.argmax(logits, dim=-1) == labels
        y = y.type(torch.float)
        return torch.mean(y).item()

    @staticmethod
    def test(dataloader_test, model, num_test_tasks: int):
        """Evaluate the evaluation result on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        Returns:
            No return but print out the accuracy over the batch, mean, and its 95% confidence interval.
        """
        accuracies = []
        for task_batch in dataloader_test:
            _, _, accuracy_query = model.eval(task_batch, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(num_test_tasks)
        print(
            f"Accuracy over {num_test_tasks} test tasks: "
            f"mean {mean:.3f}, "
            f"95% confidence interval {mean_95_confidence_interval:.3f}"
        )

    @classmethod
    def eval_non_parametric_nn(cls, dl) -> float:
        return cls._eval_per_instance(
            dl=dl,
            fn_eval=cls._nn,
        )

    @classmethod
    def eval_random(cls, dl, seed):
        return cls._eval_per_instance(
            dl=dl,
            fn_eval=cls._rand,
            seed=seed,
        )

    @classmethod
    def eval_clf(cls, dl, seed):
        return cls._eval_per_instance(dl=dl, fn_eval=cls._clf, seed=seed)

    @staticmethod
    def _eval_per_instance(dl, fn_eval, seed=None):
        if seed is not None:
            np.random.seed(seed)
        correct_total = 0
        cnt_total = 0
        for data_batch in tqdm(dl):
            for idx in range(len(data_batch)):
                correct, cnt = fn_eval(data_batch[idx])
                cnt_total += cnt
                correct_total += correct

        return correct_total / cnt_total

    @staticmethod
    def _nn(data):
        x_tr, y_tr, x_ts, y_ts = data
        uniq = sorted(np.unique(y_tr))
        lookup = {cls: idx for idx, cls in enumerate(uniq)}
        lookup_rev = list(lookup.keys())
        labels = torch.tensor([lookup[a.item()] for a in y_tr])
        centroids = np.vstack(
            [x_tr[labels == i].mean(axis=0).numpy() for i in range(len(uniq))]
        )
        ps = cosine_similarity(x_ts.numpy(), centroids).argmax(axis=0)
        correct = np.sum([y.item() == lookup_rev[p] for y, p in zip(y_ts, ps)])
        return correct, len(ps)

    @staticmethod
    def _rand(data):
        _, _, _, y_ts = data
        ps = np.random.choice(np.unique(y_ts), len(y_ts), replace=False)
        correct = np.sum(y_ts.numpy() == ps)
        return correct, len(ps)

    @staticmethod
    def _clf(data):
        x_tr, y_tr, x_ts, y_ts = data
        w = len(np.unique(y_tr))
        n = len(x_tr) // w
        clf = LogisticRegressionCV(cv=min(n, 3)) if n >= 2 else LogisticRegression()
        clf.fit(x_tr.numpy(), y_tr.numpy())
        ps = clf.predict(x_ts.numpy())
        correct = np.sum(ps == y_ts.numpy())
        return correct, len(ps)

    @staticmethod
    def eval_text_encoder(dl):
        cs = search.CLIPSearch()

        correct = 0
        tot = 0

        for data_batch in dl:
            for _, _, x_ts, y_ts in data_batch:
                labels = set()
                tot += len(x_ts)
                for label in y_ts:
                    labels.add(label.item())
                labels = sorted(labels)
                l2i = {l: i for i, l in enumerate(labels)}
                embs_text = np.vstack(
                    [cs.get_text_emb(f"a photo of a {label}") for label in labels]
                )
                ps = cosine_similarity(x_ts.numpy(), embs_text).argmax(axis=1)
                correct += sum(l2i[label.item()] == p for label, p in zip(y_ts, ps))

        return correct / tot
