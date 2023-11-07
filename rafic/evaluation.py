"""Evaluation component for the model."""
import torch
import numpy as np


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
