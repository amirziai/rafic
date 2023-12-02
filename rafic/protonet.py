"""Implementation of prototypical networks for Omniglot."""

import argparse
import os

import numpy as np
import torch

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from torch import nn
import torch.nn.functional as F  # pylint: disable=unused-import
from torch.utils import tensorboard

from . import data, experiments
from .evaluation import Evaluation  # pylint: disable=unused-import

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 1
VAL_INTERVAL = PRINT_INTERVAL * 1
NUM_TEST_TASKS = 600


class ProtoNetNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        layers = [
            nn.Linear(in_features=768, out_features=NUM_HIDDEN_CHANNELS),
            nn.ReLU(),
        ]
        self._layers = nn.Sequential(*layers)
        self.to(device)

    def forward(self, images):
        return self._layers(images)


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(
        self,
        learning_rate,
        log_dir,
        device,
        compile=False,
        backend=None,
        learner=None,
        val_interval=None,
        save_interval=None,
        bio=False,
    ):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        self.device = device
        if learner is None:
            self._network = ProtoNetNetwork(device)
        else:
            self._network = learner.to(device)

        self.val_interval = VAL_INTERVAL if val_interval is None else val_interval
        self.save_interval = SAVE_INTERVAL if save_interval is None else save_interval
        self.bio = bio

        if compile == True:
            try:
                self._network = torch.compile(self._network, backend=backend)
                print(f"ProtoNetNetwork model compiled")
            except Exception as err:
                print(f"Model compile not supported: {err}")

        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=learning_rate)
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []
        for i, task in enumerate(task_batch):
            # print(i)
            images_support, labels_support, images_query, labels_query = task
            # print(images_support.shape, labels_support.shape, images_query.shape, labels_query.shape)
            images_support = images_support.to(self.device)
            labels_support = labels_support.to(self.device)
            images_query = images_query.to(self.device)
            labels_query = labels_query.to(self.device)

            def _logit(x, p):
                return -((x.unsqueeze(1) - p.unsqueeze(0)) ** 2).sum(-1)

            with torch.no_grad():
                # Generate support features
                support_features = self._network(images_support)

                # Calculate prototypes from features
                labels_uniq = torch.unique(labels_support)
                prototypes = torch.stack(
                    [
                        support_features[labels_support == i].mean(axis=0)
                        for i in sorted(labels_uniq)
                    ]
                )

                # Calculate features for support
                support_logits = _logit(x=support_features, p=prototypes)

            # Generate query features
            query_features = self._network(images_query)
            query_logits = _logit(x=query_features, p=prototypes)

            # Compute loss
            loss = F.cross_entropy(query_logits, labels_query)
            loss_batch.append(loss)
            accuracy_support_batch.append(
                Evaluation.score(logits=support_logits.detach(), labels=labels_support)
            )
            accuracy_query_batch.append(
                Evaluation.score(logits=query_logits.detach(), labels=labels_query)
            )

        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch),
        )

    def train(self, dataloader_meta_train, dataloader_meta_val, writer):
        """Train the ProtoNet.

        Consumes dataloader_meta_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_meta_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_meta_train (DataLoader): loader for train tasks
            dataloader_meta_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f"Starting training at iteration {self._start_train_step}.")
        MAX_TRAIN = len(dataloader_meta_train)
        # exit()
        for i_step, task_batch in enumerate(
            dataloader_meta_train, start=self._start_train_step
        ):
            if i_step > MAX_TRAIN:
                break
            self._optimizer.zero_grad()
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                print(
                    f"Iteration {i_step}: "
                    f"loss: {loss.item():.3f}, "
                    f"support accuracy: {accuracy_support.item():.3f}, "
                    f"query accuracy: {accuracy_query.item():.3f}"
                )
                writer.add_scalar("loss/train", loss.item(), i_step)
                writer.add_scalar(
                    "train_accuracy/support", accuracy_support.item(), i_step
                )
                writer.add_scalar("train_accuracy/query", accuracy_query.item(), i_step)

            if i_step % self.val_interval == 0:
                print("Start Validation...")
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for i, val_task_batch in enumerate(dataloader_meta_val):
                        if self.bio and i > 600:
                            break
                        loss, accuracy_support, accuracy_query = self._step(
                            val_task_batch
                        )
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                    ci95 = 1.96 * np.std(accuracies_query) / np.sqrt(600 * 4)
                if self.bio:
                    print(
                        f"Validation: "
                        f"loss: {loss:.3f}, "
                        f"support accuracy: {accuracy_support:.3f}, "
                        f"query accuracy: {accuracy_query:.3f}",
                        f"Ci95: {ci95:.3f}",
                    )
                else:
                    print(
                        f"Validation: "
                        f"loss: {loss:.3f}, "
                        f"support accuracy: {accuracy_support:.3f}, "
                        f"query accuracy: {accuracy_query:.3f}"
                    )
                writer.add_scalar("loss/val", loss, i_step)
                writer.add_scalar("val_accuracy/support", accuracy_support, i_step)
                writer.add_scalar("val_accuracy/query", accuracy_query, i_step)
                if self.bio:
                    writer.add_scalar("val_accuracy/ci95", ci95, i_step)
            if i_step % self.save_interval == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        raise NotImplementedError(
            "TODO: NUM_TEST_TASKS must be a function of the dataset"
        )
        accuracies = []
        for i, task_batch in enumerate(dataloader_test):
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f"Accuracy over {NUM_TEST_TASKS} test tasks: "
            f"mean {mean:.3f}, "
            f"95% confidence interval {mean_95_confidence_interval:.3f}"
        )

    def load(self, checkpoint_step, filename=""):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load
            filename (str): directly setting name of checkpoint file, default ="", when argument is passed, then checkpoint will be ignored

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            (f'{os.path.join(self._log_dir, "state")}' f"{checkpoint_step}.pt")
            if filename == ""
            else filename
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state["network_state_dict"])
            self._optimizer.load_state_dict(state["optimizer_state_dict"])
            self._start_train_step = checkpoint_step + 1
            print(f"Loaded checkpoint iteration {checkpoint_step}.")
        else:
            raise ValueError(f"No checkpoint for iteration {checkpoint_step} found.")

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(
                network_state_dict=self._network.state_dict(),
                optimizer_state_dict=self._optimizer.state_dict(),
            ),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt',
        )
        print("Saved checkpoint.")


def main(args):
    print(args)

    if (
        args.device == "gpu"
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        DEVICE = "mps"
    elif args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f"./logs/protonet/birds.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.lr_{args.learning_rate}.batch_size_{args.batch_size}"  # pylint: disable=line-too-long
    print(f"log_dir: {log_dir}")
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    protonet = ProtoNet(args.learning_rate, log_dir, DEVICE, args.compile, args.backend)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print("Checkpoint loading skipped.")

    if not args.test:
        num_training_tasks = args.batch_size * (
            args.num_train_iterations - args.checkpoint_step - 1
        )
        print(
            f"Training on tasks with composition "
            f"num_way={args.num_way}, "
            f"num_support={args.num_support}, "
            f"num_query={args.num_query}"
        )
        dataloader_meta_train = data.get_dataloader(
            dataset_name=args.dataset_name,
            split="train",
            batch_size=args.batch_size,
            num_way=args.num_way,
            num_support=args.num_support,
            num_query=args.num_query,
            num_workers=args.num_workers,
            num_aug=args.num_aug,
            aug_combine=args.aug_combine,
            aug_thr=args.aug_thr,
        )
        dataloader_meta_val = data.get_dataloader(
            dataset_name=args.dataset_name,
            split="val",
            batch_size=args.batch_size,
            num_way=args.num_way,
            num_support=args.num_support,
            num_query=args.num_query,
            num_workers=args.num_workers,
            num_aug=args.num_aug,
            aug_combine=args.aug_combine,
            aug_thr=args.aug_thr,
            seed=0,
        )
        protonet.train(dataloader_meta_train, dataloader_meta_val, writer)
    else:
        print(
            f"Testing on tasks with composition "
            f"num_way={args.num_way}, "
            f"num_support={args.num_support}, "
            f"num_query={args.num_query}"
        )
        dataloader_test = data.get_dataloader(
            dataset_name=args.dataset_name,
            split="test",
            batch_size=1,
            num_way=args.num_way,
            num_support=args.num_support,
            num_query=args.num_query,
            num_workers=args.num_workers,
            num_aug=args.num_aug,
            aug_combine=args.aug_combine,
            aug_thr=args.aug_thr,
            seed=0,
        )
        protonet.test(dataloader_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a ProtoNet!")
    parser.add_argument(
        "--log_dir", type=str, default=None, help="directory to save to or load from"
    )
    parser.add_argument(
        "--num_way", type=int, default=experiments.N, help="number of classes in a task"
    )
    parser.add_argument(
        "--num_support",
        type=int,
        default=1,
        help="number of support examples per class in a task",
    )
    parser.add_argument(
        "--num_query",
        type=int,
        default=experiments.Q,
        help="number of query examples per class in a task",
    )
    parser.add_argument("--num_aug", type=int, default=0, help="Number of retrievals")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate for the network",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=experiments.BATCH_SIZE,
        help="number of tasks per outer-loop update",
    )
    parser.add_argument(
        "--num_train_iterations",
        type=int,
        default=5000,
        help="number of outer-loop updates to train for",
    )
    parser.add_argument(
        "--test", default=False, action="store_true", help="train or test"
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=-1,
        help=(
            "checkpoint iteration to load for resuming "
            "training, or for evaluation (-1 is ignored)"
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=experiments.NUM_WORKERS,
        help=("needed to specify the dataloader"),
    )
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument(
        "--backend",
        type=str,
        default="inductor",
        choices=["inductor", "aot_eager", "cudagraphs"],
    )
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset_name", type=str, default="birds")
    parser.add_argument("--aug_thr", type=float, default=None)
    parser.add_argument("--aug_combine", action="store_true")
    args = parser.parse_args()
    main(args)
