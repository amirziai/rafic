"""Implementation of model-agnostic meta-learning."""

import argparse
import os

import numpy as np
import torch

from torch import nn
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils import tensorboard

from . import data, experiments, util

torch.multiprocessing.set_sharing_strategy("file_system")


# MAML with FC
NUM_FC_LAYERS = 1
INPUT_CLIP_EMD_DIM = 768
NUM_HIDDEN_FEATURES = [64]
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 1
VAL_INTERVAL = 1
NUM_TEST_TASKS = 600


class MAML:
    """Trains and assesses a MAML."""

    def __init__(
        self,
        num_outputs,
        num_inner_steps,
        num_support,
        inner_lr,
        learn_inner_lrs,
        aug_lr,
        inner_lr_aug,
        learn_inner_lrs_aug,
        outer_lr,
        log_dir,
        device,
        append_cos_sim,
        add_class_cos_sims,
    ):
        """Inits MAML.

        The network consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and ReLU activation.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            num_outputs (int): dimensionality of output, i.e. number of classes
                in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        meta_parameters = {}

        self.device = device

        self._append_cos_sim = append_cos_sim
        # construct feature extractor
        dim = INPUT_CLIP_EMD_DIM
        dim += 1 if append_cos_sim else 0
        dim += num_outputs if add_class_cos_sims else 0
        for i in range(NUM_FC_LAYERS):
            if i == 0:
                meta_parameters[f"fc{i}"] = nn.init.xavier_uniform_(
                    torch.empty(
                        NUM_HIDDEN_FEATURES[i],
                        dim,
                        requires_grad=True,
                        device=self.device,
                    )
                )
                meta_parameters[f"b{i}"] = nn.init.zeros_(
                    torch.empty(
                        NUM_HIDDEN_FEATURES[i], requires_grad=True, device=self.device
                    )
                )
            else:
                meta_parameters[f"fc{i}"] = nn.init.xavier_uniform_(
                    torch.empty(
                        NUM_HIDDEN_FEATURES[i],
                        NUM_HIDDEN_FEATURES[i - 1],
                        requires_grad=True,
                        device=self.device,
                    )
                )
                meta_parameters[f"b{i}"] = nn.init.zeros_(
                    torch.empty(
                        NUM_HIDDEN_FEATURES[i], requires_grad=True, device=self.device
                    )
                )

        # construct last linear layer
        meta_parameters[f"w{NUM_FC_LAYERS}"] = nn.init.xavier_uniform_(
            torch.empty(
                num_outputs,
                NUM_HIDDEN_FEATURES[-1],
                requires_grad=True,
                device=self.device,
            )
        )
        meta_parameters[f"b{NUM_FC_LAYERS}"] = nn.init.zeros_(
            torch.empty(num_outputs, requires_grad=True, device=self.device)
        )

        self._meta_parameters = meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._inner_lrs_aug = {
            k: torch.tensor(inner_lr_aug, requires_grad=learn_inner_lrs_aug)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        if aug_lr:
            self._optimizer = torch.optim.Adam(
                list(self._meta_parameters.values())
                + list(self._inner_lrs.values())
                + list(self._inner_lrs_aug.values()),
                lr=self._outer_lr,
            )
        else:
            self._optimizer = torch.optim.Adam(
                list(self._meta_parameters.values()) + list(self._inner_lrs.values()),
                lr=self._outer_lr,
            )

        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0
        self._num_support = num_support
        self._aug_lr = aug_lr

    def _forward(self, images, parameters):
        """Computes predicted classification logits.

        Args:
            images (Tensor): batch of images
                shape (num_images, channels, height, width)
            parameters (dict[str, Tensor]): parameters to use for
                the computation

        Returns:
            a Tensor consisting of a batch of logits
                shape (num_images, classes)
        """
        x = images
        for i in range(NUM_FC_LAYERS):
            x = F.linear(input=x, weight=parameters[f"fc{i}"], bias=parameters[f"b{i}"])
            x = F.relu(x)
        # x = torch.mean(x, dim=[2, 3])
        return F.linear(
            input=x,
            weight=parameters[f"w{NUM_FC_LAYERS}"],
            bias=parameters[f"b{NUM_FC_LAYERS}"],
        )

    def _inner_loop(self, images, labels, train):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            images (Tensor): task support set inputs
                shape (num_images, channels, height, width)
            labels (Tensor): task support set outputs
                shape (num_images,)
            train (bool): whether we are training or evaluating

        Returns:
            parameters (dict[str, Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
            gradients(list[float]): gradients computed from auto.grad, just needed
                for autograders, no need to use this value in your code and feel to replace
                with underscore
        """
        accuracies = []
        parameters = {k: torch.clone(v) for k, v in self._meta_parameters.items()}
        gradients = None
        ### START CODE HERE ###
        # TODO: finish implementing this method.
        # This method computes the inner loop (adaptation) procedure
        # over the course of _num_inner_steps steps for one
        # task. It also scores the model along the way.
        # Make sure to populate accuracies and update parameters.
        # Use F.cross_entropy to compute classification losses.
        # Use util.score to compute accuracies.

        for step in range(self._num_inner_steps + 1):
            logits = self._forward(images, parameters)  # computes the logits
            loss = F.cross_entropy(logits, labels)
            accuracy = util.score(logits, labels)
            if step == 0:  # record the accuracy before any parameter adaptation
                accuracies.append(accuracy)

            if step < self._num_inner_steps:
                if not self._aug_lr:
                    # computes the gradients based on support tasks - skip if it is test
                    gradients = torch.autograd.grad(
                        loss, parameters.values(), create_graph=train
                    )
                    # update the model's parameter by the support tasks
                    for (name, param), grad in zip(parameters.items(), gradients):
                        parameters[name] = param - self._inner_lrs[name] * grad
                else:
                    ## code for separate loss - support and augment
                    logits_support = self._forward(
                        images[: self._num_support], parameters
                    )  # computes the support data logits
                    loss_support = F.cross_entropy(
                        logits_support, labels[: self._num_support]
                    )
                    logits_aug = self._forward(
                        images[self._num_support :], parameters
                    )  # computes the augmented data logits
                    # use the cos sim as weights in augmented data loss computation
                    aug_weights = (
                        images[self._num_support :, -1]
                        if self._append_cos_sim
                        else None
                    )
                    loss_aug = F.cross_entropy(
                        logits_aug, labels[self._num_support :], aug_weights
                    )
                    gradients_support = torch.autograd.grad(
                        loss_support, parameters.values(), create_graph=train
                    )
                    gradients_aug = torch.autograd.grad(
                        loss_aug, parameters.values(), create_graph=train
                    )
                    for (name, param), grad_support, grad_aug in zip(
                        parameters.items(), gradients_support, gradients_aug
                    ):
                        parameters[name] = (
                            param
                            - self._inner_lrs[name] * grad_support
                            - self._inner_lrs_aug[name] * grad_aug
                        )
                logits = self._forward(images, parameters)
                accuracy = util.score(logits, labels)
                accuracies.append(accuracy)

        ### END CODE HERE ###
        return parameters, accuracies, gradients

    def _outer_step(self, task_batch, train):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(self.device)
            labels_support = labels_support.to(self.device)
            images_query = images_query.to(self.device)
            labels_query = labels_query.to(self.device)
            ### START CODE HERE ###
            # TODO: finish implementing this method.
            # For a given task, use the _inner_loop method to adapt for
            # _num_inner_steps steps, then compute the MAML loss and other
            # metrics. Reminder you can replace gradients with _ when calling
            # _inner_loop.
            # Use F.cross_entropy to compute classification losses.
            # Use util.score to compute accuracies.
            # Make sure to populate outer_loss_batch, accuracies_support_batch,
            # and accuracy_query_batch.
            # support accuracy: The first element (index 0) should be the accuracy before any steps are taken.

            ## inner loop
            parameters, accuracies_support, _ = self._inner_loop(
                images_support, labels_support, train
            )

            ## query set
            logits_query = self._forward(images_query, parameters)
            loss_query = F.cross_entropy(logits_query, labels_query)
            accuracy_query = util.score(logits_query, labels_query)
            accuracy_query_batch.append(accuracy_query)

            ## batch
            accuracies_support_batch.append(accuracies_support)
            outer_loss_batch.append(loss_query)

            ### END CODE HERE ###
        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(accuracies_support_batch, axis=0)
        accuracy_query = np.mean(accuracy_query_batch)
        return outer_loss, accuracies_support, accuracy_query

    def train(self, dataloader_meta_train, dataloader_meta_val, writer):
        """Train the MAML.

        Consumes dataloader_meta_train to optimize MAML meta-parameters
        while periodically validating on dataloader_meta_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_meta_train (DataLoader): loader for train tasks
            dataloader_meta_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f"Starting training at iteration {self._start_train_step}.")
        for i_step, task_batch in enumerate(
            dataloader_meta_train, start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query = self._outer_step(
                task_batch, train=True
            )
            outer_loss.backward()
            self._optimizer.step()

            if i_step % LOG_INTERVAL == 0:
                print(
                    f"Iteration {i_step}: "
                    f"loss: {outer_loss.item():.3f}, "
                    f"pre-adaptation support accuracy: "
                    f"{accuracies_support[0]:.3f}, "
                    f"post-adaptation support accuracy: "
                    f"{accuracies_support[-1]:.3f}, "
                    f"post-adaptation query accuracy: "
                    f"{accuracy_query:.3f}"
                )
                writer.add_scalar("loss/train", outer_loss.item(), i_step)
                writer.add_scalar(
                    "train_accuracy/pre_adapt_support", accuracies_support[0], i_step
                )
                writer.add_scalar(
                    "train_accuracy/post_adapt_support", accuracies_support[-1], i_step
                )
                writer.add_scalar(
                    "train_accuracy/post_adapt_query", accuracy_query, i_step
                )

            # if i_step % VAL_INTERVAL == 0:
            if i_step % 10 == 0 or i_step >= len(dataloader_meta_train) - 1:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for val_task_batch in dataloader_meta_val:
                    outer_loss, accuracies_support, accuracy_query = self._outer_step(
                        val_task_batch, train=False
                    )
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(accuracies_pre_adapt_support)
                accuracy_post_adapt_support = np.mean(accuracies_post_adapt_support)
                accuracy_post_adapt_query = np.mean(accuracies_post_adapt_query)
                print(
                    f"Validation: "
                    f"loss: {loss:.3f}, "
                    f"pre-adaptation support accuracy: "
                    f"{accuracy_pre_adapt_support:.2f}, "
                    f"post-adaptation support accuracy: "
                    f"{accuracy_post_adapt_support:.2f}, "
                    f"post-adaptation query accuracy: "
                    f"[{accuracy_post_adapt_query:.2f}]"
                )
                writer.add_scalar("loss/val", loss, i_step)
                writer.add_scalar(
                    "val_accuracy/pre_adapt_support", accuracy_pre_adapt_support, i_step
                )
                writer.add_scalar(
                    "val_accuracy/post_adapt_support",
                    accuracy_post_adapt_support,
                    i_step,
                )
                writer.add_scalar(
                    "val_accuracy/post_adapt_query", accuracy_post_adapt_query, i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        raise NotImplementedError(
            "TODO: NUM_TEST_TASKS must be a function of the dataset"
        )
        accuracies = []
        for task_batch in dataloader_test:
            _, _, accuracy_query = self._outer_step(task_batch, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f"Accuracy over {NUM_TEST_TASKS} test tasks: "
            f"mean {mean:.3f}, "
            f"95% confidence interval {mean_95_confidence_interval:.3f}"
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = f'{os.path.join(self._log_dir, "state")}' f"{checkpoint_step}.pt"
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._meta_parameters = state["meta_parameters"]
            self._inner_lrs = state["inner_lrs"]
            self._optimizer.load_state_dict(state["optimizer_state_dict"])
            self._start_train_step = checkpoint_step + 1
            print(f"Loaded checkpoint iteration {checkpoint_step}.")
        else:
            raise ValueError(f"No checkpoint for iteration {checkpoint_step} found.")

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(
                meta_parameters=self._meta_parameters,
                inner_lrs=self._inner_lrs,
                optimizer_state_dict=optimizer_state_dict,
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
        # on MPS the derivative for aten::linear_backward is not implemented ... Waiting for PyTorch 2.1.0
        # DEVICE = "mps"

        # Due to the above, default for now to cpu
        DEVICE = "cpu"
    elif args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f"../../logs/maml/birds.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.aug_{args.num_aug}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.batch_size}.aug_lr_{args.aug_lr}.inner_lr_aug_{args.inner_lr_aug}.learn_inner_lrs_aug_{args.learn_inner_lrs_aug}"  # pylint: disable=line-too-long
    print(f"log_dir: {log_dir}")
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    maml = MAML(
        num_outputs=args.num_way,
        num_inner_steps=args.num_inner_steps,
        num_support=args.num_support,
        inner_lr=args.inner_lr,
        learn_inner_lrs=args.learn_inner_lrs,
        aug_lr=args.aug_lr,
        inner_lr_aug=args.inner_lr_aug,
        learn_inner_lrs_aug=args.learn_inner_lrs_aug,
        outer_lr=args.outer_lr,
        log_dir=log_dir,
        device=DEVICE,
        append_cos_sim=args.append_cos_sim,
        add_class_cos_sims=args.add_class_cos_sims,
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print("Checkpoint loading skipped.")

    if not args.test:
        num_training_tasks = args.batch_size * (
            args.num_train_iterations - args.checkpoint_step - 1
        )
        print(
            f"Training on {num_training_tasks} tasks with composition: "
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
            seed=args.seed,
            num_aug=args.num_aug,
            aug_thr=args.aug_thr,
            aug_combine=args.aug_combine,
            aug_by_text=args.aug_by_text,
            append_cos_sim=args.append_cos_sim,
            train_repeat_cnt=args.train_repeat_cnt,
            add_class_cos_sims=args.add_class_cos_sims,
        )
        dataloader_meta_val = data.get_dataloader(
            dataset_name=args.dataset_name,
            split="val",
            batch_size=args.batch_size,
            num_way=args.num_way,
            num_support=args.num_support,
            num_query=args.num_query,
            num_workers=args.num_workers,
            seed=args.seed,
            num_aug=args.num_aug,
            aug_thr=args.aug_thr,
            aug_combine=args.aug_combine,
            aug_by_text=args.aug_by_text,
            append_cos_sim=args.append_cos_sim,
            add_class_cos_sims=args.add_class_cos_sims,
        )
        maml.train(dataloader_meta_train, dataloader_meta_val, writer)
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
            batch_size=args.batch_size,
            num_way=args.num_way,
            num_support=args.num_support,
            num_query=args.num_query,
            num_workers=args.num_workers,
            seed=args.seed,
            num_aug=args.num_aug,
            aug_thr=args.aug_thr,
            aug_combine=args.aug_combine,
            aug_by_text=args.aug_by_text,
            append_cos_sim=args.append_cos_sim,
            add_class_cos_sims=args.add_class_cos_sims,
        )
        maml.test(dataloader_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a MAML!")
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="directory to save to or load from"
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
    parser.add_argument(
        "--num_inner_steps", type=int, default=100, help="number of inner-loop updates"
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=0.04,
        help="inner-loop learning rate initialization",
    )
    parser.add_argument(
        "--learn_inner_lrs",
        default=False,
        action="store_true",
        help="whether to optimize inner-loop learning rates",
    )
    parser.add_argument(
        "--outer_lr", type=float, default=0.001, help="outer-loop learning rate"
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
        default=15000,
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
        "--num_workers", type=int, default=2, help=("needed to specify dataloader")
    )
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--raw_image_features",
        default=True,
        action="store_true",
        help=("whether to use the raw image features or the CLIP emgeddings"),
    )
    parser.add_argument("--num_aug", type=int, default=0, help="Number of retrievals")
    parser.add_argument(
        "--dataset_name", type=str, default="birds", help="dataset name"
    )
    parser.add_argument("--aug_lr", action="store_true")
    parser.add_argument(
        "--inner_lr_aug",
        type=float,
        default=0.2,
        help="inner-loop learning rate initialization for augmented data",
    )
    parser.add_argument(
        "--learn_inner_lrs_aug",
        default=False,
        action="store_true",
        help="whether to optimize inner-loop learning rates",
    )
    parser.add_argument("--seed", default=0)
    parser.add_argument("--aug_thr", type=float, default=None)
    parser.add_argument("--aug_combine", action="store_true")
    parser.add_argument("--aug_by_text", type=float, default=0.8)
    parser.add_argument("--append_cos_sim", action="store_true")
    parser.add_argument("--train_repeat_cnt", type=float, default=1)
    parser.add_argument("--add_class_cos_sims", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    main(args)
