#!/bin/bash

# maml
python3 -m rafic.maml --device gpu --num_aug 1 --dataset_name aircraft --append_cos_sim --aug_lr --inner_lr_aug 0.04 --learn_inner_lrs_aug

# protonet
python3 -m rafic.protonet --device gpu --num_aug 1 --dataset_name aircraft --append_cos_sim
