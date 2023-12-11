#!/bin/bash

# maml
python3 -m rafic.maml --device gpu --num_aug 1 --dataset_name aircraft --cross_eval

# protonet
python3 -m rafic.protonet --device gpu --num_aug 1 --dataset_name aircraft --cross_eval
