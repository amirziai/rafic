#!/bin/bash

# Define the arrays
methods=("protonet" "maml")
datasets=("birds" "aircraft")
num_augs=("0" "1" "2" "5" "20" "50")

# Double loop over the arrays
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
      for num_aug in "${num_augs[@]}"; do
        python -m rafic.${method} --num_aug ${num_aug} --dataset_name ${dataset}
      done
    done
done
