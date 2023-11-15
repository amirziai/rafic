# support 1, augment 0, 1, 2, 5
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 1 --num_aug 0
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 1 --num_aug 1
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 1 --num_aug 2
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 1 --num_aug 5

# support 2, augment 0, 1, 2, 5
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 2 --num_aug 0
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 2 --num_aug 1
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 2 --num_aug 2
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 2 --num_aug 5

# support 6, augment 0, 1, 2, 5
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 5 --num_aug 0
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 5 --num_aug 1
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 5 --num_aug 2
python3 -m rafic.maml --inner_lr 0.04 --num_train_iterations 1000 --num_inner_steps 5 --batch_size 64 --device gpu --num_way 5 --num_support 5 --num_aug 5