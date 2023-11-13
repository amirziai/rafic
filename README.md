# Retrieval-Augmented Few-shot Image Classification (RAFIC)
Stanford CS330 project

## Demos
- [Search and eval](search-and-eval-demo.ipynb)

## Env
- Grab a conda env, same as HW2
- `pip install -r requirements.txt`
- Install CLIP: `pip install git+https://github.com/openai/CLIP.git`

## Data
- `rafic/config.py` has a list of all the required data files
- Download from GDrive and place things in the appropriate folders
  - Unzip `bird_embeddings.zip` into `data/birds/CUB_200_2011/CUB_200_2011/`

## ProtoNet
From the base directory:
```shell
python -m rafic.protonet --log_dir logs --num_support 2 --num_aug 1
```
