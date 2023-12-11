# Retrieval-Augmented Few-shot Image Classification (RAFIC)
Stanford CS330 project

# Baseline
Baselines
```python
from rafic import experiments as exp

res = dict()
for dataset in ("birds", "aircraft"):
  res[dataset] = dict(
    zs=exp.zero_shot_text_label(dataset_name=dataset),
    lr=exp.logistic_regression(dataset_name=dataset),
  )  
```

# Experiments
```shell
bash exp1.sh
bash exp2.sh
bash exp3.sh
```

## Env
- Grab a conda env, same as HW2
- `pip install -r requirements.txt`
- Install CLIP: `pip install git+https://github.com/openai/CLIP.git`

## Data
- `rafic/config.py` has a list of all the required data files
- Download from [GDrive](https://drive.google.com/drive/folders/14yglk5frfSxRCl_GJK1paJofLwxpuecT) and place things in the appropriate folders
