# retrieval data and index
_PATH_BASE_LAION = "/root/data/laion"
PATH_SEARCH = f"{_PATH_BASE_LAION}/faiss/clip-laion-aircraft-birds.pkl"
PATH_SEARCH_EMB = _PATH_BASE_LAION
PATH_FAISS_INDEX = f"{_PATH_BASE_LAION}/faiss/clip-laion-aircraft-birds-approx.index"
PATH_IMAGES_LAION = f"{_PATH_BASE_LAION}/images"

# datasets
PATH_BASE_BIRDS = "/root/notebooks/cs330-project/data/birds/CUB_200_2011/CUB_200_2011"
PATH_BASE_AIRCRAFT = "/root/data/rafic/aircrafts/fgvc-aircraft-2013b"

# birds
DATASET_BIRDS_FRACTION_TRAIN = 0.7
DATASET_BIRDS_FRACTION_VAL = 0.15

# CLIP
CLIP_MODEL_TYPE = "ViT-L/14@336px"
CLIP_CLASS_TEXT_EMBS = "/root/data/rafic/class-text-embs-{dataset_name}.pickle"
SEED = 0

# general
NUM_WORKERS = 2
