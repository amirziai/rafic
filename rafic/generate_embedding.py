import clip
import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# We use CLIP encoder
# Make sure you install CLIP
# https://github.com/openai/CLIP
# ```bash
# pip install git+https://github.com/openai/CLIP.git
# ```

# Instruction
# The images should be in the images folder under the dataset's root folder. Then the
# embedding will be generated under the dataset's root folder with embedding as folder name.
# The generated embeddings will be placed in the same folder structure as the images
# but with file extension of .py.

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)


def encode_image(path: str) -> np.ndarray:
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    emb = model.encode_image(image).squeeze().detach().numpy().astype(np.float16)
    assert emb.shape == (768,)
    return emb


def generate_and_save_embeddings(source_folder, dest_folder):
    """
    Load images from subfolders, generate embeddings and save them with nested progress bars.

    Args:
    - source_folder (str): Folder where images are located.
    - dest_folder (str): Folder where embeddings will be saved.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    all_dirs = [x[0] for x in os.walk(source_folder)]

    # Overall progress bar for all directories
    with tqdm(total=len(all_dirs), desc="Overall Progress", position=0) as pbar_all:
        for root, dirs, files in os.walk(source_folder):
            # Nested progress bar for files in each directory
            with tqdm(
                total=len(files),
                desc=f"Processing {os.path.basename(root)}",
                position=1,
                leave=False,
            ) as pbar_dir:
                for name in files:
                    if name.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(root, name)
                        embedding = encode_image(img_path)

                        # Construct destination path
                        relative_path = os.path.relpath(root, source_folder)
                        dest_path = os.path.join(dest_folder, relative_path)

                        if not os.path.exists(dest_path):
                            os.makedirs(dest_path)

                        # Save embedding
                        embedding_file = name.rsplit(".", 1)[0]
                        embedding_path = os.path.join(dest_path, embedding_file)
                        np.save(embedding_path, embedding)

                    pbar_dir.update(1)
            pbar_all.update(1)


# Example usage
# data_folder = "/Users/hangfeilin/Desktop/stanford_homework/CS330/cs330-project/data/birds/CUB_200_2011/CUB_200_2011/"
data_folder = (
    "/Users/hangfeilin/Desktop/stanford_homework/CS330/cs330-project/data/EuroSAT_RGB/"
)
source_folder = data_folder + "images"  # Path to the folder containing training images
destination_folder = (
    data_folder + "embeddings"
)  # Path to the folder where embeddings will be saved
generate_and_save_embeddings(source_folder, destination_folder)
