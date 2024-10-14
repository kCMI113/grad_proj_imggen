import os
import random

import numpy as np
import torch
from fashion_clip.fashion_clip import FashionCLIP
from tqdm import tqdm

IMAGES_PATH = "./dataset_fashion/images"
EMBEDDINGS_PATH = "./dataset_fashion/embeddings"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def main():
    fclip = FashionCLIP('fashion-clip')
    
    images_list = [f'{IMAGES_PATH}/basic_{j}/item_{i}.jpg' for j in range(0, 3) for i in range(0,26235)]
    
    image_embeddings = fclip.encode_images(images_list, batch_size=8096)
    
    print(f"size of image embeddings : {image_embeddings.shape}")
    torch.save(image_embeddings,os.path.join(EMBEDDINGS_PATH, "embeddings.pth"))
    
    image_embeddings = image_embeddings.reshape(-1, 3, 512)
    print(f"size of image embeddings : {image_embeddings.shape}")
    
    print(image_embeddings.shape)
    
    for image_id in range(26235):
        torch.save(image_embeddings[image_id], f"{EMBEDDINGS_PATH}/{image_id}.pth")

if __name__ == '__main__':
    seed_everything()
    main()