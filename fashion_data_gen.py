
import argparse
import os

import pandas as pd
import torch
from model import BatchGSDF

BATCH_SIZE = 10
WIDTH = 512
HEIGHT = 512
INFERENCE_STEP = 30

# np.random.seed(42) # query 색상 랜덤 추가 시 seed 고정

items = torch.load("./data/small_uni_enhanced_prompt3.pt")
# cut_items = pd.read_csv("./data/additional_id_2024-04-28.csv")
# cut_items = pd.read_csv("./VBPR/new_item_data.csv")

# items = items[items["article_id"].isin(cut_items["article_id"])]
# items_desc = items["detail_desc"].values
# items_id = items["article_id"].astype(str).values

gen_img_save_path = "./dataset_fashion/images/"
gen_latent_save_path = "./dataset_fashion/latents/"
gen_embed_save_path = "./dataset_fashion/embeddings/"


# def dump_pickle(data, path):
#     with open(path, "wb") as file:
#         pickle.dump(data, file)

# def query_add_color() -> List[str]:
#     new_item_desc = []
#     for desc, color in zip(items_desc, items["colour_group_name"].values):
#         if color == "Other":
#             color = "Unknown"
#         if len(color.split(" ")) == 2 and color.split(" ")[0] == "Other":
#             color = color.split(" ")[1]
#         if not isinstance(desc, str):
#             new_item_desc.append("nan")
#         else:
#             if np.random.uniform() <= 0.7 and color != "Unknown":
#                 new_item_desc.append(f"{desc} {color} colored.")
#             else:
#                 new_item_desc.append(f"{desc}")
                
#     return new_item_desc

def main(args):
    number_gpu = args.number_gpu
    image_id = args.image_id if args.image_id is not None else number_gpu
    print(image_id)

    model_config={"number_gpu":number_gpu,
                "sd_version":"2.1",
                "scheduler" : "DPMSolverMultistep",
                "clip_model_name" : "patrickjohncyh/fashion-clip"}
    
    pipe = BatchGSDF(model_config).to(f"cuda:{number_gpu}")
    
    
    for i in range(0,len(items),BATCH_SIZE):
        print(f"start {i}")
        prompts = items[i:i+BATCH_SIZE]
        numbers = list(range(i,i+BATCH_SIZE))
        filename = f"basic_{args.image_id}"

        images = pipe.gen_img(prompts, width=WIDTH, height=HEIGHT, num_inference_steps=INFERENCE_STEP, return_dict=True)[0]
    
        os.makedirs(os.path.join(gen_img_save_path, filename), exist_ok=True)
        for num, img in zip(numbers, images):
            os.makedirs(os.path.join(gen_img_save_path, filename), exist_ok=True)
            img.save(os.path.join(gen_img_save_path, filename, f"item_{num}.jpg"))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--number_gpu", dest="number_gpu", action="store", required=False, default='0')
    parser.add_argument("-i", "--image_id", dest="image_id", action="store", required=False, default=1)
    args = parser.parse_args()
    
    os.makedirs(gen_img_save_path, exist_ok=True)
    os.makedirs(gen_latent_save_path, exist_ok=True)
    os.makedirs(gen_embed_save_path, exist_ok=True)
    
    main(args)