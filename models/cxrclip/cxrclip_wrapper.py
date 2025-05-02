import torch
import numpy as np
import pickle
import json
import argparse
from sklearn import metrics
from PIL import Image
from .cxrclip.data.data_utils import load_tokenizer, load_transform, transform_image
from .cxrclip.model import build_model

# 修改 tokenizer 和 text_encoder 缓存目录为你本地的路径
# ckpt_config['tokenizer']['cache_dir'] = "D:/CXR_CLIP/Code/cache/tokenizer"
# ckpt_config['model']['text_encoder']['cache_dir'] = "D:/CXR_CLIP/Code/cache/text_encoder"


### load tokenizer ###
# tokenizer = load_tokenizer(**ckpt_config["tokenizer"])
# print(tokenizer)

# ### load model ###
# model = build_model(
#     model_config=ckpt_config["model"], loss_config=ckpt_config["loss"], tokenizer=tokenizer
# )
# model = model.to(device)
# model.load_state_dict(ckpt["model"], strict=False)
# model.eval()

### load texts from openi ###
# with open('openi_texts.json', 'r', encoding='utf-8') as f:
#     texts = json.load(f)

# texts = [
#     "Central venous catheter",
#     "Pleural effusion",
#     "Normal heart size",
#     "Intubation malposition",
#     "Atelectasis",
#     "Cardiomegaly",
#     "Consolidation",
#     "Pulmonary edema",
#     "No Finding",
#     "Mediastinum widening",
#     "Lung Lesion",
#     "Lung Opacity",
#     "Pneumonia",
#     "Pneumothorax",
#     "Pleural Other",
#     "Fracture",
#     "Support Devices",
#     "pleural effusion, Normal heart size",
#     "Pneumonia on the right side.",
#     "Pneumonia on the left side.",
#     "Pneumothorax on the right side.",
#     "Pneumothorax on the left side."
# ]

# text_tokens = []
# for t in texts:
#     text_tokens.append(tokenizer(t, padding="longest", truncation=True, return_tensors="pt", max_length=ckpt_config["base"]["text_max_length"]))
# print(np.array(text_tokens).shape)

# ### load image ###
# image_transforms = load_transform(split="test", transform_config=ckpt_config["transform"])
# image = Image.open(image_path).convert("RGB")
# image = transform_image(image_transforms, image, normalize="huggingface")
# print(image.shape)

# ### get image and text features ###
# image = image.unsqueeze(0)
# image_embeddings = model.encode_image(image.to(device))
# image_embeddings = model.image_projection(image_embeddings) if model.projection else image_embeddings
# image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=1, keepdim=True) # normalize
# image_embeddings = image_embeddings.detach().cpu().numpy()

# text_embeddings = []
# with torch.no_grad():
#     for tt in text_tokens:
#         text_emb = model.encode_text(tt.to(device))
#         text_emb = model.text_projection(text_emb) if model.projection else text_emb
#         text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True) # normalize
#         text_embeddings.append(text_emb.cpu())
#         torch.cuda.empty_cache()


class Cxrclip():
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model_path = "./pretrained/cxrclip/r50_mcc.tar"
        ckpt = torch.load(model_path, map_location="cpu")
        self.ckpt_config = ckpt["config"]
        self.ckpt_config['tokenizer']['cache_dir'] = "./pretrained/cxrclip/cache/tokenizer"
        self.ckpt_config['model']['text_encoder']['cache_dir'] = "./pretrained/cxrclip/cache/text_encoder"

        self.tokenizer = load_tokenizer(**self.ckpt_config["tokenizer"])
        self.model = build_model(
            model_config=self.ckpt_config["model"], loss_config=self.ckpt_config["loss"], tokenizer=self.tokenizer
        )
        self.model = self.model.to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()
    

    def get_embeddings(self,image_paths,labels):
        text_tokens = []

        with torch.no_grad():
            #Text embeddings
            for t in labels:
                text_tokens.append(self.tokenizer(t, padding="longest", truncation=True, return_tensors="pt", max_length=self.ckpt_config["base"]["text_max_length"]))

            text_embeddings = []
            for tt in text_tokens:
                text_emb = self.model.encode_text(tt.to(self.device))
                text_emb = self.model.text_projection(text_emb) if self.model.projection else text_emb
                text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True) # normalize
                text_embeddings.append(text_emb.tolist())
                torch.cuda.empty_cache()


        #Image embeddings
        image_transforms = load_transform(split="test", transform_config=self.ckpt_config["transform"])
        images = []
        for p in image_paths:
            image = Image.open(p).convert("RGB")
            image = transform_image(image_transforms, image, normalize="huggingface")
            images.append(image)
        images = torch.stack(images).to(self.device)
        print(images.size())
        images_embeddings = self.model.encode_image(images.to(self.device))
        images_embeddings = self.model.image_projection(images_embeddings) if self.model.projection else images_embeddings
        images_embeddings = images_embeddings / torch.norm(images_embeddings, dim=1, keepdim=True) # normalize
        images_embeddings = images_embeddings.tolist()
        embeddings = {'img_embeds':images_embeddings,'text_embeds':text_embeddings}
        return embeddings

    def get_predictions(self,image_paths,labels):
        pass

    def get_embeddings_predictions(self,image_paths,labels):
        pass
