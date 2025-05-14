#Inspired by https://github.com/yctsai-hub/cxr-clip/blob/main/inference.py
import torch
import numpy as np
import pickle
import json
import argparse
from sklearn import metrics
from PIL import Image
from .cxrclip.data.data_utils import load_tokenizer, load_transform, transform_image
from .cxrclip.model import build_model


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
                text_embeddings.append(text_emb.tolist()[0])
                torch.cuda.empty_cache()


        #Image embeddings
        image_transforms = load_transform(split="test", transform_config=self.ckpt_config["transform"])
        images = []
        for p in image_paths:
            image = Image.open(p).convert("RGB")
            image = transform_image(image_transforms, image, normalize="huggingface")
            images.append(image)
        images = torch.stack(images).to(self.device)
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
