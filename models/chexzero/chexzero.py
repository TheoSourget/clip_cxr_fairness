import subprocess
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple

import torch
from torch.utils import data
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor,InterpolationMode

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

from .CheXzero import clip
from .CheXzero.model import CLIP
# from .CheXzero.eval import evaluate, plot_roc, accuracy, sigmoid, bootstrap, compute_cis


class CXRTestDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_paths: str, 
        transform = None, 
    ):
        super().__init__()
        self.img_paths = img_paths
        self.transform = transform
            
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):

        img = Image.open(self.img_paths[idx])

        if self.transform:
            img = self.transform(img)
                
        return img
    


class Chexzero():
    def __init__(self,model_path="./pretrained/chexzero/clip_weights.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_clip(model_path=model_path,pretrained=True)
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def load_clip(self,model_path, pretrained=False, context_length=77): 
        """
        FUNCTION: load_clip
        ---------------------------------
        """
        if pretrained is False: 
            # use new model params
            params = {
                'embed_dim':768,
                'image_resolution': 320,
                'vision_layers': 12,
                'vision_width': 768,
                'vision_patch_size': 16,
                'context_length': context_length, 
                'vocab_size': 49408,
                'transformer_width': 512,
                'transformer_heads': 8,
                'transformer_layers': 12
            }

            model = CLIP(**params)
        else: 
            model, preprocess = clip.load("./pretrained/chexzero/ViT-B-32.pt", device=self.device, jit=False) 
        try: 
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except: 
            print("Argument error. Set pretrained = True.", sys.exc_info()[0])
            raise
        return model


    def get_embeddings(self,image_paths,labels):
        dataset = CXRTestDataset(image_paths,self.transform)
        loader = torch.utils.data.DataLoader(dataset, shuffle=False)
        images_embeddings = []
        for i, images in enumerate(loader):
            batch_images_emb = self.model.encode_image(images)
            batch_images_emb /= batch_images_emb.norm(dim=-1, keepdim=True)
            images_embeddings += batch_images_emb.tolist()
            
        embeddings = {'img_embeds':images_embeddings,'text_embeds':[]}
        # processed_txt = self.model.process_class_prompts(labels, self.device)
        # processed_imgs = self.model.process_img(image_paths, self.device)
        # img_emb_l, img_emb_g = self.model.image_encoder_forward(processed_imgs)
        # text_embeddings = []
        # for cls_name, cls_txt in processed_txt.items():
        #     text_emb_l, text_emb_g, _ = self.model.text_encoder_forward(
        #         cls_txt["caption_ids"], cls_txt["attention_mask"], cls_txt["token_type_ids"]
        #     )
        #     text_emb_g = text_emb_g.detach().cpu().numpy().tolist()
        #     text_embeddings += text_emb_g

        # embeddings = {
        #     'img_embeds':img_emb_g,
        #     'text_embeds':text_embeddings
        # }
        
        #Return the emddings
        return embeddings 