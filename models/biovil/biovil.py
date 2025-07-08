from locale import normalize
import tempfile
from enum import Enum, unique
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd
import tqdm

import requests
from torchvision.datasets.utils import check_integrity
from torch.nn.functional import softmax
import torch.nn.functional as F

import torch
from sklearn.metrics import roc_auc_score,f1_score

from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.model.pretrained import get_biovil_image_encoder,get_biovil_t_image_encoder
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine

RESIZE = 512
CENTER_CROP_SIZE = 512

class Biovil():
    def __init__(self,image_model="biovil"):
        self.image_model = image_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._get_vlp_inference_engine()
        self.model.image_inference_engine.to(self.device)
        self.model.text_inference_engine.to(self.device)
        
    def _get_vlp_inference_engine(self) -> ImageTextInferenceEngine:
        if self.image_model == "biovil":
            image_inference = ImageInferenceEngine(
                image_model=get_biovil_image_encoder(),
                transform=create_chest_xray_transform_for_inference(resize=RESIZE, center_crop_size=CENTER_CROP_SIZE),
            )
            img_txt_inference = ImageTextInferenceEngine(
                image_inference_engine=image_inference,
                text_inference_engine=get_bert_inference(BertEncoderType.CXR_BERT),
            )
        elif self.image_model == "biovil-t":
            image_inference = ImageInferenceEngine(
                image_model=get_biovil_t_image_encoder(),
                transform=create_chest_xray_transform_for_inference(resize=RESIZE, center_crop_size=CENTER_CROP_SIZE),
            )
            img_txt_inference = ImageTextInferenceEngine(
                image_inference_engine=image_inference,
                text_inference_engine=get_bert_inference(BertEncoderType.BIOVIL_T_BERT),
            )
        else:
            raise Exception(f"Image model name is unknown, must be biovil or biovil-t, got {self.image_model}")
        
        
        return img_txt_inference
    
    def process_batch_images(self,image_paths):
        images = []
        for image_path in image_paths:
            images.append(self.model.image_inference_engine.load_and_transform_input_image(Path(image_path),self.model.image_inference_engine.transform)[0][0]) 
        images = torch.stack(images,dim=0).to(self.device)
        projected_img_embs = self.model.image_inference_engine.model.forward(images).projected_global_embedding
        for i,emb in enumerate(projected_img_embs):
            tmp_tensor = emb.tolist()
            tmp_tensor = torch.tensor([tmp_tensor])
            projected_img_embs[i] = F.normalize(tmp_tensor, dim=-1)
        return projected_img_embs

    def get_embeddings(self,image_paths,labels):

        # image_embeddings = []
        # for p in image_paths:
        #     image_embedding = self.model.image_inference_engine.get_projected_global_embedding(Path(p))        
        #     image_embeddings.append(image_embedding.tolist())
        image_embeddings = self.process_batch_images(image_paths).tolist()
        text_embeddings =   self.model.text_inference_engine.get_embeddings_from_prompt(labels, normalize=False).tolist()

        embeddings = {
            'img_embeds':image_embeddings,
            'text_embeds':text_embeddings
        }
        #Return the emddings
        return embeddings

    def get_predictions(self,image_paths,labels):
        text_embeddings =   self.model.text_inference_engine.get_embeddings_from_prompt(labels, normalize=False)
        image_embeddings = self.process_batch_images(image_paths)  
        cos_similarity = image_embeddings @ text_embeddings.t()
        print(cos_similarity)
        softmaxs = cos_similarity.softmax(dim=1)
        print(softmaxs)
        # softmaxs = []
        # for p in image_paths:
        #     logits_image = []
        #     for l in labels:
        #         image_path = Path(p)
        #         score = self.model.get_similarity_score_from_raw_data(image_path=image_path, query_text=l)
        #         logits_image.append(score)
        #     softmaxs.append(softmax(torch.tensor(logits_image)).tolist())

        return softmaxs.tolist()
    
    def get_embeddings_and_predictions(self,image_paths,labels):
        embeddings = self.get_embeddings(image_paths,labels)
        softmaxs = self.get_predictions(image_paths,labels)

        results = {
            'img_embeds':embeddings['img_embeds'],
            'text_embeds':embeddings['text_embeds'],
            'softmaxs':softmaxs
        }

        return results