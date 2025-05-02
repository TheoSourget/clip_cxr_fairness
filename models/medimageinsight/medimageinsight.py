from .medimageinsightmodel import MedImageInsight
from PIL import Image
import base64
import torch
import tqdm

class MedImageInsightWrapper():
    def __init__(self):
        self.model = MedImageInsight(
                model_dir="models/medimageinsight/2024.09.27",
                vision_model_name="medimageinsigt-v1.0.0.pt",
                language_model_name="language_model.pth"
            )
        self.model.load_model()
        self.model.model.eval()


    def read_image(self,image_path):
        with open(image_path, "rb") as f:
            return f.read()


    def get_embeddings(self,image_paths,labels):
        with torch.no_grad():
            text_embeddings = self.model.encode(texts=labels)['text_embeddings'].tolist()
            image_embeddings = []
            images = []
            for p in image_paths:
                image = base64.encodebytes(self.read_image(p)).decode("utf-8")
                images.append(image)
            image_embeddings = self.model.encode(images=images)['image_embeddings'].tolist()    

        embeddings = {
            'img_embeds':image_embeddings,
            'text_embeds':text_embeddings
        }
        #Return the emddings
        return embeddings

    def get_predictions(self,image_paths,labels):
        softmaxs = []
        images = []
        with torch.no_grad():
            for p in image_paths:
                image = base64.encodebytes(self.read_image(p)).decode("utf-8")
                images.append(image)
            result = self.model.predict(images, labels)[0]
            for r in result:
                softmaxs += [[r[l] for l in labels]]
        #Return the emddings   
        return softmaxs
    
    def get_embeddings_and_predictions(self,image_paths,labels):
        embeddings = self.get_embeddings(image_paths,labels)
        softmaxs = self.get_predictions(image_paths,labels)

        results = {
            'img_embeds':embeddings['img_embeds'],
            'text_embeds':embeddings['text_embeds'],
            'softmaxs':softmaxs
        }
        return results
