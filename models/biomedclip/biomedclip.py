import torch
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer


class Biomedclip():
    def __init__(self):
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.template = 'this is a photo of '
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        self.context_length = 256

    def get_embeddings(self,image_paths,labels):
        
        #Load the images
        images = torch.stack([self.preprocess(Image.open(img)) for img in image_paths]).to(self.device)
        texts = self.tokenizer([self.template + l for l in labels], context_length=self.context_length).to(self.device)
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(images, texts)

        embeddings = {
            'img_embeds':image_features.tolist(),
            'text_embeds':text_features.tolist()
        }
        #Return the emddings
        return embeddings

    def get_predictions(self,image_paths,labels):        
        images = torch.stack([self.preprocess(Image.open(img)) for img in image_paths]).to(self.device)
        texts = self.tokenizer([self.template + l for l in labels], context_length=self.context_length).to(self.device)
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(images, texts)

            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)

            logits = logits.tolist()
            sorted_indices = sorted_indices.tolist()

        #Return the logits
        return logits
    
    def get_embeddings_and_predictions(self,image_paths,labels):
        images = torch.stack([self.preprocess(Image.open(img)) for img in image_paths]).to(self.device)
        texts = self.tokenizer([self.template + l for l in labels], context_length=self.context_length).to(self.device)
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(images, texts)

            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            logits = logits.tolist()

        result = {
            'img_embeds':image_features.tolist(),
            'text_embeds':text_features.tolist(),
            'logits':logits
        }

        #Return the emdedings and logits
        return result
