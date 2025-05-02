import torch
import gloria
import pandas as pd 
import glob
from sklearn.metrics import roc_auc_score,f1_score

# # load model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# gloria_model = gloria.load_gloria(device=device)

# gloria_model.eval()
# df = pd.read_csv('../data/stage_2_train_labels.csv')
# df['path'] = df['patientId'].apply(lambda id:f'../data/RSNA/{id}.dcm')
# # generate class prompt
# # cls_promts = {
# #    'Atelectasis': ['minimal residual atelectasis ', 'mild atelectasis' ...]
# #    'Cardiomegaly': ['cardiomegaly unchanged', 'cardiac silhouette enlarged' ...] 
# # ...
# # } 
# #cls_prompts = gloria.generate_chexpert_class_prompts()

# cls_prompts = {
#     'Healthy':[
#         'There is no pneumonia',
#         'No evidence of pneumonia',
#         'No evidence of acute pneumonia',
#         'No signs of pneumonia',
#     ],
#     'Pneumonia':[
#         'Findings consistent with pneumonia',
#         'Findings suggesting pneumonia',
#         'This opacity can represent pneumonia',
#         'Findings are most compatible with pneumonia',         
#     ]
# }

# paths = df['path'].to_numpy()[:150]
# labels = df['Target'].to_numpy()[:150]

# # process input images and class prompts 
# processed_txt = gloria_model.process_class_prompts(cls_prompts, device)
# processed_imgs = gloria_model.process_img(paths, device)


# similarities = gloria.zero_shot_classification(gloria_model, processed_imgs, processed_txt)

# probas = torch.nn.functional.softmax(torch.tensor(similarities[cls_prompts.keys()].to_numpy()))
# preds = similarities[cls_prompts.keys()].to_numpy().argmax(axis=1)

# print(labels)
# print(preds)
# #print(probas)
# print(sum(preds==labels)/len(labels),f1_score(labels,preds))

class Gloria():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = gloria.load_gloria(device=self.device)


    def get_embeddings(self,image_paths,labels):
        processed_txt = self.model.process_class_prompts(labels, self.device)
        processed_imgs = self.model.process_img(image_paths, self.device)
        img_emb_l, img_emb_g = self.model.image_encoder_forward(processed_imgs)
        text_embeddings = []
        for cls_name, cls_txt in processed_txt.items():
            text_emb_l, text_emb_g, _ = self.model.text_encoder_forward(
                cls_txt["caption_ids"], cls_txt["attention_mask"], cls_txt["token_type_ids"]
            )
            text_emb_g = text_emb_g.detach().cpu().numpy().tolist()
            text_embeddings += text_emb_g

        embeddings = {
            'img_embeds':img_emb_g,
            'text_embeds':text_embeddings
        }
        
        #Return the emddings
        return embeddings

    def get_predictions(self,image_paths,labels):        
        return
    
    def get_embeddings_and_predictions(self,image_paths,labels):
        return