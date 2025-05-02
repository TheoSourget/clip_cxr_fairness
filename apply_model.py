from models.medclip.medclip import MedCLIP
from models.biomedclip.biomedclip import Biomedclip
from models.biovil.biovil import Biovil
from models.medimageinsight.medimageinsight import MedImageInsightWrapper
from models.chexzero.chexzero import Chexzero
# from models.gloria.gloria_wrapper import Gloria
import argparse
import pandas as pd
import numpy as np
import tqdm
import torch

def save_embeddings(image_paths,label_texts,model,batch_size,save_file):
    embeddings = {
        'img_embeds':[],
        'text_embeds':[]
    }
    for i in tqdm.tqdm(range(len(image_paths)//batch_size),total=len(image_paths)//batch_size):
        batch_embedding = model.get_embeddings(image_paths[i*batch_size:(i+1)*batch_size],label_texts)
        embeddings['img_embeds'] += batch_embedding['img_embeds']
        if i == 0:
            embeddings['text_embeds'] = batch_embedding['text_embeds']
    if len(image_paths)%batch_size != 0:
        embeddings['img_embeds'] += model.get_embeddings(image_paths[len(image_paths)-(len(image_paths)%batch_size):],label_texts)['img_embeds']

    np.save(save_file,embeddings['img_embeds'])
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='medclip')
    parser.add_argument('--image_folder', default='./data/RSNA_png/')
    parser.add_argument('--batch_size', default=1, type=int)

    args, unknown = parser.parse_known_args()

    df = pd.read_csv('./data/stage_2_train_labels.csv')
    df_unique = df.drop_duplicates(subset=['patientId'])

    df_unique['path'] = df_unique['patientId'].apply(lambda id:f'{args.image_folder}{id}.png')
    image_paths = df_unique['path'].to_numpy()[:10]
    label_texts = [
        'Healthy',
        'Pneumonia',
    ]

    if args.model_name == 'medclip':
        model = MedCLIP()
    elif args.model_name == 'biomedclip':
        model = Biomedclip()
    elif args.model_name == 'biovil':
        model = Biovil(image_model="biovil")
    elif args.model_name == 'biovil-t':
        model = Biovil(image_model="biovil-t")
    elif args.model_name == 'medimageinsight':
        model = MedImageInsightWrapper()
    elif args.model_name == 'chexzero':
        model = Chexzero()
    # elif args.model_name == 'gloria':
    #     label_texts = {
    #         'Healthy':[
    #             'There is no pneumonia',
    #             'No evidence of pneumonia',
    #             'No evidence of acute pneumonia',
    #             'No signs of pneumonia',
    #         ],
    #         'Pneumonia':[
    #             'Findings consistent with pneumonia',
    #             'Findings suggesting pneumonia',
    #             'This opacity can represent pneumonia',
    #             'Findings are most compatible with pneumonia',         
    #         ]
    #     }
    #     model = Gloria()
    else:
        print('Unknown model name, choose in the following list: medclip,biomedclip,gloria,biovil,biovil-t,medimageinsight,chexzero')
        return
    
    
    # result = model.get_predictions(image_paths,label_texts)
    # result = model.get_embeddings(image_paths,label_texts)
    save_embeddings(image_paths,label_texts,model,args.batch_size,f'./data/embeddings/RSNA_images_{args.model_name}')
    # print(result['img_embeds'],len(result['text_embeds']))

    
if __name__ == "__main__":
    main()
