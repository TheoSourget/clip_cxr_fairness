from models.medclip.medclip import MedCLIP
from models.biomedclip.biomedclip import Biomedclip
from models.biovil.biovil import Biovil
from models.medimageinsight.medimageinsight import MedImageInsightWrapper
from models.chexzero.chexzero import Chexzero
from models.cxrclip.cxrclip_wrapper import Cxrclip

# from models.gloria.gloria_wrapper import Gloria
import argparse
import pandas as pd
import numpy as np
import tqdm
import torch
np.random.seed(1907)

def save_embeddings(image_paths,label_texts,model,batch_size,save_file):
    embeddings = {
        'img_embeds':[],
        'text_embeds':[]
    }
    for i in tqdm.tqdm(range(len(image_paths)//batch_size),total=len(image_paths)//batch_size):
        imgs_batch = image_paths[i*batch_size:(i+1)*batch_size]
        
        #Check whether text in lagel_texts are actual labels (then size of label_texts is different than the number of images) or reports (1 report per image)
        if len(label_texts) != len(image_paths):
            #Actual labels: same text for every batch
            texts_batch = label_texts
        else:
            #Reports: the texts are also batched
            texts_batch = label_texts[i*batch_size:(i+1)*batch_size]

        batch_embedding = model.get_embeddings(imgs_batch,texts_batch)
        embeddings['img_embeds'] += batch_embedding['img_embeds']
        
        #Only save the text embeddings once if the text are labels (same for every batch), otherwise add them at every batch
        if len(label_texts) != len(image_paths):
            embeddings['text_embeds'] = batch_embedding['text_embeds']
        else:
            embeddings['text_embeds'] += batch_embedding['text_embeds']

    #Extra step if needed for the last samples of the dataset
    if (len(image_paths)%batch_size != 0) and (len(image_paths)>batch_size):
        imgs_batch = image_paths[len(image_paths)-(len(image_paths)%batch_size):]
        if len(label_texts) != len(image_paths):
            texts_batch = label_texts
        else:
            texts_batch = label_texts[len(image_paths)-(len(image_paths)%batch_size):]
        batch_embedding = model.get_embeddings(imgs_batch,texts_batch)

        embeddings['img_embeds'] += batch_embedding['img_embeds']
        if len(label_texts) != len(image_paths):
            embeddings['text_embeds'] = batch_embedding['text_embeds']
        else:
            embeddings['text_embeds'] += batch_embedding['text_embeds']

    #Save the embeddings
    np.save(f"{save_file}_images",embeddings['img_embeds'])
    np.save(f"{save_file}_texts",embeddings['text_embeds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='medclip')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dataset', default='MIMIC')

    args, unknown = parser.parse_known_args()

    #Load the dataset
    #FOR MIMIC
    if args.dataset == "MIMIC":
        mimic_path = '/gpfs/workdir/shared/cpm4c/datasets/MIMIC/mimic/'
        df = pd.read_csv(f'{mimic_path}test_preproc_filtered.csv')
        df = df.dropna(subset=["findings"])
        df['path_preproc'] = df['path_preproc'].apply(lambda img_path:f'{mimic_path}{img_path}')
        image_paths = df['path_preproc'].to_numpy()[:]
        label_texts = df['findings'].to_list()[:]
    elif args.dataset == "CXR14":
        cxr14_path = "/gpfs/workdir/sourgetth/datasets/processed/CXR14"
        df = pd.read_csv(f'{cxr14_path}/processed_labels_alldrains.csv')
        df['path_preproc'] = df['Image Index'].apply(lambda img_path:f'{cxr14_path}/imgs/{img_path}')
        image_paths = df['path_preproc'].to_numpy()[:]
        label_texts = df['Finding Labels'].to_list()[:]

    with torch.no_grad():
        #Load the model
        if args.model_name == 'medclip':
            model = MedCLIP()
        elif args.model_name == 'biovil':
            model = Biovil(image_model="biovil")
        elif args.model_name == 'biovil-t':
            model = Biovil(image_model="biovil-t")
        elif args.model_name == 'medimageinsight':
            model = MedImageInsightWrapper()
        elif args.model_name == 'chexzero':
            model = Chexzero()
        elif args.model_name == 'cxrclip':
            model = Cxrclip()
        else:
            print('Unknown model name, choose in the following list: medclip,biovil,biovil-t,medimageinsight,chexzero,cxrclip')
            return
   
        #Compute and save the embeddings
        save_embeddings(image_paths,label_texts,model,args.batch_size,f'./data/embeddings/{args.dataset}/{args.dataset}_{args.model_name}')

    
if __name__ == "__main__":
    main()
