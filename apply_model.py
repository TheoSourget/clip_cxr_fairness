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
        
        if len(label_texts) != len(image_paths):
            embeddings['text_embeds'] = batch_embedding['text_embeds']
        else:
            embeddings['text_embeds'] += batch_embedding['text_embeds']

    
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

    np.save(f"{save_file}_images",embeddings['img_embeds'])
    np.save(f"{save_file}_texts",embeddings['text_embeds'])

    return


def extract_text(report):
    if "FINDINGS:" in report:
        sentences = report.split("FINDINGS:")[-1].split(".")
    elif "IMPRESSION:" in report:
        sentences = report.split("IMPRESSION:")[-1].split(".")
    else:
        sentences = report.split(".")
    
    if len(sentences) == 1:
        return sentences[0]
     
    
    random_sentence_idx = np.random.randint(max(1,len(sentences)-3))
    return ".".join(sentences[random_sentence_idx:random_sentence_idx+3])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='medclip')
    parser.add_argument('--image_folder', default='./data/RSNA_png/')
    parser.add_argument('--batch_size', default=1, type=int)

    args, unknown = parser.parse_known_args()

    #FOR RSNA
    # df = pd.read_csv('./data/stage_2_train_labels.csv')
    # df_unique = df.drop_duplicates(subset=['patientId'])

    # df_unique['path'] = df_unique['patientId'].apply(lambda id:f'{args.image_folder}{id}.png')
    # image_paths = df_unique['path'].to_numpy()[:]

    #FOR MIMIC
    mimic_path = './data/'
    df = pd.read_csv(f'{mimic_path}test_preproc.csv')
    df['path_preproc'] = df['path_preproc'].apply(lambda img_path:f'{mimic_path}{img_path}')
    image_paths = df['path_preproc'].to_numpy()[:]
    label_texts = df['report'].apply(extract_text).to_list()[:]
    df['processed_reports'] = label_texts
    df.to_csv(f'{mimic_path}preproc_reports_mimic.csv')
    return 
    # label_texts = [
    #     'Healthy',
    #     'Abnormal',
    # ]
    with torch.no_grad():
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
   
        # result = model.get_predictions(image_paths,label_texts)
        # result = model.get_embeddings(image_paths,label_texts)
        save_embeddings(image_paths,label_texts,model,args.batch_size,f'./data/embeddings/MIMIC_{args.model_name}')
        # print(result['img_embeds'],len(result['text_embeds']))

    
if __name__ == "__main__":
    main()
