from cProfile import label
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
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

np.random.seed(1907)


def compute_probas(image_paths,label_texts,model,batch_size):
    lst_probas = {label:[] for label in label_texts}
    for i in tqdm.tqdm(range(len(image_paths)//batch_size),total=len(image_paths)//batch_size):
        imgs_batch = image_paths[i*batch_size:(i+1)*batch_size]
        for label in label_texts:
            templates = [f"Chest {label}",f"Chest No findings"]
            probas = model.get_predictions(imgs_batch,templates)
            lst_probas[label] = np.append(lst_probas[label],np.array(probas)[:,0])

    if (len(image_paths)%batch_size != 0) and (len(image_paths)>batch_size):
        imgs_batch = image_paths[len(image_paths)-(len(image_paths)%batch_size):]
        for label in label_texts:
            templates = [f"Chest {label}",f"Chest No findings"]
            probas = model.get_predictions(imgs_batch,templates)
            lst_probas[label] = np.append(lst_probas[label],np.array(probas)[:,0])

    return lst_probas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='medclip')
    parser.add_argument('--dataset', default='MIMIC')
    parser.add_argument('--batch_size', default=32, type=int)

    args, unknown = parser.parse_known_args()

    #FOR MIMIC
    if args.dataset == "MIMIC":
        mimic_path = './data/'
        df = pd.read_csv(f'{mimic_path}/test_preproc_filtered.csv')
        df = df.dropna(subset=["findings"])
        df['path_preproc'] = df['path_preproc'].apply(lambda img_path:f'{mimic_path}{img_path}')
        labels = [
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Enlarged Cardiomediastinum',
            'Fracture',
            'Lung Lesion',
            'Lung Opacity',
            'Pleural Effusion',
            'Pleural Other',
            'Pneumonia',
            'Pneumothorax',
        ]
        attributes = ["sex","race","age_group"]

    elif args.dataset == "CXR14":
        cxr14_path = "./data/processed/CXR14"
        df = pd.read_csv(f'{cxr14_path}/processed_labels_alldrains.csv')
        df['path_preproc'] = df['Image Index'].apply(lambda img_path:f'{cxr14_path}/imgs/{img_path}')
        labels = [
            'Pneumothorax',
        ]
        attributes = ["Drain"]


    image_paths = df['path_preproc'].to_numpy()[:]
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
   
        print("Computing predictions")
        lst_probas = compute_probas(image_paths,labels,model,args.batch_size)
        for label in labels:
            df[f"proba_{label}"] = lst_probas[f"{label}"]
        
        Path(f"./data/probas_{args.dataset}/").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"./data/probas_{args.dataset}/probas_{args.dataset}_{args.model_name}.csv")

        df = pd.read_csv(f"./data/probas_{args.dataset}/probas_{args.dataset}_{args.model_name}.csv")
        # df["age"] = df["age"].astype(int)
        # df["age_group"] = pd.cut(df["age"],bins=[18,25,50,65,80,np.inf],labels=["18-25","25-50","50-65","65-80","80+"],right=False)
        # df = df.sort_values(by=["age"])
        
        Path(f"./data/performance/{args.dataset}/").mkdir(parents=True, exist_ok=True)
        with open(f"./data/performance/{args.dataset}/zeroshot_{args.model_name}.csv","w") as perf_file:
            perf_file.write("class,group,AUC,AUPRC")
            for label in labels:
                y_true = df[label].fillna(0)
                y_proba = df[f"proba_{label}"]
                auc = roc_auc_score(y_true,y_proba)
                auprc = average_precision_score(y_true,y_proba)
                perf_file.write(f"\n{label},global,{round(auc,2)},{round(auprc,2)}")
                # df[["dicom_id",label,f"proba_{label}"]].fillna(0).sort_values(by=[f"proba_{label}"]).to_csv(f"./data/ordered_proba__{args.model_name}_{label}.csv")
                for attribute in attributes:
                    attribute_values = df[attribute].unique()
                    for subgroup in attribute_values:
                        df_subgroup = df[df[attribute]==subgroup]
                        y_true = df_subgroup[label].fillna(0)
                        y_proba = df_subgroup[f"proba_{label}"]
                        auc = roc_auc_score(y_true,y_proba)
                        auprc = average_precision_score(y_true,y_proba)
                        perf_file.write(f"\n{label},{subgroup},{round(auc,2)},{round(auprc,2)}")
        # print(f"COMPUTING AUC FOR {args.model_name}:")
        # for label in labels:
        #     y_true = df[label].fillna(0)
        #     y_pred = df[f"proba_{label}"]
        #     auc = roc_auc_score(y_true,y_pred)
        #     print(label,auc)

if __name__ == "__main__":
    main()
