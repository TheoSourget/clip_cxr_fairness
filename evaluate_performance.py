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
        #Get paths of images for this batch
        imgs_batch = image_paths[i*batch_size:(i+1)*batch_size]

        for label in label_texts:
            #Probabilities for a label will be computed using the comparison to the "Chest No findings" text.
            templates = [f"Chest {label}",f"Chest No findings"]
            probas = model.get_predictions(imgs_batch,templates)
            lst_probas[label] = np.append(lst_probas[label],np.array(probas)[:,0])

    #Extra steps if needed
    if (len(image_paths)%batch_size != 0) and (len(image_paths)>batch_size):
        imgs_batch = image_paths[len(image_paths)-(len(image_paths)%batch_size):]
        for label in label_texts:
            templates = [f"Chest {label}",f"Chest No findings"]
            probas = model.get_predictions(imgs_batch,templates)
            lst_probas[label] = np.append(lst_probas[label],np.array(probas)[:,0])

    return lst_probas

def adjusted_auprc(y_true,y_proba):
    auprc = average_precision_score(y_true,y_proba)
    baseline_auprc = sum(y_true)/len(y_true)
    return 1 - (np.log(auprc)/np.log(baseline_auprc))

def bootstrap(labels,probas,metric,nb_selection=1000):
    #Taken from https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
    scores = []
    for _ in range(nb_selection):
        indices = np.random.randint(0, len(labels), len(labels))
        labels_selection = labels[indices]
        #Skip if the random selection resulted in only negative samples
        if sum(labels_selection) == 0:
            continue
        probas_selection = probas[indices]
        score = metric(labels_selection,probas_selection)
        scores.append(score)

    alpha = 5.0
    lower_p = alpha / 2.0
    lower = np.percentile(scores, lower_p)
    upper_p = (100 - alpha) + (alpha / 2.0)
    upper = np.percentile(scores, upper_p)
    return lower, upper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='medclip')
    parser.add_argument('--dataset', default='MIMIC')
    parser.add_argument('--batch_size', default=32, type=int)

    args, unknown = parser.parse_known_args()

    #Load the dataset and define which labels and attributes to use for the evaluation
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
            'Pleural Effusion',
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
        # #Load the model
        # if args.model_name == 'medclip':
        #     model = MedCLIP()
        # elif args.model_name == 'biovil':
        #     model = Biovil(image_model="biovil")
        # elif args.model_name == 'biovil-t':
        #     model = Biovil(image_model="biovil-t")
        # elif args.model_name == 'medimageinsight':
        #     model = MedImageInsightWrapper()
        # elif args.model_name == 'chexzero':
        #     model = Chexzero()
        # elif args.model_name == 'cxrclip':
        #     model = Cxrclip()
        # else:
        #     print('Unknown model name, choose in the following list: medclip,biovil,biovil-t,medimageinsight,chexzero,cxrclip')
        #     return
    
        # #Get the probabilities for each sample
        # print("Computing predictions")
        # lst_probas = compute_probas(image_paths,labels,model,args.batch_size)
        # for label in labels:
        #     df[f"proba_{label}"] = lst_probas[f"{label}"]
        
        # #Save the probabilities
        # Path(f"./data/probas_{args.dataset}/").mkdir(parents=True, exist_ok=True)
        # df.to_csv(f"./data/probas_{args.dataset}/probas_{args.dataset}_{args.model_name}.csv")

        #Load the proababilites (in case you saved them before, you can comment line 97-122)
        df = pd.read_csv(f"./data/probas_{args.dataset}/probas_{args.dataset}_{args.model_name}.csv")
        if args.dataset == "MIMIC":
            df["age"] = df["age"].astype(int)
            df["age_group"] = pd.cut(df["age"],bins=[18,25,50,65,80,np.inf],labels=["18-25","25-50","50-65","65-80","80+"],right=False)
            df = df.sort_values(by=["age"])
        
        #Compute the AUC and AUPRC and confidence interval for the whole dataset and then per the subgroups in the attribute defined before
        Path(f"./data/performance/{args.dataset}/").mkdir(parents=True, exist_ok=True)
        with open(f"./data/performance/{args.dataset}/zeroshot_{args.model_name}.csv","w") as perf_file:
            perf_file.write("class,group,AUC,CI_AUC_low,CI_AUC_up,AUPRC,CI_AUPRC_low,CI_AUPRC_up,baseline_auprc")
            for label in labels:
                #Global performance per label
                y_true = df[label].fillna(0).to_numpy()
                y_proba = df[f"proba_{label}"].to_numpy()

                auc = roc_auc_score(y_true,y_proba)
                auprc = adjusted_auprc(y_true,y_proba)
                baseline_auprc = sum(y_true)/len(y_true)
                ci_auc = bootstrap(y_true,y_proba,roc_auc_score)
                ci_auprc = bootstrap(y_true,y_proba,adjusted_auprc)
                perf_file.write(f"\n{label},global,{round(auc,2)},{round(ci_auc[0],2)},{round(ci_auc[1],2)},{round(auprc,2)},{round(ci_auprc[0],2)},{round(ci_auprc[1],2)},{round(baseline_auprc,2)}")
                
                if args.dataset == "MIMIC":
                    #Performance per label and subgroup
                    for attribute in attributes:
                        attribute_values = df[attribute].unique()
                        for subgroup in attribute_values:
                            df_subgroup = df[df[attribute]==subgroup]
                            y_true = df_subgroup[label].fillna(0).to_numpy()
                            y_proba = df_subgroup[f"proba_{label}"].to_numpy()
                            auc = roc_auc_score(y_true,y_proba)
                            auprc = adjusted_auprc(y_true,y_proba)
                            baseline_auprc = sum(y_true)/len(y_true)
                            ci_auc = bootstrap(y_true,y_proba,roc_auc_score)
                            ci_auprc = bootstrap(y_true,y_proba,adjusted_auprc)
                            perf_file.write(f"\n{label},{subgroup},{round(auc,2)},{round(ci_auc[0],2)},{round(ci_auc[1],2)},{round(auprc,2)},{round(ci_auprc[0],2)},{round(ci_auprc[1],2)},{round(baseline_auprc,2)}")

                if args.dataset == "CXR14":
                    pneumothorax_only_drains = df[(df["Pneumothorax"])&(df["Drain"] == 1)]
                    pneumothorax_no_drains = df[(df["Pneumothorax"])&(df["Drain"] == 0)]
                    # healthy_no_drains = df[(~df["Pneumothorax"])&(df["Drain"] == 0)]
                    healthy = df[(~df["Pneumothorax"])]

                    #Without drains
                    df_eval = pd.concat([pneumothorax_no_drains,healthy])
                    y_true = df_eval["Pneumothorax"].fillna(0).to_numpy()
                    y_proba = df_eval["proba_Pneumothorax"].to_numpy()
                    auc = roc_auc_score(y_true,y_proba)
                    auprc = adjusted_auprc(y_true,y_proba)
                    baseline_auprc = sum(y_true)/len(y_true)
                    ci_auc = bootstrap(y_true,y_proba,roc_auc_score)
                    ci_auprc = bootstrap(y_true,y_proba,adjusted_auprc)
                    perf_file.write(f"\n{label},0.0,{round(auc,2)},{round(ci_auc[0],2)},{round(ci_auc[1],2)},{round(auprc,2)},{round(ci_auprc[0],2)},{round(ci_auprc[1],2)},{round(baseline_auprc,2)}")

                    #With drains
                    df_eval = pd.concat([pneumothorax_only_drains,healthy])
                    y_true = df_eval["Pneumothorax"].fillna(0).to_numpy()
                    y_proba = df_eval["proba_Pneumothorax"].to_numpy()
                    auc = roc_auc_score(y_true,y_proba)
                    auprc = adjusted_auprc(y_true,y_proba)
                    baseline_auprc = sum(y_true)/len(y_true)
                    ci_auc = bootstrap(y_true,y_proba,roc_auc_score)
                    ci_auprc = bootstrap(y_true,y_proba,adjusted_auprc)
                    perf_file.write(f"\n{label},1.0,{round(auc,2)},{round(ci_auc[0],2)},{round(ci_auc[1],2)},{round(auprc,2)},{round(ci_auprc[0],2)},{round(ci_auprc[1],2)},{round(baseline_auprc,2)}")


if __name__ == "__main__":
    main()
