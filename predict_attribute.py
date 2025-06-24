import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='biovil')
parser.add_argument('--projection_type', default='None')
parser.add_argument('--attribute', default='sex_label')
parser.add_argument('--modality', default='image')
parser.add_argument('--classification_head', default='mlp')


args, unknown = parser.parse_known_args()

mimic_path = './data/'
df = pd.read_csv(f'{mimic_path}test_preproc_filtered.csv')
df = df.dropna(subset=["findings"])
df['path_preproc'] = df['path_preproc'].apply(lambda img_path:f'{mimic_path}{img_path}')
image_paths = df['path_preproc'].to_numpy()[:]
df["age"] = df["age"].astype(int)
df["age_label"] = pd.cut(df["age"],bins=[18,25,50,65,80,np.inf],labels=["18-25","25-50","50-65","65-80","80+"],right=False)

#Load the embedding
if args.modality == 'image':
    embeddings= np.load(f'./data/embeddings/MIMIC/MIMIC_{args.model_name}_images.npy')
elif args.modality == 'text':
    embeddings= np.load(f'./data/embeddings/MIMIC/MIMIC_{args.model_name}_texts.npy')
elif args.modality == 'combined':
    #Load both image and text embeddings and add the type in the dataframe for possible classification on target
    embeddings_images = np.load(f'./data/embeddings/MIMIC/MIMIC_{args.model_name}_images.npy')
    embeddings_texts = np.load(f'./data/embeddings/MIMIC/MIMIC_{args.model_name}_texts.npy')
    embeddings = np.concatenate((embeddings_images, embeddings_texts), axis=0)
    embedding_type = np.array([0 if i < len(embeddings_images) else 1 for i in range(len(embeddings))]) #0 for image embeddings and 1 for text embeddings
    df = pd.concat([df,df],ignore_index=True)
    df["embedding_type"] = embedding_type
else:
    print('modality is not in the list of possible values [image, text, combined]')
    exit()

#A single train test split
train_test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1907)

#A single train/val split
train_valid_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1907)


if args.classification_head == 'mlp':
    clf_class = MLPClassifier
    #List of possible layer_size for the hyperparameter tuning
    hp_name = "hidden_layer_sizes"
    hyperparams = {hp_name:[(128,),(256,),(512,),(1024,)]}
elif args.classification_head == 'lp':
    hp_name = "learning_rate_init"
    hyperparams = {hp_name:[0.001,0.0001,0.00001]}
    params = {"hidden_layer_sizes":(),"alpha":0,"max_iter":1000}
    clf_class = MLPClassifier
else:
    clf_class = KNeighborsClassifier
    #List of possible k for the hyperparameter tuning
    hp_name = "n_neighbors"
    hyperparams = {hp_name:[1,3,5,7,9,11,13,15,17,19,21]}
    params = None


for train_val_idx, test_idx in train_test_split.split(df, groups=df['subject_id']):
    df_train_val = df.iloc[train_val_idx]
    x_train_val = embeddings[train_val_idx]
    y_train_val = df_train_val[args.attribute].to_numpy()

    df_test = df.iloc[test_idx]
    x_test = embeddings[test_idx]
    y_test = df_test[args.attribute].to_numpy()

    lst_auc_test = []
    for train_idx, val_idx in train_valid_split.split(df_train_val, groups=df_train_val['subject_id']):
        df_train = df_train_val.iloc[train_idx]
        x_train = embeddings[train_idx]
        y_train = df_train[args.attribute].to_numpy()
        

        df_val = df_train_val.iloc[val_idx]
        x_val = embeddings[val_idx]
        y_val = df_val[args.attribute].to_numpy()
        
        #Apply PCA or TSNE if chosen in the call parameter
        if args.projection_type == "PCA":
            projection = PCA(n_components=2,random_state=1907)
            projection.fit(x_train)
            x_train = projection.transform(x_train)
            x_val = projection.transform(x_val)
        elif args. projection_type == "TSNE":
            projection = TSNE(n_components=2, learning_rate="auto",random_state=1907)
            projection.fit(x_train)
            x_train = projection.transform(x_train)
            x_val = projection.transform(x_val)
        
        lst_auc_val = []
        #select best hyperparam
        for p in hyperparams[hp_name]:
            if params:
                clf = clf_class(**params,**{hp_name:p})
            else:
                clf = clf_class(**{hp_name:p})
            clf.fit(x_train,y_train)
            probas_val = clf.predict_proba(x_val)
            pred_val = np.argmax(probas_val, axis=1)

            if probas_val.shape[1] == 2: #Case of binary classification
                probas_val = probas_val[:,1]

            auc = roc_auc_score(y_val,probas_val,multi_class='ovr')
            lst_auc_val.append(auc)
        
        #Retrain on best param
        best_p = hyperparams[hp_name][np.argmax(lst_auc_val)]
        if params:
            clf = clf_class(**params,**{hp_name:best_p})
        else:
            clf = clf_class(**{hp_name:best_p})
        clf.fit(x_train_val,y_train_val)

        # Apply PCA or TSNE if chosen in the call parameter
        if args.projection_type == "PCA" or args.projection_type == "TSNE":
            x_test_projected = projection.transform(x_test)
            probas_test = clf.predict_proba(x_test_projected)
        else:
            probas_test = clf.predict_proba(x_test)
        
        if probas_test.shape[1] == 2:
            probas_test = probas_test[:,1]
        
        #Apply to test set and get AUC
        auc = roc_auc_score(y_test,probas_test,multi_class='ovr')
        lst_auc_test.append(auc)
    print(np.mean(lst_auc_test))