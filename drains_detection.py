import argparse

import cv2
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

from torchvision.models import densenet121
from torchvision.io import read_image, ImageReadMode


from sklearn.metrics import roc_auc_score
from torch.nn.functional import sigmoid
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

torch.manual_seed(1907)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TubesDataset(Dataset):
    def __init__(self, labels,data_path):
        self.labels = labels
        self.data_path = data_path

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.labels.iloc[idx]
        image = read_image(f"{self.data_path}/imgs/{sample['Image Index']}",ImageReadMode.RGB)
        image = torch.Tensor(image)
        
        label = sample["Drain"]
        label = torch.Tensor([label])
        return image, label
    
def training_epoch(model,criterion,optimizer,train_dataloader):
    model.to(DEVICE)
    model.train()
    train_loss = 0.0
    lst_labels = []
    lst_probas = []
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        output_sigmoid = sigmoid(outputs)
        lst_labels.extend(labels.cpu().detach().numpy())
        lst_probas.extend(output_sigmoid.cpu().detach().numpy())
    lst_labels = np.array(lst_labels)
    lst_probas = np.array(lst_probas)
    auc_score=roc_auc_score(lst_labels,lst_probas)
    print(f"train ({len(lst_labels)} images)",auc_score,flush=True)
    return train_loss/lst_labels.shape[0],auc_score

def test_epoch(model,criterion,valid_dataloader):
    model.to(DEVICE)
    model.eval()
    train_loss = 0.0
    lst_labels = []
    lst_probas = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            inputs, labels = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            output_sigmoid = sigmoid(outputs)
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_sigmoid.cpu().detach().numpy())
        lst_labels = np.array(lst_labels)
        lst_probas = np.array(lst_probas)
        auc_score=roc_auc_score(lst_labels,lst_probas)
        print(f"val ({len(lst_labels)} images)",auc_score,flush=True)
    return train_loss/lst_labels.shape[0],auc_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epochs', default=200,type=int)
    parser.add_argument('--batch_size', default=32,type=int)
    parser.add_argument('--lr', default=0.0001,type=float)
    parser.add_argument('--data_path', default="/gpfs/workdir/sourgetth/datasets/processed/CXR14")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction)
    parser.add_argument('--weights', default="./models/tube_detection_model.pt")

    args, unknown = parser.parse_known_args()

    model = densenet121(weights='DEFAULT')
    kernel_count = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
     torch.nn.Flatten(),
     torch.nn.Linear(kernel_count, 1)
    )
    if args.weights != "DEFAULT":
        model.load_state_dict(torch.load(args.weights,map_location=torch.device('cpu')))
    model.to(DEVICE)
    df_tubes = pd.read_csv(f"{args.data_path}/processed_labels.csv")
    if args.train:
        print("START TRAINING")
        criterion = torch.nn.BCEWithLogitsLoss()
        criterion.requires_grad = True
        optimizer = optim.Adam(model.parameters(),lr=args.lr)
        
        
        df_tubes_annotations = df_tubes[df_tubes["Drain"]!=-1]
        train_test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1907)
        for train_idx, test_idx in train_test_split.split(df_tubes_annotations, groups=df_tubes_annotations['Patient ID']):
            df_train = df_tubes_annotations.iloc[train_idx]
            df_test = df_tubes_annotations.iloc[test_idx]
            training_data = TubesDataset(labels=df_train,data_path=args.data_path)
            testing_data = TubesDataset(labels=df_test,data_path=args.data_path)
            
            train_dataloader = DataLoader(training_data, batch_size=args.batch_size)
            test_dataloader = DataLoader(testing_data, batch_size=args.batch_size)
            
            for epoch in tqdm(range(args.nb_epochs)):
                train_loss,train_metric = training_epoch(model,criterion,optimizer,train_dataloader)
                test_loss,test_metric = test_epoch(model,criterion,test_dataloader)

                torch.save(model.state_dict(),f'./models/drains_detection_model2.pt')
                print("Train metrics:",train_loss,train_metric)
                print("Test metrics:",test_loss,test_metric)

    df_tubes_to_annotate = df_tubes[df_tubes["Drain"]==-1]
    to_annotate_data = TubesDataset(labels=df_tubes_to_annotate,data_path=args.data_path)
    to_annotate_dataloader = DataLoader(to_annotate_data, batch_size=args.batch_size)
    lst_preds = []
    with torch.no_grad():
        for i, data in enumerate(to_annotate_dataloader, 0):
            inputs, labels = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
            outputs = model(inputs)
            output_sigmoid = sigmoid(outputs)
            lst_preds.extend(output_sigmoid.cpu().detach().numpy()>0.5)  
    df_tubes[df_tubes["Drain"]==-1]["Drain"] = lst_preds
    df_tubes.to_csv(f"{args.data_path}/processed_labels_alldrains.csv")
if __name__ == "__main__":
    main()
