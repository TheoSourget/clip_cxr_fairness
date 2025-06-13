# Fairness and Robustness of CLIP-Based Models for Chest X-rays
This repo contain the code to evaluate six CLIP-based architectures for the classification of chest X-rays.

# How to install
Clone the repo, create your environment and install the dependencies using the following commands (you may need to change your pytorch version to fit your system):

```sh
#Clone the repo
git clone https://github.com/TheoSourget/clip_cxr_fairness.git

#Create a new python env
conda create --name clip_fairness python=3.10
conda activate clip_fairness

#Install the dependencies
pip install -r requirements.txt
``` 

or

```sh
#Clone the repo
git clone https://github.com/TheoSourget/clip_cxr_fairness.git

#Create the env and install the dependencies
make setup_env
``` 

## Models:
You will need to download pretrained weights before using some models
### CXR-CLIP
Download the model weight from their [original repo](https://github.com/Soombit-ai/cxr-clip?tab=readme-ov-file#pre-trained-model-checkpoint) (we used the ResNet50 M,M,C14) and place it in the pretrained/cxrclip folder

### CheXzero
1. Download the weights from [this link](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno) rename and place the file at pretrained/chexzero/clip_weights.pt 
2. Download the weights from [this link](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) and place the file in pretrained/chexzero/ViT-B-32.pt

## Data:

### MIMIC-CXR
TBA

### NIH-CXR14
Download the dataset from [this link](https://www.kaggle.com/datasets/nih-chest-xrays/data)
In the data folder place all the images from the orginal subfolders into a single data/CXR14/imgs folder
Place Data_Entry_2017.csv in the data/CXR14 folder

### NEATX
Download NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv from [this link](https://zenodo.org/records/14944064) and place it into the data/CXR14 folder

### Process the datasets
You can process all the data generating the original datasets with the command:
```sh
python process_datasets.py
```
The images will be resized to 224x224 and normalized.


To generate the drains label launch drains_detection.py script after processing the data with the previous command.

# How to use

## Get the embeddings
The script generate_embeddings.py can be used to generate the embeddings. An example using the following command:
```sh
python generate_embeddings.py --model_name medimageinsight --batch_size 32
``` 

the options are:
* **--model_name**: name of the model to apply. Choose between:
* **--batch_size**: The number of image to process at the same time. For some model the image will still be processed one by one
* **--image_folder**: Path to the folder containing the png images, if you followed the instruction the default value should be working.

## Get the probabilities
The script evaluate_performance.py compute the probabality for the labels defined within the file and saved them in data/probas_dataset/. It will also compute the AUC and AUCPR and save the results in data/performance/dataset/.
Here is an example to launch the script:

```sh
python evaluate_performance.py --model_name medimageinsight --batch_size 32
```
The options are:
the options are:
* **--model_name**: name of the model to apply. Choose between:
* **--batch_size**: The number of image to process at the same time. For some model the image will still be processed one by one
* **--dataset**: Name of the dataset you want to use, MIMIC or CXR14.


## Generate the tables and visualisations
To reproduce most of the tables and figures from the paper you can launch generate_figures_tables.py:
```sh
python generate_figures_tables.py
```

# Acknowledgement
This repo contains code from the base repo of the models, we want to thank the authors of these repos:
* [MedCLIP](https://github.com/RyanWangZf/MedCLIP)
* [MedImageInsight](https://huggingface.co/lion-ai/MedImageInsights)
* [Biovil and Biovil-t](https://github.com/microsoft/hi-ml/tree/e011bb996056f81e6cca98eae5b0f1223461dda1/hi-ml-multimodal)
* [CheXzero](https://github.com/rajpurkarlab/CheXzero)
* [CXR CLIP](https://github.com/Soombit-ai/cxr-clip)



**If you're using any of the model and/or dataset for research, please remember to cite the corresponding original papers following their authors guidelines.**
