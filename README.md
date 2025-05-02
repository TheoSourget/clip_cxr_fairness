# clip_cxr_wrapper
This repository wrap multiple CLIP-based VLM models in a uniform way to obtain embeddings or predictions. This code doesn't contain any training or fine-tuning script as it only apply pretrained VLMs. 

# How to install
Clone the repo, create your environment and install the dependencies using the following commands (you may need to change your pytorch version to fit your system):

```sh
#Clone the repo
git clone https://github.com/TheoSourget/clip_cxr_wrapper.git

#Create a new python env
conda create --name vlm python=3.10
conda activate vlm

#Install the dependencies
pip install -r requirements.txt
``` 

## Models:
For some model, you'll need to download some pretrained weights:
TBA

## Data:

This code was applied to the RSNA pneumonia dataset, please download the dataset from its official challenge on [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data).
Download the stage_2_train_labels.csv and the stage_2_train_images folder. Put them in the data folder.

Convert the dicom files to png:

``` sh
python dicom_to_png.py
``` 

# How to use

## Get the embeddings
The script apply_model.py gives examples on how to apply the models. Here is an example of command to use the script:
```sh
python apply_model.py --model_name medimageinsight --batch_size 5
``` 

the options are:
* **--model_name**: name of the model to apply. Choose between:
* **--batch_size**: The number of image to process at the same time. For some model the image will still be processed one by one
* **--image_folder**: Path to the folder containing the png images, if you followed the instruction the default value should be working.

# Acknowledgement
This repo contains code from the other repositories, we want to thank the authors of these repos:
* [Biomedclip](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
* [Medclip](https://github.com/RyanWangZf/MedCLIP)
* [Medimageinsight](https://huggingface.co/lion-ai/MedImageInsights)
* [Biovil](https://github.com/microsoft/hi-ml/tree/e011bb996056f81e6cca98eae5b0f1223461dda1/hi-ml-multimodal)

**If you're using any of the model and/or dataset for research, please remember to cite the corresponding original papers following their authors guidelines.**