from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
import seaborn as sns
import argparse
from pathlib import Path

#FOR RSNA dataset
# df = pd.read_csv('./data/stage_2_train_labels.csv')
# df_unique = df.drop_duplicates(subset=['patientId'])[:]
# labels = df_unique['Target'].to_numpy()

# def size_bb(row):
#     return row[1]["width"] * row[1]["height"]

# df["bounding_box_coverage"] = df["width"]*df["height"]
# df["bounding_box_coverage"] = df["bounding_box_coverage"].fillna(0)
# df["bounding_box_coverage_bin"] = pd.qcut(df['bounding_box_coverage'], q=5, labels=False)

# bb_coverage_per_patient = df.groupby("patientId").sum()["bounding_box_coverage"]
# nb_bounding_box_per_patient = df['patientId'].value_counts()[df_unique['patientId']]
# nb_bounding_box_per_patient = [nb_bounding_box_per_patient[i] if labels[i]==1 else 0 for i in range(len(labels))]
# embeddings_images = np.load('./data/embeddings/RSNA/RSNA_medimageinsight_images.npy')
# embeddings_texts = np.load('./data/embeddings/RSNA/RSNA_medimageinsight_texts.npy')


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='medclip')
parser.add_argument('--projection_type', default='PCA')

args, unknown = parser.parse_known_args()
model_name = args.model_name
projection_type = args.projection_type

df = pd.read_csv(f'./data/test_preproc.csv')

embeddings_images = np.load(f'./data/embeddings/MIMIC/MIMIC_{model_name}_images.npy')
embeddings_texts = np.load(f'./data/embeddings/MIMIC/MIMIC_{model_name}_texts.npy')

embeddings = np.concatenate((embeddings_images, embeddings_texts), axis=0)
embedding_type = ["images" if i < len(embeddings_images) else "texts" for i in range(len(embeddings))] #0 for image embeddings and 1 for text embeddings

if projection_type == "PCA":
    projection = PCA(n_components=2,random_state=1907)
elif projection_type == "TSNE":
    projection = TSNE(n_components=2, learning_rate="auto",random_state=1907)

Path(f"./reports/figures/{projection_type}/{model_name}").mkdir(parents=True, exist_ok=True)

projection_imagestexts = projection.fit_transform(embeddings)
projection_images = projection.fit_transform(embeddings_images)
projection_texts = projection.fit_transform(embeddings_texts)

sns.set_theme(style="white", palette=None)
fig = sns.jointplot(x=projection_imagestexts[:,0], y=projection_imagestexts[:,1], xlim=(min(-1,min(projection_imagestexts[:,0])),max(1,max(projection_imagestexts[:,0]))), ylim=(min(-1,min(projection_imagestexts[:,1])),max(1,max(projection_imagestexts[:,1]))), hue=embedding_type, kind='scatter', alpha=0.6, marker='o', s=40, hue_order=["images","texts"], joint_kws=dict(rasterized=True))
fig.ax_joint.legend(loc='upper right')
plt.xlabel(f"{projection_type} 1")
plt.ylabel(f"{projection_type} 2")
plt.savefig(f"./reports/figures/{projection_type}/{model_name}/{projection_type}_{model_name}_modality.png", bbox_inches='tight', dpi=300)



characteristics = {
    "sex":df["sex"].unique(),
    "race":df["race"].unique(),
    "disease":df["disease"].unique(),
    "ViewPosition":df["ViewPosition"].unique()
}

for charac in characteristics:

    lst_charac = df[charac].tolist()
    # if charac == "disease":
    #     lst_charac = ["Normal" if c == "No Finding" else "Abnormal" for c in lst_charac]

    #Both image and text embeddings
    sns.set_theme(style="white", palette=None)
    fig = sns.jointplot(x=projection_imagestexts[:,0], y=projection_imagestexts[:,1], xlim=(min(-1,min(projection_imagestexts[:,0])),max(1,max(projection_imagestexts[:,0]))), ylim=(min(-1,min(projection_imagestexts[:,1])),max(1,max(projection_imagestexts[:,1]))), hue=np.concatenate((lst_charac, lst_charac),axis=0), kind='scatter', alpha=0.6, marker='o', s=40, hue_order=characteristics[charac], joint_kws=dict(rasterized=True))
    fig.ax_joint.legend(loc='upper right')
    plt.xlabel(f"{projection_type} 1")
    plt.ylabel(f"{projection_type} 2")
    plt.savefig(f"./reports/figures/{projection_type}/{model_name}/{projection_type}_{model_name}_{charac.lower()}_imagestexts.png", bbox_inches='tight', dpi=300)


    #Only image embeddings
    sns.set_theme(style="white", palette=None)
    fig = sns.jointplot(x=projection_images[:,0], y=projection_images[:,1], xlim=(min(-1,min(projection_imagestexts[:,0])),max(1,max(projection_imagestexts[:,0]))), ylim=(min(-1,min(projection_imagestexts[:,1])),max(1,max(projection_imagestexts[:,1]))), hue=lst_charac, kind='scatter', alpha=0.6, marker='o', s=40, hue_order=characteristics[charac], joint_kws=dict(rasterized=True))
    fig.ax_joint.legend(loc='upper right')
    plt.xlabel(f"{projection_type} 1")
    plt.ylabel(f"{projection_type} 2")
    plt.savefig(f"./reports/figures/{projection_type}/{model_name}/{projection_type}_{model_name}_{charac.lower()}_images.png", bbox_inches='tight', dpi=300)


    #Only text embeddings
    sns.set_theme(style="white", palette=None)
    fig = sns.jointplot(x=projection_texts[:,0], y=projection_texts[:,1], xlim=(min(-1,min(projection_imagestexts[:,0])),max(1,max(projection_imagestexts[:,0]))), ylim=(min(-1,min(projection_imagestexts[:,1])),max(1,max(projection_imagestexts[:,1]))), hue=lst_charac, kind='scatter', alpha=0.6, marker='o', s=40, hue_order=characteristics[charac], joint_kws=dict(rasterized=True))
    fig.ax_joint.legend(loc='upper right')
    plt.xlabel(f"{projection_type} 1")
    plt.ylabel(f"{projection_type} 2")
    plt.savefig(f"./reports/figures/{projection_type}/{model_name}/{projection_type}_{model_name}_{charac.lower()}_texts.png", bbox_inches='tight', dpi=300)


# plt.colorbar()
# create the annotations box
# # Generate data x, y for scatter and an array of images.
# arr = np.empty((1024,1024))
# im = OffsetImage(arr, cmap='gray', zoom=5)
# xybox=(50., 50.)
# ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
#         boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
# # add it to the axes and make it invisible
# ax.add_artist(ab)
# ab.set_visible(False)

# def show_xray(event):
#     ind = event.ind[0]
#     w,h = fig.get_size_inches()*fig.dpi
#     ws = (pca_embeddings[ind,0] > w/2.)*-1 + (pca_embeddings[ind,0] <= w/2.) 
#     hs = (pca_embeddings[ind,1] > h/2.)*-1 + (pca_embeddings[ind,1] <= h/2.)
#     # if event occurs in the top or right quadrant of the figure,
#     # change the annotation box position relative to mouse.
#     ab.xybox = (xybox[0]*ws, xybox[1]*hs)
#     # make annotation box visible
#     ab.set_visible(True)
#     # place it at the position of the hovered scatter point
#     ab.xy =(pca_embeddings[ind,0], pca_embeddings[ind,1])
#     # set the image corresponding to that point
    
#     path = f'./data/RSNA_png/{df_unique["patientId"].iloc[ind]}.png'
#     img=Image.open(path)
#     img.thumbnail((100,100),Image.LANCZOS)
#     im.set_data(img)
#     fig.canvas.draw_idle()

# def hide_xray(event):
#     if event.key == 'escape':
#         ab.set_visible(False)
#         fig.canvas.draw_idle()

# # add callback for mouse moves
# fig.canvas.mpl_connect('pick_event', show_xray)  
# fig.canvas.mpl_connect('key_press_event', hide_xray)           