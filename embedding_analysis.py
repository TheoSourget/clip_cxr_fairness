from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image


df = pd.read_csv('./data/stage_2_train_labels.csv')
df_unique = df.drop_duplicates(subset=['patientId'])[:10]
labels = df_unique['Target'].to_numpy()

nb_bounding_box_per_patient = df['patientId'].value_counts()[df_unique['patientId']]
nb_bounding_box_per_patient = [nb_bounding_box_per_patient[i] if labels[i]==1 else 0 for i in range(len(labels))]

embeddings = np.load('./data/embeddings/RSNA_images_cxrclip.npy')
print(embeddings.shape)
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeddings)

# Generate data x, y for scatter and an array of images.
arr = np.empty((1024,1024))


# create figure and plot scatter
fig = plt.figure(figsize=(50,50))
ax = fig.add_subplot(111)
line = plt.scatter(pca_embeddings[:,0],pca_embeddings[:,1],c=nb_bounding_box_per_patient,picker=True)
handles, labels = line.legend_elements()
plt.legend(handles = handles, labels = labels, title="Nb bounding box")

# create the annotations box
im = OffsetImage(arr, cmap='gray', zoom=5)
xybox=(50., 50.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
# add it to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)

def show_xray(event):
    ind = event.ind[0]
    w,h = fig.get_size_inches()*fig.dpi
    ws = (pca_embeddings[ind,0] > w/2.)*-1 + (pca_embeddings[ind,0] <= w/2.) 
    hs = (pca_embeddings[ind,1] > h/2.)*-1 + (pca_embeddings[ind,1] <= h/2.)
    # if event occurs in the top or right quadrant of the figure,
    # change the annotation box position relative to mouse.
    ab.xybox = (xybox[0]*ws, xybox[1]*hs)
    # make annotation box visible
    ab.set_visible(True)
    # place it at the position of the hovered scatter point
    ab.xy =(pca_embeddings[ind,0], pca_embeddings[ind,1])
    # set the image corresponding to that point
    
    path = f'./data/RSNA_png/{df_unique["patientId"].iloc[ind]}.png'
    img=Image.open(path)
    img.thumbnail((100,100),Image.LANCZOS)
    im.set_data(img)
    fig.canvas.draw_idle()

def hide_xray(event):
    if event.key == 'escape':
        ab.set_visible(False)
        fig.canvas.draw_idle()

# add callback for mouse moves
fig.canvas.mpl_connect('pick_event', show_xray)  
fig.canvas.mpl_connect('key_press_event', hide_xray)           

plt.show()