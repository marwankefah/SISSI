import pickle
from sklearn.cluster import KMeans
import numpy as np
import cv2
from pathlib import Path
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image, ImageFilter

data_dir = Path("../data/cropped/inhib/inhib/")
inhib_images_raw = [
    [cv2.imread(str(img)), str(img).split('\\')[-1]] for img in data_dir.iterdir()
]
inhib_images_raw.remove([None, '.gitignore'])
color_range=[]
borders_pixels=[]
list_cell_names=[]
for image, cell_name in inhib_images_raw:
    list_cell_names.append(cell_name)
    # Extract color intensity range
    image_gray = rgb2gray(image)
    color_range.append(np.amax(image_gray)-np.amin(image_gray))

    # Applying the Canny Edge filter
    borders_pixels.append(sum(sum(cv2.Canny((image_gray*255).astype(np.uint8), 30, 100)/255)))

features_vector=list(zip(color_range,borders_pixels))

kmeans_inhib = KMeans(n_clusters=2, random_state=0).fit(features_vector)
print(sum(kmeans_inhib.labels_))

zip_iterator = zip(list_cell_names, kmeans_inhib.labels_)
zip_iterator = sorted(zip_iterator, key = lambda x: x[1])

dictionary_0={}
dictionary_1={}

for k in range(len(zip_iterator)):
    if zip_iterator[k][1]==0:
        dictionary_0[zip_iterator[k][0]]=0
    elif zip_iterator[k][1]==1:
        dictionary_1[zip_iterator[k][0]]=1

dictionary_0 = dict(dictionary_0)
dictionary_1 = dict(dictionary_1)
list_0=list(dictionary_0.keys())
list_1=list(dictionary_1.keys())
list_0=list_0[:1000]
list_1=list_1[:1000]

data_dir = Path("../data/cropped/inhib/inhib/")
for j in range(10):
    fig, axs = plt.subplots(2, 10)

    for i in range(10):
        axs[0, i].imshow(cv2.imread(str(data_dir)+"\\"+list_0[i+j*10]))
        axs[0, i].set_title('0')

    for i in range(10):
        axs[1, i].imshow(cv2.imread(str(data_dir)+"\\"+list_1[i+j*10]))
        axs[1, i].set_title('1')

for ax in axs.flat:
    ax.label_outer()

plt.show()