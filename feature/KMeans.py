import pickle
from sklearn.cluster import KMeans
import numpy as np
import cv2
from pathlib import Path
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

filename = "gabor_features.pkl"
infile =open(filename,"rb")
l=pickle.load(infile)
infile.close()

all_cells_names_list=l[1]
all_images_with_features=l[0].numpy()

features_alive=[]
features_alive_names=[]
features_dead=[]
features_dead_names=[]
features_inhib=[]
features_inhib_names=[]

for i in range(len(all_cells_names_list)):
    if "alive" in all_cells_names_list[i][0]:
        features_alive.append(all_images_with_features[i])
        features_alive_names.append(all_cells_names_list[i][0].removeprefix('data/cropped/alive/alive/'))
    elif "dead" in all_cells_names_list[i][0]:
        features_dead.append(all_images_with_features[i])
        features_dead_names.append(all_cells_names_list[i][0].removeprefix('data/cropped/dead/dead/'))
    elif "inhib" in all_cells_names_list[i][0]:
        features_inhib.append(all_images_with_features[i])
        features_inhib_names.append(all_cells_names_list[i][0].removeprefix('data/cropped/inhib/inhib/'))

kmeans_alive = KMeans(n_clusters=2, random_state=0).fit(features_alive)
# Alive_0=len(kmeans_alive.labels_)-sum(kmeans_alive.labels_)
# Alive_1=sum(kmeans_alive.labels_)
# print(Alive_0/len(kmeans_alive.labels_),Alive_1/len(kmeans_alive.labels_))

kmeans_dead = KMeans(n_clusters=2, random_state=0).fit(features_dead)
# Dead_0=len(kmeans_dead.labels_)-sum(kmeans_dead.labels_)
# Dead_1=sum(kmeans_dead.labels_)
# print(Dead_0/len(kmeans_dead.labels_),Dead_1/len(kmeans_dead.labels_))

kmeans_inhib = KMeans(n_clusters=2, random_state=0).fit(features_inhib)
# Inhib_0=len(kmeans_inhib.labels_)-sum(kmeans_inhib.labels_)
# Inhib_1=sum(kmeans_inhib.labels_)
# print(Inhib_0/len(kmeans_inhib.labels_),Inhib_1/len(kmeans_inhib.labels_))

n_iterations=4;

features_inhib_2=[];
features_inhib_names_2 = []

for i in range(len(kmeans_inhib.labels_)):
    if kmeans_inhib.labels_[i]==0:
        features_inhib_2.append(features_inhib[i])
        features_inhib_names_2.append(features_inhib_names[i])

kmeans2_inhib = KMeans(n_clusters=2, random_state=0).fit(features_inhib_2)

features_inhib_3=[];
features_inhib_names_3 = []

for i in range(len(kmeans2_inhib.labels_)):
    if kmeans2_inhib.labels_[i]==0:
        features_inhib_3.append(features_inhib_2[i])
        features_inhib_names_3.append(features_inhib_names_2[i])

kmeans3_inhib = KMeans(n_clusters=2, random_state=0).fit(features_inhib_3)

features_inhib_4=[];
features_inhib_names_4 = []

for i in range(len(kmeans3_inhib.labels_)):
    if kmeans3_inhib.labels_[i]==1:
        features_inhib_4.append(features_inhib_3[i])
        features_inhib_names_4.append(features_inhib_names_3[i])

kmeans4_inhib = KMeans(n_clusters=2, random_state=0).fit(features_inhib_4)

features_inhib_5=[];
features_inhib_names_5 = []

for i in range(len(kmeans4_inhib.labels_)):
    if kmeans4_inhib.labels_[i]==0:
        features_inhib_5.append(features_inhib_4[i])
        features_inhib_names_5.append(features_inhib_names_4[i])

kmeans5_inhib = KMeans(n_clusters=2, random_state=0).fit(features_inhib_5)

features_inhib_6=[];
features_inhib_names_6 = []

for i in range(len(kmeans5_inhib.labels_)):
    if kmeans5_inhib.labels_[i]==0:
        features_inhib_6.append(features_inhib_5[i])
        features_inhib_names_6.append(features_inhib_names_5[i])

kmeans6_inhib = KMeans(n_clusters=2, random_state=0).fit(features_inhib_6)

features_inhib_7=[];
features_inhib_names_7 = []

for i in range(len(kmeans6_inhib.labels_)):
    if kmeans6_inhib.labels_[i]==0:
        features_inhib_7.append(features_inhib_6[i])
        features_inhib_names_7.append(features_inhib_names_6[i])

kmeans7_inhib = KMeans(n_clusters=2, random_state=0).fit(features_inhib_7)


zip_iterator = zip(features_inhib_names_7, kmeans7_inhib.labels_)
zip_iterator = sorted(zip_iterator, key = lambda x: x[1])

dictionary_0={}
dictionary_1={}
dictionary_2={}
# list_0=

for k in range(len(zip_iterator)):
    if zip_iterator[k][1]==0:
        dictionary_0[zip_iterator[k][0]]=0
    elif zip_iterator[k][1]==1:
        dictionary_1[zip_iterator[k][0]]=1

    # elif zip_iterator[k][1]==2:
    #     dictionary_2[zip_iterator[k][0]] = 2
# dictionary_1=zip_iterator[sum(kmeans_inhib.labels_):]
#
# dictionary_0 = dict(dictionary_0)
# dictionary_1 = dict(dictionary_1)

###################### SECOND ITERATION #######################
kmeans_inhib = KMeans(n_clusters=2, random_state=0).fit(features_inhib)


# # For INHIB
# data_dir = Path("../data/cropped/inhib/inhib/")
# dead_images_raw = [(cv2.imread(str(img)), str(img)) for img in data_dir.iterdir()]
#
# del dead_images_raw[0]
# for image, cell_name in dead_images_raw:
#     cell_name='cell'+cell_name.split('cell')[1]
#     plt.imshow(image)
#     plt.title(cell_name+'-->'+str(a_dictionary[cell_name]))
#     plt.show()

# For INHIB
list_0=list(dictionary_0.keys())
list_1=list(dictionary_1.keys())
list_2=list(dictionary_2.keys())
list_0=list_0[:1000]
list_1=list_1[:1000]
list_2=list_2[:1000]
data_dir = Path("../data/cropped/inhib/inhib/")

for j in range(10):
    fig, axs = plt.subplots(2, 10)

    for i in range(10):
        axs[0, i].imshow(cv2.imread(str(data_dir)+"\\"+list_0[i+j*10]))
        axs[0, i].set_title('0')

    for i in range(10):
        axs[1, i].imshow(cv2.imread(str(data_dir)+"\\"+list_1[i+j*10]))
        axs[1, i].set_title('1')

    # for i in range(10):
    #     axs[2, i].imshow(cv2.imread(str(data_dir)+"\\"+list_2[i+j*10]))
    #     axs[2, i].set_title('2')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()