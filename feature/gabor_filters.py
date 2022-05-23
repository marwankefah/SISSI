# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import feature.Ada as Ada
from torch import nn
import pandas as pd
# %%


def build_gabor_filters():
    realfilters = []
    imagfilters = []
    sigmas = [4, 4 * np.sqrt(2), 8, 8 * np.sqrt(2), 16]
    for theta in np.arange(0, np.pi, np.pi / 8):
        for sigma in sigmas:
            g_kernel_real = cv2.getGaborKernel(ksize=(7, 7), sigma=sigma, theta=theta, lambd=sigma, gamma=1, psi=0,
                                               ktype=cv2.CV_32F)
            g_kernel_imag = cv2.getGaborKernel(ksize=(7, 7), sigma=sigma, theta=theta, lambd=sigma, gamma=1,
                                               psi=np.pi / 2, ktype=cv2.CV_32F)
            realfilters.append(g_kernel_real)
            imagfilters.append(g_kernel_imag)

    return realfilters, imagfilters


def loadData(path, transforms=[], batchsize=4096):
    data = datasets.ImageFolder(path, transform=transforms)
    trainloader = torch.utils.data.DataLoader(
        data, batch_size=batchsize, shuffle=True)
    return trainloader


def GaborFeatures(X, real, imag, device="cpu"):
    a = np.stack(real)
    a = torch.from_numpy(a.reshape(a.shape[0], 1, a.shape[1], a.shape[2]))
    b = np.stack(imag)
    b = torch.from_numpy(b.reshape(b.shape[0], 1, b.shape[1], b.shape[2]))
    conv1 = nn.Conv2d(1, out_channels=40, kernel_size=7, bias=False, padding=3)
    conv2 = nn.Conv2d(1, out_channels=40, kernel_size=7, bias=False, padding=3)

    conv1.weight.data = a
    conv2.weight.data = b
    conv1.weight.requires_grad = False
    conv2.weight.requires_grad = False

    conv1.to(device)
    conv2.to(device)
    conv1.weight.to(device)
    conv2.weight.to(device)
    X = X.to(device)
    with torch.no_grad():
        outreal = conv1(X)
        outimag = conv2(X)
        output = torch.sqrt(outreal ** 2 + outimag ** 2)
    return output


def build_filters():
    realfilters = []
    imagfilters = []
    sigmas = [4, 4 * np.sqrt(2), 8, 8 * np.sqrt(2), 16]
    for theta in np.arange(0, np.pi, np.pi / 8):
        for sigma in sigmas:
            g_kernel_real = cv2.getGaborKernel(ksize=(7, 7), sigma=sigma, theta=theta, lambd=sigma, gamma=1, psi=0,
                                               ktype=cv2.CV_32F)
            g_kernel_imag = cv2.getGaborKernel(ksize=(7, 7), sigma=sigma, theta=theta, lambd=sigma, gamma=1,
                                               psi=np.pi / 2, ktype=cv2.CV_32F)
            realfilters.append(g_kernel_real)
            imagfilters.append(g_kernel_imag)

    return realfilters, imagfilters


def gaborvector(images, real, imag, device="cpu"):
    images = GaborFeatures(images, real, imag, device=device)
    images = images.reshape(images.shape[0], -1)
    return images


train_transforms = transforms.Compose([transforms.Resize(22),
                                       transforms.CenterCrop(22),
                                       transforms.Grayscale(),
                                       transforms.ToTensor()])


dead_path = "/Users/manasikattel/cell-segmentation/raw/named_images_type/dead"
alive_path = "/Users/manasikattel/cell-segmentation/raw/named_images_type/alive"


dead_trainloader = loadData(dead_path, train_transforms, batchsize=70)
alive_trainloader = loadData(alive_path, train_transforms, batchsize=94)

deaditer = iter(dead_trainloader)
images, labels = deaditer.next()

aliveiter = iter(alive_trainloader)
images2, labels2 = aliveiter.next()


X1 = torch.concat((images[0:46], images2[0:62])).reshape(-1, 22, 22)
Y1 = np.ones((108, 1))
Y1[46::] = 0
X1 = X1.unsqueeze(1)

X2 = torch.concat((images[46::], images2[62::])).reshape(-1, 22, 22)
Y2 = np.ones((56, 1))
Y2[24::] = 0
X2 = X2.unsqueeze(1)


real, imag = build_filters()


X_feat1 = gaborvector(X1, real, imag)
X_feat2 = gaborvector(X2, real, imag)

ada = Ada.AdaBoostSelection(150)
ada.fit(X_feat1.numpy(), Y1)

acctrain = ada.validate(X_feat1.numpy(), Y1)
accvalidate = ada.validate(X_feat2.numpy(), Y2)

print("Training accuracy: ", acctrain)
print("Validation accuracy:     ", accvalidate)

breakpoint()
a = []
for classifier in ada.Classifiers:
    a.append(classifier.feature_index)

col = {"x": a}
df = pd.DataFrame(col)
df.insert(0, "x", a, True)
df.to_csv("feature/output/gabor_index.csv")
