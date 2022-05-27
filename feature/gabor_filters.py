# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import feature.Ada as Ada
from torch import nn
import pandas as pd
from pathlib import Path
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


def loadcifar(path, transforms=[], batchsize=4096):
    data = datasets.CIFAR10(
        path, train=True, transform=transforms, download=True)
    trainloader = torch.utils.data.DataLoader(
        data, batch_size=batchsize, shuffle=True)
    return trainloader


train_transforms = transforms.Compose([transforms.Resize(22),
                                       transforms.CenterCrop(22),
                                       transforms.Grayscale(),
                                       transforms.ToTensor()])

cropped_data_path = Path("data/cropped")
dead_path = cropped_data_path/Path("dead")
alive_path = cropped_data_path/Path("alive")
inhib_path = cropped_data_path/Path("inhib")
cifar_path = Path("data/cifar")


dead_trainloader = loadData(str(dead_path), train_transforms, batchsize=5000)
alive_trainloader = loadData(str(alive_path), train_transforms, batchsize=5000)
inhib_trainloader = loadData(str(inhib_path), train_transforms, batchsize=5000)
cifarloader = loadcifar(str(cifar_path), train_transforms, batchsize=15000)


deaditer = iter(dead_trainloader)
aliveiter = iter(alive_trainloader)
inhibiter = iter(inhib_trainloader)
cifariter = iter(cifarloader)

images_dead, labels_dead = deaditer.next()
images_alive, labels_alive = aliveiter.next()
images_inhib, labels_inhib = inhibiter.next()
images_cifar, labels_cifar = cifariter.next()

X1 = torch.concat((images_dead[0:4000], images_alive[0:4000],
                  images_inhib[0:4000], images_cifar[0:12000])).reshape(-1, 22, 22)
Y1 = np.ones((24000, 1))
Y1[12000::] = 0
X1 = X1.unsqueeze(1)

X2 = torch.concat((images_dead[4000::], images_alive[4000::],
                  images_inhib[4000::], images_cifar[12000::])).reshape(-1, 22, 22)
Y2 = np.ones((6000, 1))
Y2[3000::] = 0
X2 = X2.unsqueeze(1)


real, imag = build_filters()

X_feat1 = gaborvector(X1, real, imag)
X_feat2 = gaborvector(X2, real, imag)

ada = Ada.AdaBoostSelection(200)

ada.fit(X_feat1.numpy(), Y1)

acctrain = ada.validate(X_feat1.numpy(), Y1)
accvalidate = ada.validate(X_feat2.numpy(), Y2)

print("Training accuracy: ", acctrain)
print("Validation accuracy:     ", accvalidate)

a = []
for classifier in ada.Classifiers:
    a.append(classifier.feature_index)

col = {"x": a}
df = pd.DataFrame(col)
df.insert(0, "x", a, True)
df.to_csv("feature/output/gabor_index.csv")
