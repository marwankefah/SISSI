import cv2
import numpy as np
import pandas as pd
from torch import nn
from torchvision import datasets, transforms
from pathlib import Path
from feature.Classifier import *
from sklearn.model_selection import train_test_split
import pickle as pkl


def loadData(path, transforms=[], batchsize=4096):
    data = datasets.ImageFolder(path, transform=transforms)
    trainloader = torch.utils.data.DataLoader(
        data, batch_size=batchsize, shuffle=True)
    return trainloader


def loadsplitData(path, transforms, train_batchsize=10000, test_batchsize=4000):
    data = datasets.ImageFolder(path, transform=transforms)
    data_train, data_test = torch.utils.data.random_split(data, (
        int(len(data.imgs) * 0.8), len(data.imgs) - int(len(data.imgs) * 0.8)))
    trainloader = torch.utils.data.DataLoader(
        data_train, batch_size=train_batchsize, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        data_test, batch_size=test_batchsize)

    return trainloader, testloader


def load_balanced_data(path, transforms, train_batchsize=10000, test_batchsize=4000):
    dead_path = path/Path("dead")
    alive_path = path/Path("alive")
    inhib_path = path/Path("inhib")

    dead_trainloader = loadData(
        str(dead_path), transforms, batchsize=5000)
    alive_trainloader = loadData(
        str(alive_path), transforms, batchsize=5000)
    inhib_trainloader = loadData(
        str(inhib_path), transforms, batchsize=5000)

    deaditer = iter(dead_trainloader)
    aliveiter = iter(alive_trainloader)
    inhibiter = iter(inhib_trainloader)

    images_dead, labels_dead = deaditer.next()
    images_alive, labels_alive = aliveiter.next()
    images_inhib, labels_inhib = inhibiter.next()
    X = torch.concat((images_dead[0:5000], images_alive[0:5000],
                      images_inhib[0:5000]))
    y = torch.concat((labels_dead[0:5000], labels_alive[0:5000]+1,
                      labels_inhib[0:5000]+2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def loadcifar(path, transforms=[], batchsize=4096):
    data = datasets.CIFAR10(path, train=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(
        data, batch_size=batchsize, shuffle=True)
    return trainloader


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


def defineTransforms():
    train_transforms = transforms.Compose([transforms.Resize(22),
                                           transforms.RandomRotation(2),
                                           transforms.CenterCrop(22),
                                           transforms.Grayscale(),
                                           transforms.ToTensor()])
    return train_transforms


def defineTransformstest():
    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize(22),
                                           transforms.CenterCrop(22),
                                           transforms.Grayscale(),
                                           transforms.ToTensor()])
    return train_transforms


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


def gaborvector(images, real, imag, device="cpu"):
    images = GaborFeatures(images, real, imag, device=device)
    images = images.reshape(images.shape[0], -1)
    return images


def processimage(img, path, transforms=defineTransforms(), device="cpu"):
    df = pd.read_csv(path)
    real, imag = build_filters()
    img = transforms(img)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = gaborvector(img, real, imag, device=device)
    data = img[:, df["x"].values]

    return data


def getProcessedData(path1, path2, train_batchsize=512, test_batchsize=512, transform=defineTransforms(), device="cpu"):
    df = pd.read_csv(path2)
    X_train, X_test, y_train, y_test = load_balanced_data(
        path1, transforms=transform, train_batchsize=train_batchsize, test_batchsize=test_batchsize)
    real, imag = build_filters()
    images_train = gaborvector(X_train, real, imag, device=device)
    train_data = images_train[:, df["x"].values]
    train_labels = y_train

    images_test = gaborvector(X_test, real, imag, device=device)
    test_data = images_test[:, df["x"].values]
    test_labels = y_test

    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels),
                                              batch_size=train_batchsize)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_labels),
                                             batch_size=test_batchsize, shuffle=True)
    return trainloader, testloader


def save_gabor_feats(path1, path2, transform, train_batchsize, device="cpu"):
    df = pd.read_csv(path2)
    datas = datasets.ImageFolder(path1, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        datas, batch_size=train_batchsize, shuffle=False)
    real, imag = build_filters()

    for images, labels in trainloader:
        # images = images.reshape(-1, images.shape[2], images.shape[3])
        images = gaborvector(images, real, imag, device=device)
        data = images[:, df["x"].values]
        break

    arraytosave = np.array([data, data.imgs])
    fileObject = open("data/output/gabor_features.pkl", 'wb')
    pkl.dump(arraytosave, fileObject)
    return


def getmodelfortesting(path, lr=0.001, dropout=0.2, device="cpu"):
    model, optimizer, criterion = load_checkpoint(
        path, lr=lr, dropout=dropout, device=device)
    return model


def getemotion(img, path1, path2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getmodelfortesting(path1, device=device)
    transforms = defineTransformstest()
    img = processimage(img=img, transforms=transforms,
                       path=path2, device=device)
    prob, topclass = predictforImage(model, img, device)
    prob *= 100
    emotion = None
    if topclass == 0:
        emotion = "Happy"
    elif topclass == 1:
        emotion = "Angry"
    elif topclass == 2:
        emotion = "Sad"
    elif topclass == 3:
        emotion = "Surprised"
    elif topclass == 4:
        emotion = "Neutral"
    elif topclass == 5:
        emotion = "Other"

    return emotion, prob
