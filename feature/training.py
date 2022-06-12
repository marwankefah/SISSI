from sklearn import svm
from sklearn.decomposition._pca import PCA
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.neighbors._classification import KNeighborsClassifier

import feature.helper as hf
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
import pickle as pkl
from settings import model_path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

train_transforms = hf.defineTransforms()
test_transforms = hf.defineTransformstest()
# Path To Dataset
data_path = Path("data/cropped")
test_path = Path("data/cropped_test/1/")
feat_path = Path("gabor_index_1000.csv")
root_output = Path("data/ML_Results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 256
max_epoch = 100
max_iter = int(30000 / batch_size * max_epoch)
n_iter_no_change = 30

trainloader, testloader = hf.getProcessedData(
    data_path, feat_path, 15000, 15000, transform=train_transforms)

gold_standard_test_loader = hf.getProcessedDataTest(
    test_path, feat_path, 1000, transform=test_transforms)

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA"
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(),

    RandomForestClassifier(max_depth=2, n_estimators=128),

    MLPClassifier(solver='adam', alpha=0.0001,
                  hidden_layer_sizes=(128, 64),
                  learning_rate_init=0.0001, random_state=1,
                  batch_size=128, verbose=True, max_iter=max_iter,
                  early_stopping=True, n_iter_no_change=n_iter_no_change),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),

]


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(
            new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


for train_images, train_labels in trainloader:
    train_images = train_images.numpy()
    train_labels = train_labels.numpy()
    # pca.fit(train_images)
    # train_images = pca.transform(train_images)

for images, labels in testloader:
    val_images = images.numpy()
    val_labels = labels.numpy()

for images, labels in gold_standard_test_loader:
    test_images = images.numpy()
    test_labels = labels.numpy()

for name, clf in zip(names, classifiers):
    print(name)
    clf.fit(train_images, train_labels)
    pred = clf.predict(train_images)
    train_auc = roc_auc_score_multiclass(
        train_labels, pred)
    train_metrics = pd.DataFrame(classification_report(
        train_labels, pred, output_dict=True)).reset_index()
    train_metrics.loc[train_metrics.shape[0]] = [
        "ROC", train_auc[0], train_auc[1], train_auc[2], "", "", ""]

    train_metrics.to_csv("data/ML_Results/" + name + "_train_metrics.csv")

    val_pred = clf.predict(val_images)
    val_auc = roc_auc_score_multiclass(
        val_labels, val_pred)
    val_metrics = pd.DataFrame(classification_report(
        val_labels, val_pred, output_dict=True)).reset_index()
    val_metrics.loc[val_metrics.shape[0]] = [
        "ROC", val_auc[0], val_auc[1], val_auc[2], "", "", ""]
    print(val_metrics)
    val_metrics.to_csv("data/ML_Results/" + name + "_val_metrics.csv")

    test_pred = clf.predict(test_images)

    test_auc = roc_auc_score_multiclass(test_labels, test_pred)
    test_metrics = pd.DataFrame(classification_report(
        test_labels, test_pred, output_dict=True)).reset_index()
    test_metrics.loc[test_metrics.shape[0]] = [
        "ROC", test_auc[0], test_auc[1], test_auc[2], "", "", ""]
    print(test_metrics)
    test_metrics.to_csv("data/ML_Results/" + name + "_test_metrics.csv")

    modelpath = root_output / Path("model_" + name + "_.pkl")

    fileObject = open(modelpath, 'wb')
    pkl.dump(clf, fileObject)
