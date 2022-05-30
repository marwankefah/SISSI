import feature.Classifier as cl
import feature.helper as hf
import torch
from pathlib import Path
from torch import nn
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier


train_transforms = hf.defineTransforms()

# Path To Dataset
data_path = Path("data/cropped")
feat_path = Path("feature/output/gabor_index.csv")
modelpath = Path("model/checkpoint.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

trainloader, testloader = hf.getProcessedData(
    data_path, feat_path, 21000, 5000, transform=train_transforms)


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


batch_size = 128
max_epoch = 100
max_iter = int(30000/batch_size*max_epoch)
n_iter_no_change = 20

for train_images, train_labels in trainloader:
    train_images = train_images.numpy()
    train_labels = train_labels.numpy()
    # model = SVC(probability=True)

    model = MLPClassifier(solver='adam', alpha=1e-5,
                          hidden_layer_sizes=(256, 128, 64),
                          learning_rate_init=0.0003, random_state=1,
                          batch_size=128, verbose=True, max_iter=max_iter,
                          early_stopping=True, n_iter_no_change=n_iter_no_change)

    model.fit(train_images, train_labels)
    pred = model.predict(train_images)
    print("Training AUC: ", roc_auc_score_multiclass(
        train_labels, pred))
    print(classification_report(train_labels, pred))

for images, labels in testloader:
    images = images.numpy()
    labels = labels.numpy()
    test_pred = model.predict(images)
    print("Test AUC: ", roc_auc_score_multiclass(
        labels, test_pred))
    print(classification_report(labels, test_pred))
