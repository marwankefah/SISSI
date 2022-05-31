from dead_cells.dead_cells_bboxes import get_bboxes_dead
from feature.gabor_filters import gaborvector
from inhib_cells.inhib_cells_bboxes import get_bboxes_inhib
# from alive_cells.alive_cells_bboxes import get_bboxes_alive
from feature.extract_features import crop_w_bboxes
from settings import model_path, data_dir
import pickle as pkl
from pathlib import Path
import cv2
import numpy as np
import feature.helper as hf
import torch
from torchvision import transforms
import pandas as pd

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(22),
                                transforms.CenterCrop(22),
                                transforms.Grayscale(),
                                transforms.ToTensor()])


def get_bboxes(image):

    return {"bboxes_dead": get_bboxes_dead(image),
            # "bboxes_alive": get_bboxes_alive(image),
            "bboxes_inhib": get_bboxes_inhib(image)}


def pipeline(image, gabor_filer, feature_df):
    bboxes = get_bboxes(image)
    cropped_dead = crop_w_bboxes(image=image, bboxes=bboxes["bboxes_dead"])
    cropped_dead_images = [cropped_dead[i][2]
                           for i in range(len(cropped_dead))]
    # cropped_alive = crop_w_bboxes(image=image, bboxes=bboxes["bboxes_alive"])
    cropped_inhib = crop_w_bboxes(image=image, bboxes=bboxes["bboxes_inhib"])
    cropped_inhib_images = [cropped_inhib[i][2]
                            for i in range(len(cropped_inhib))]
    # cropped_dead_images = [torch.from_numpy(
    #     img).float()for img in cropped_dead_images]

    # cropped_inhib_images = [torch.from_numpy(
    #     img).float() for img in cropped_inhib_images]
    cropped_dead_images = [transform(img) for img in cropped_dead_images]
    cropped_inhib_images = [transform(img) for img in cropped_inhib_images]

    dead_feat = gaborvector(torch.stack(cropped_dead_images),
                            gabor_filer[0], gabor_filer[1])
    # alive_feat = gaborvector(cropped_alive)
    inhib_feat = gaborvector(torch.stack(cropped_inhib_images),
                             gabor_filer[0], gabor_filer[1])

    dead_feat = dead_feat[:, feature_df["x"].values]
    inhib_feat = inhib_feat[:, feature_df["x"].values]

    with open(model_path, 'rb') as f:
        model = pkl.load(f)

    pr_dead = model.predict_proba(np.array(dead_feat))
    # pr_alive = model.predict_proba(cropped_alive)
    pr_inhib = model.predict_proba(inhib_feat)

    print(np.around(pr_dead, 2))
    print(np.around(pr_inhib, 2))


if __name__ == "__main__":
    image_path = data_dir/Path("dead/cell9.jpg")
    image = cv2.imread(str(image_path))
    real, imag = hf.build_filters()
    feature_path = Path("feature/output/gabor_index.csv")
    feature_df = pd.read_csv(feature_path)
    pipeline(image, (real, imag), feature_df)
