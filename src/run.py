from dead_cells.dead_cells_bboxes import get_bboxes_dead
from feature.gabor_filters import gaborvector
from inhib_cells.inhib_cells_bboxes import get_bboxes_inhib
from alive_cells.alive_cells_bboxes import get_bboxes_alive
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
import matplotlib.pyplot as plt
from deep_learning_code.odach_our import nms, weighted_boxes_fusion
# from deep_learning_code.torchvision_our.ops import batched_nms
from torchvision.ops import batched_nms
# from deep_learning_code.reference.engine import visualize

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(22),
                                transforms.CenterCrop(22),
                                transforms.Grayscale(),
                                transforms.ToTensor()])


def get_bboxes_df(image):

    bboxes_dead = get_bboxes_dead(image)
    bboxes_alive = get_bboxes_alive(
        image)
    bboxes_inhib = get_bboxes_inhib(
        image)
    return {"bboxes_dead": bboxes_dead,
            "bboxes_alive": bboxes_alive,
            "bboxes_inhib": bboxes_inhib}


def get_bboxes_list(bboxes_df):
    bboxes_dead = bboxes_df["bboxes_dead"][[
        "x_min", "y_min", "x_max", "y_max"]]
    bboxes_alive = bboxes_df["bboxes_alive"][[
        "x_min", "y_min", "x_max", "y_max"]]
    bboxes_inhib = bboxes_df["bboxes_inhib"][[
        "x_min", "y_min", "x_max", "y_max"]]

    return {"bboxes_dead": bboxes_dead.values.tolist(),
            "bboxes_alive": bboxes_alive.values.tolist(),
            "bboxes_inhib": bboxes_inhib.values.tolist()}


def select_bboxes(pr, bbox, cell_index):
    amax = np.argmax(pr, axis=1)
    max_val = np.max(pr, axis=1)
    idx = np.where(amax == cell_index)
    return np.array(bbox)[idx], max_val[idx]


def visualize_bbox(img, bbox, class_name, color=(150, 0, 0), thickness=1):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(150, 150, 150),
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, scores, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id, score in zip(bboxes, category_ids, scores):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, '{} {:.2f}'.format(class_name, score))
    return img


def pipeline(image, gabor_filter, feature_df):
    bboxes_df = get_bboxes_df(image)

    cropped_dead = crop_w_bboxes(image=image, bboxes=bboxes_df["bboxes_dead"])
    cropped_alive = crop_w_bboxes(
        image=image, bboxes=bboxes_df["bboxes_alive"])
    cropped_inhib = crop_w_bboxes(
        image=image, bboxes=bboxes_df["bboxes_inhib"])
    cropped_dead_images = [cropped_dead[i][2]
                           for i in range(len(cropped_dead))]
    cropped_alive_images = [cropped_alive[i][2]
                            for i in range(len(cropped_alive))]
    cropped_inhib_images = [cropped_inhib[i][2]
                            for i in range(len(cropped_inhib))]

    cropped_dead_images = [transform(img) for img in cropped_dead_images]
    cropped_alive_images = [transform(img) for img in cropped_alive_images]
    cropped_inhib_images = [transform(img) for img in cropped_inhib_images]

    dead_feat = gaborvector(torch.stack(cropped_dead_images),
                            gabor_filter[0], gabor_filter[1])
    alive_feat = gaborvector(torch.stack(cropped_alive_images),
                             gabor_filter[0], gabor_filter[1])
    inhib_feat = gaborvector(torch.stack(cropped_inhib_images),
                             gabor_filter[0], gabor_filter[1])

    dead_feat = dead_feat[:, feature_df["x"].values]
    alive_feat = alive_feat[:, feature_df["x"].values]
    inhib_feat = inhib_feat[:, feature_df["x"].values]

    with open(model_path, 'rb') as f:
        model = pkl.load(f)

    pr_dead = model.predict_proba(dead_feat)
    pr_alive = model.predict_proba(alive_feat)
    pr_inhib = model.predict_proba(inhib_feat)

    bboxes = get_bboxes_list(bboxes_df)
    dead_bboxes, dead_pr = select_bboxes(pr_dead, bboxes["bboxes_dead"], 0)
    alive_bboxes, alive_pr = select_bboxes(
        pr_alive, bboxes["bboxes_alive"], 1)
    inhib_bboxes, inhib_pr = select_bboxes(
        pr_inhib, bboxes["bboxes_inhib"], 2)
    selected_bboxes = [dead_bboxes.astype(float), alive_bboxes.astype(float),
                       inhib_bboxes.astype(float)]
    selected_pr = [dead_pr,  alive_pr, inhib_pr]
    selected_label = [np.array([0] * len(dead_bboxes)), np.array([1] *
                      len(alive_bboxes)), np.array([2]*len(inhib_bboxes))]

    # keep = nms(
    #     torch.from_numpy(selected_bboxes).float(), torch.from_numpy(
    #         selected_pr), torch.from_numpy(selected_label).float(), 0.2)

    for idx in range(len(selected_pr)):
        if np.max(selected_bboxes[idx], initial=1) > 1:
            selected_bboxes[idx][:, 0] /= image.shape[1]
            selected_bboxes[idx][:, 2] /= image.shape[1]
            selected_bboxes[idx][:, 1] /= image.shape[0]
            selected_bboxes[idx][:, 3] /= image.shape[0]

    selected_bboxes, selected_pr, selected_label = weighted_boxes_fusion(selected_bboxes, selected_pr,
                                                                         selected_label, None, iou_thr=0, skip_box_thr=0.9)

    # selected_bboxes, selected_pr, selected_label = nms(selected_bboxes, selected_pr,
    #                                                    selected_label, 0)
    # arg_pr = np.where(selected_pr >= 0.99)
    selected_bboxes[:, 0] *= image.shape[1]
    selected_bboxes[:, 2] *= image.shape[1]
    selected_bboxes[:, 1] *= image.shape[0]
    selected_bboxes[:, 3] *= image.shape[0]

    out_img = visualize(image, selected_bboxes, selected_pr,
                        selected_label, category_id_to_name={
                            1: 'alive', 2: 'inhib', 0: 'dead'})

    plt.imshow(out_img)
    plt.show()

    # breakpoint()


if __name__ == "__main__":
    image_path = data_dir/Path("test_labelled/cell49.jpg")
    image = cv2.imread(str(image_path))
    real, imag = hf.build_filters()
    feature_path = Path("feature/output/gabor_index.csv")
    feature_df = pd.read_csv(feature_path)
    pipeline(image, (real, imag), feature_df)
