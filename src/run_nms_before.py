from dead_cells.dead_cells_bboxes import get_bboxes_dead
from feature.gabor_filters import gaborvector
from inhib_cells.inhib_cells_bboxes import get_bboxes_inhib
from alive_cells.alive_cells_bboxes import get_bboxes_alive
from feature.extract_features import crop_w_bboxes
from settings import model_path, data_dir, feature_path, deep_learning_out_dir, image_dir
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
import warnings
from sklearn.metrics import roc_auc_score, classification_report

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


def cytotox(dead_count, alive_count, inhib_count):

    total_count = dead_count + inhib_count + alive_count

    # if less than 5 percent of total cells are dead
    if(total_count/100*5 > dead_count):
        # all alive
        return "none"
    if(total_count/100*20 > dead_count):
        # less than 20% round
        return "slight"
    elif(total_count/100*50 > dead_count):
        # less than 50% round
        return "mild"
    elif(total_count/100*70 > dead_count):
        # less than 70% round
        return "moderate"
    else:
        # more
        return "severe"


def pipeline(image, gabor_filter, feature_df):
    bboxes_df = get_bboxes_df(image)

    bboxes_all = bboxes_df["bboxes_dead"].values.tolist() + bboxes_df["bboxes_alive"].values.tolist() + bboxes_df[
        "bboxes_inhib"].values.tolist()

    bboxes_all_df = pd.DataFrame(
        bboxes_all, columns=['cell_type', 'x_min', 'y_min', 'x_max', 'y_max'])

    def nms(bounding_boxes, confidence_score, threshold):
        # If no bounding boxes, return empty list
        if len(bounding_boxes) == 0:
            return [], []

        # Bounding boxes
        boxes = np.array(bounding_boxes)

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = np.array(confidence_score)

        # Picked bounding boxes
        picked_boxes = []
        picked_score = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / \
                (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < threshold)
            order = order[left]

        return picked_boxes, picked_score

    bboxes_post_nms = \
        nms(bboxes_all_df[['x_min', 'y_min', 'x_max', 'y_max']
                          ].to_numpy(), np.ones((len(bboxes_all_df))), 0.1)[0]
    boxes_final = pd.DataFrame(bboxes_post_nms, columns=[
        "x_min", "y_min", "x_max", "y_max"])
    boxes_final['cell_type'] = 'anyth'

    cropped_all = crop_w_bboxes(image=image, bboxes=boxes_final)

    cropped_all_images = [cropped_all[i][2]
                          for i in range(len(cropped_all))]

    cropped_all_images = [transform(img) for img in cropped_all_images]

    all_feat = gaborvector(torch.stack(cropped_all_images),
                           gabor_filter[0], gabor_filter[1])

    all_feat = all_feat[:, feature_df["x"].values]

    with open(model_path, 'rb') as f:
        model = pkl.load(f)

    pr_all = model.predict_proba(all_feat)

    selected_bboxes = [np.array(bboxes_post_nms).astype(float)]
    selected_pr = [np.max(pr_all, axis=1)]
    selected_label = [np.argmax(pr_all, axis=1)]
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
                                                                         selected_label, None, iou_thr=0.5,
                                                                         skip_box_thr=0.25)

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
    # plt.title("Image processing output")
    # plt.imshow(out_img)
    # plt.show()

    boxes_final["pred"] = selected_label
    return {"dead": sum(boxes_final["pred"] == 0),
            "alive": sum(boxes_final["pred"] == 1),
            "inhib": sum(boxes_final["pred"] == 2)}


def pipeline_dl(image, gabor_filter, bbox_path):
    bboxes_df = pd.read_csv(bbox_path, delimiter=' ', names=[
                            "cell_type", "x_min", "y_min", "x_max", "y_max"])
    cropped = crop_w_bboxes(image=image, bboxes=bboxes_df)

    cropped_images = [cropped[i][2]
                      for i in range(len(cropped))]

    cropped_images = [transform(img) for img in cropped_images]

    feat = gaborvector(torch.stack(cropped_images),
                       gabor_filter[0], gabor_filter[1])

    feat = feat[:, feature_df["x"].values]

    with open(model_path, 'rb') as f:
        model = pkl.load(f)

    pred_labels = model.predict(feat)
    pred_pr = model.predict_proba(feat)

    bboxes_numpy = bboxes_df[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
    bboxes_df["pred"] = pred_labels

    out_img = visualize(image, bboxes_numpy, np.max(pred_pr, axis=1).tolist(),
                        pred_labels.tolist(), category_id_to_name={
        1: 'alive', 2: 'inhib', 0: 'dead'})
    plt.title("Deep learning output")
    plt.imshow(out_img)
    plt.show()
    return {"dead": sum(bboxes_df["pred"] == 0),
            "alive": sum(bboxes_df["pred"] == 1),
            "inhib": sum(bboxes_df["pred"] == 2)}


def get_gt_count(boxes_gt):
    return {"dead": sum(boxes_gt["cell_type"] == "dead"),
            "alive": sum(boxes_gt["cell_type"] == "alive"),
            "inhib": sum(boxes_gt["cell_type"] == "inhib")}


if __name__ == "__main__":
    real, imag = hf.build_filters()
    feature_df = pd.read_csv(feature_path)

    gt_cytotoxicity_1 = []
    gt_cytotoxicity_2 = []
    gt_cytotoxicity_3 = []
    gt_cytotoxicity_union = []

    ip_pred_cytotoxicity = []
    dl_pred_cytotoxicity = []

    for img in image_dir.rglob("*"):
        if ".jpg" in str(img):
            image_path = img
            bbox_path = deep_learning_out_dir / Path(img.stem + '.txt')
            image = cv2.imread(str(image_path))
            bbox_gt_3 = pd.read_csv(Path("data/chrisi/test_labelled_3") /
                                    Path(f'{img.stem}.txt'), delimiter=' ', names=[
                "cell_type", "x_min", "y_min", "x_max", "y_max"])

            bbox_gt_1 = pd.read_csv(Path("data/chrisi/test_labelled_1/test_labelled_1") /
                                    Path(f'{img.stem}.txt'), delimiter=' ', names=[
                "cell_type", "x_min", "y_min", "x_max", "y_max"])
            bbox_gt_2 = pd.read_csv(Path("data/chrisi/test_labelled_1/test_labelled_2") /
                                    Path(f'{img.stem}.txt'), delimiter=' ', names=[
                "cell_type", "x_min", "y_min", "x_max", "y_max"])
            bbox_gt_union = pd.read_csv(Path("data/chrisi/test_labelled_1/test_labelled_union") /
                                        Path(f'{img.stem}.txt'), delimiter=' ', names=[
                "cell_type", "x_min", "y_min", "x_max", "y_max"])

            warnings.filterwarnings("ignore")
            # print(bbox_gt)

            ip_counts = pipeline(image, (real, imag), feature_df)
            dl_counts = pipeline_dl(image, (real, imag), bbox_path=bbox_path)
            gt_counts_1 = get_gt_count(bbox_gt_1)
            gt_counts_2 = get_gt_count(bbox_gt_2)
            gt_counts_3 = get_gt_count(bbox_gt_3)
            gt_counts_union = get_gt_count(bbox_gt_union)

            print(img.stem)

            gt_cytotoxicity_1.append(cytotox(gt_counts_1["dead"],
                                             gt_counts_1["alive"], gt_counts_1["inhib"]))

            gt_cytotoxicity_2.append(cytotox(gt_counts_2["dead"],
                                             gt_counts_2["alive"], gt_counts_2["inhib"]))

            gt_cytotoxicity_3.append(cytotox(gt_counts_3["dead"],
                                             gt_counts_3["alive"], gt_counts_3["inhib"]))

            gt_cytotoxicity_union.append(cytotox(gt_counts_union["dead"],
                                                 gt_counts_union["alive"], gt_counts_union["inhib"]))

            ip_pred_cytotoxicity.append(cytotox(ip_counts["dead"],
                                                ip_counts["alive"], ip_counts["inhib"]))
            dl_pred_cytotoxicity.append(cytotox(dl_counts["dead"],
                                                dl_counts["alive"], dl_counts["inhib"]))


print(gt_cytotoxicity_1)
print(gt_cytotoxicity_2)
print(gt_cytotoxicity_3)
print(gt_cytotoxicity_union)

print(ip_pred_cytotoxicity)
print(dl_pred_cytotoxicity)


print("READER1")
ip_test_metrics = pd.DataFrame(classification_report(
    gt_cytotoxicity_1, ip_pred_cytotoxicity, output_dict=True)).reset_index()
dl_test_metrics = pd.DataFrame(classification_report(
    gt_cytotoxicity_1, dl_pred_cytotoxicity, output_dict=True)).reset_index()
print(ip_test_metrics)
print(dl_test_metrics)


print("READER2")
ip_test_metrics = pd.DataFrame(classification_report(
    gt_cytotoxicity_2, ip_pred_cytotoxicity, output_dict=True)).reset_index()
dl_test_metrics = pd.DataFrame(classification_report(
    gt_cytotoxicity_2, dl_pred_cytotoxicity, output_dict=True)).reset_index()
print(ip_test_metrics)
print(dl_test_metrics)


print("READER3")
ip_test_metrics = pd.DataFrame(classification_report(
    gt_cytotoxicity_3, ip_pred_cytotoxicity, output_dict=True)).reset_index()
dl_test_metrics = pd.DataFrame(classification_report(
    gt_cytotoxicity_3, dl_pred_cytotoxicity, output_dict=True)).reset_index()
print(ip_test_metrics)
print(dl_test_metrics)


print("READERUNION")
ip_test_metrics = pd.DataFrame(classification_report(
    gt_cytotoxicity_union, ip_pred_cytotoxicity, output_dict=True)).reset_index()
dl_test_metrics = pd.DataFrame(classification_report(
    gt_cytotoxicity_union, dl_pred_cytotoxicity, output_dict=True)).reset_index()
print(ip_test_metrics)
print(dl_test_metrics)
