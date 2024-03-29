import logging
import os
import random

import numpy as np
import torch
from PIL import Image
import cv2
import dataloaders.utils as utils


class cell_lab_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms, cache_labels=False, shuffle=False, patch_size=[512, 512],
                 need_seam_less_clone=False,
                 seam_less_clone_k_size=(71, 71),
                 seam_less_blur_sigma=30,
                 blob_detection=True,
                 dictionary_mapping={'alive': 1, 'inhib': 2, 'dead': 3}, is_test=False):
        self.root = root
        self.transforms = transforms
        self.split = split
        self.blob_detection = blob_detection
        self.patch_size = patch_size
        self.cache_labels = cache_labels
        self.need_seam_less_clone = need_seam_less_clone
        self.seam_less_clone_k_size = seam_less_clone_k_size
        self.seam_less_blur_sigma = seam_less_blur_sigma
        self.dictionary_mapping = dictionary_mapping
        self.is_test = is_test
        self.bboxes_path_or_cache = []
        self.images_path_or_cache = []
        self.img_labels_path_or_cache = []
        self.images_path = []
        for cell_type in self.split:
            bboxes_dir_path = os.path.join(self.root, 'weak_labels_reduced_nms')
            image_dir_path = os.path.join(root, cell_type)
            images_dir = os.listdir(image_dir_path)
            img_list = list(
                sorted([os.path.join(cell_type, string) for string in images_dir if string.endswith(".jpg")]))
            bboxes_path_or_cache = []
            images_path_or_cache = []
            img_labels_path_or_cache = []
            for cell_name in img_list:
                bboxes_path = os.path.join(bboxes_dir_path, cell_name.split('.')[-2] + '.txt')
                img_path = os.path.join(self.root, cell_name)

                if cache_labels:
                    annotations = np.loadtxt(bboxes_path,
                                             dtype={'names': ('cell_name', 'x_min', 'y_min', 'x_max', 'y_max'),
                                                    'formats': ('U25', 'i4', 'i4', 'i4', 'i4')}, delimiter=' ')

                    boxes = np.dstack(
                        (annotations['x_min'], annotations['y_min'], annotations['x_max'], annotations['y_max']))
                    boxes = boxes[0].tolist()
                    bboxes_path_or_cache.append(boxes)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    images_path_or_cache.append(img)
                    labels_mapped = list(map(dictionary_mapping.get, annotations['cell_name'].reshape(-1).tolist()))
                    img_labels_path_or_cache.append(labels_mapped)
                else:
                    images_path_or_cache.append(img_path)
                    bboxes_path_or_cache.append(bboxes_path)
                    img_labels_path_or_cache.append(bboxes_path)

            self.images_path_or_cache += images_path_or_cache
            self.bboxes_path_or_cache += bboxes_path_or_cache
            self.img_labels_path_or_cache += img_labels_path_or_cache
            self.images_path += img_list

        self.sample_list = list(zip(self.images_path_or_cache, self.bboxes_path_or_cache,self.images_path))

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.sample_list[idx][2]
        if self.cache_labels:
            # cell_name =
            boxes = self.sample_list[idx][1]
            img = self.sample_list[idx][0]
            # Todo add to the sample list
            labels = self.img_labels_path_or_cache[idx]
        else:
            bboxes_path = self.sample_list[idx][1]
            annotations = np.loadtxt(bboxes_path,
                                     dtype={'names': ('cell_name', 'x_min', 'y_min', 'x_max', 'y_max'),
                                            'formats': ('U25', 'i4', 'i4', 'i4', 'i4')}, delimiter=' ')

            boxes = np.dstack((annotations['x_min'], annotations['y_min'], annotations['x_max'], annotations['y_max']))
            boxes = boxes[0].tolist()
            img = cv2.imread(self.sample_list[idx][0], cv2.IMREAD_COLOR)
            labels = list(map(self.dictionary_mapping.get, annotations['cell_name'].tolist()))

        boxes_post_process = []
        labels_post_process = []
        img_mask = np.zeros_like(img)
        for box, label in zip(boxes, labels):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            if self.is_test or (xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0 and xmax < img.shape[
                1] and ymax < img.shape[0]):
                boxes_post_process.append(box)
                labels_post_process.append(label)
                img_mask[ymin:ymax, xmin:xmax, :] = 1
            else:
                # print(xmin, xmax, ymin, ymax, img.shape)
                pass

        if self.need_seam_less_clone:
            img = utils.seam_less_clone(img, img_mask, ksize=self.seam_less_clone_k_size,
                                        sigma=self.seam_less_blur_sigma)

        boxes = boxes_post_process
        labels = labels_post_process
        if self.blob_detection:
            labels = [1] * len(boxes)

        target = {}
        target['image_size'] = torch.as_tensor([img.shape[0], img.shape[1]], dtype=torch.int64)
        if self.transforms is not None:
            img_np = np.array(img)
            if not img_np.dtype == np.uint8:
                logging.info("Error: Image is not of type np.uint8?")
                raise
            img_np = img_np.astype(np.float32) / 255
            result = self.transforms(
                image=img_np, bboxes=boxes, category_id=labels)

        boxes = result['bboxes']
        num_objs = len(boxes)

        img = result['image']

        # convert everything into a torch.Tensor
        if len(boxes) != 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(result['category_id'], dtype=torch.int64)
        else:
            print('image {} with no boxes'.format(img_path))
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0), dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        masks = torch.empty((0,), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target, 0

    def __len__(self):
        return len(self.sample_list)


class cell_pose_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms):
        self.root = root
        self.transforms = transforms
        self.split = split
        # load all image files, sorting them to
        # ensure that they are aligned
        all = os.listdir(os.path.join(root, split))
        self.imgs = list(
            sorted([string for string in all if string.endswith("img.png")]))
        self.masks = list(
            sorted([string for string in all if string.endswith("masks.png")]))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.split, self.imgs[idx])
        mask_path = os.path.join(self.root, self.split, self.masks[idx])
        cell_name = self.imgs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = np.stack((img,) * 3, axis=-1)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # convert the PIL Image into a numpy array
        mask = np.array(mask, np.int16)
        # mask = np.array(mask)
        target = {}
        target['image_size'] = torch.as_tensor([img.shape[0], img.shape[1]], dtype=torch.int64)

        if self.transforms is not None:
            img_np = np.array(img)
            if not img_np.dtype == np.uint8:
                logging.info("Error: Image is not of type np.uint8?")
                raise
            img_np = img_np.astype(np.float32) / 255
            result = self.transforms(
                image=img_np, mask=mask)

        # check images/mask shapes before  masks [N, H, W], make mask channel first in tensor
        img = result['image']
        mask = np.asarray(result['mask'])
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        invalid_ids = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # checking degenerated boxes or ugly boxes
            if xmin < xmax and ymin < ymax:
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                invalid_ids.append(i)

        masks = np.delete(masks, invalid_ids, axis=0)

        labels = torch.ones((num_objs - len(invalid_ids),), dtype=torch.int64)

        # convert everything into a torch.Tensor
        if len(boxes) != 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            print('image {} with no boxes'.format(img_path))
            boxes = torch.empty((0, 4), dtype=torch.float32)
        # there is only one class
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if torch.numel(boxes) != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = boxes

        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target, cell_name

    def __len__(self):
        return len(self.imgs)
