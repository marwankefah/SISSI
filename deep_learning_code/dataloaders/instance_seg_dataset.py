import logging
import os
import numpy as np
import torch
from PIL import Image
import cv2


class chrisi_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms):
        self.root = root
        self.transforms = transforms
        self.split = split
        # load all image files, sorting them to
        # ensure that they are aligned

        images_dir = os.listdir(os.path.join(root, split))
        # TODO to be abstracted
        self.bboxes_dir_path = os.path.join(self.root, 'weak_labels', self.split)

        self.imgs = list(
            sorted([string for string in images_dir if string.endswith(".jpg")]))
        # self.bboxes = list(
        #     sorted([string for string in self.bboxes_dir_path if string.endswith(".txt")]))
        #

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.split, self.imgs[idx])

        bboxes_path = os.path.join(self.bboxes_dir_path, self.imgs[idx].split('.')[0] + '.txt')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        annotations = np.loadtxt(bboxes_path,
                                 dtype={'names': ('cell_name', 'x_min', 'y_min', 'x_max', 'y_max'),
                                        'formats': ('U25', 'i4', 'i4', 'i4', 'i4')}, delimiter=' ')

        boxes = np.dstack((annotations['x_min'], annotations['y_min'], annotations['x_max'], annotations['y_max']))
        boxes = boxes[0].tolist()
        labels = [1] * len(boxes)
        target = {}
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

        # there is only one class
        labels = torch.as_tensor(result['category_id'], dtype=torch.int64)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        masks = torch.empty((0,),dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.imgs)


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
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
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

        return img, target

    def __len__(self):
        return len(self.imgs)


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
