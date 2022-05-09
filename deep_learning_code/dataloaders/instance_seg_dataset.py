import logging
import os
import numpy as np
import torch
from PIL import Image


class chrisi_dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms):
        self.root = root
        self.transforms = transforms
        self.split = split
        # load all image files, sorting them to
        # ensure that they are aligned

        all = os.listdir(os.path.join(root, split))
        self.imgs = list(
            sorted([string for string in all if string.endswith(".jpg")]))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.split, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        target = None
        if self.transforms is not None:
            img, target = self.transforms(img, target)

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

        # print(img_path, idx, img.size, masks.shape)

        # get bounding box coordinates for each mask
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

        labels = torch.ones((num_objs - len(invalid_ids),), dtype=torch.int64)
        mask_channel_first = np.delete(masks, invalid_ids, axis=0)
        # change channels last to channels first format
        mask_channel_first = [mask_channel_first[i].astype(np.float32) for i in range(len(mask_channel_first))]

        target = {}
        if self.transforms is not None:
            img_np = np.array(img)
            if not img_np.dtype == np.uint8:
                logging.info("Error: Image is not of type np.uint8?")
                raise
            img_np = img_np.astype(np.float32) / 255
            result = self.transforms(
                image=img_np, masks=mask_channel_first, bboxes=boxes,
                class_labels=np.array(labels))

        # check images/mask shapes before  masks [N, H, W], make mask channel first in tensor
        img = result['image']
        boxes = result['bboxes']
        masks = result['masks']

        labels = torch.ones((len(boxes),), dtype=torch.int64)

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
