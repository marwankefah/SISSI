# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:47:42 2021

@author: Prinzessin
"""
import configparser
import os
import torch.optim as optim
# from monai.transforms.utility.dictionary import Lambdad


import reference.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision_our.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision_our.models.detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn

# from monai.transforms import (
#     Activations,
#     AddChanneld,
#     AsDiscrete,
#     Compose,
#     LoadImaged,
#     RandFlipd,
#     RandRotated,
#     RandZoomd,
#     ScaleIntensityd,
#     EnsureTyped,
#     Resized,
#     RandGaussianNoised,
#     RandGaussianSmoothd,
#     Rand2DElasticd,
#     RandAffined,
#     OneOf,
#     NormalizeIntensity,
#     AsChannelFirstd,
#     EnsureType,
#     LabelToMaskd
# )
# from monai.data.image_reader import PILReader, NibabelReader
# from monai.metrics import DiceMetric


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import numpy as np


class Configs:
    def __init__(self, filename):

        # =============================================================================
        # Readable ini file
        # =============================================================================
        self.config_filename = filename
        config_file = configparser.ConfigParser(allow_no_value=True)
        config_file.read(self.config_filename)

        self.live_cells_root_path = config_file.get(
            'path', 'live_cells_root_path', fallback='../data/FETA/')
        self.live_cells_img_path = config_file.get(
            'path', 'live_cells_img_path', fallback='../data/FETA/')

        self.cell_pose_root_path = config_file.get(
            'path', 'cell_pose_root_path', fallback='../data/FETA/')
        self.cell_pose_img_root_path = config_file.get(
            'path', 'cell_pose_img_root_path', fallback='../data/FETA/')

        self.chrisi_cells_root_path = config_file.get(
            'path', 'chrisi_cells_root_path', fallback='../data/FETA/')
        self.chrisi_cells_img_path = config_file.get(
            'path', 'chrisi_cells_img_path', fallback='../data/FETA/')

        self.model_output_path = config_file.get(
            'path', 'model_output_path', fallback='..')

        self.linux_gpu_id = config_file.get('path', 'linux_gpu_id', fallback=0)
        self.linux = config_file.getboolean('path', 'linux', fallback=False)
        self.model_path = config_file.get('path', 'model_path', fallback='')
        self.fold = config_file.getint('path', 'fold', fallback=0)
        self.load_model = config_file.getint('path', 'load_model', fallback=1)

        self.exp = config_file.get('path', 'exp', fallback='FETA/Mean_Teacher')
        self.model_name = config_file.get(
            'path', 'model_name', fallback='unetResnet34')

        self.multi_class = config_file.getboolean(
            'network', 'multi_class', fallback=True)

        self.train_mask = config_file.getboolean(
            'network', 'train_mask', fallback=False)
        self.box_detections_per_img = config_file.getint(
            'network', 'box_detections_per_img', fallback=250)
        self.min_size = config_file.getint(
            'network', 'min_size', fallback=400)
        self.max_size = config_file.getint(
            'network', 'max_size', fallback=800)
        self.need_seam_less_clone = config_file.getboolean(
            'network', 'need_seam_less_clone', fallback=False)
        self.optim = config_file.get('network', 'optim', fallback='adam')
        self.box_score_thresh = config_file.getfloat('network', 'box_score_thresh', fallback=0.05)
        self.box_nms_thresh = config_file.getfloat('network', 'box_nms_thresh', fallback=0.3)
        self.label_corr_score_thresh = config_file.getfloat('network', 'label_corr_score_thresh', fallback=0.1)

        self.lr_step_size = config_file.getfloat(
            'network', 'lr_step_size', fallback=8)
        self.lr_gamma = config_file.getfloat(
            'network', 'lr_gamma', fallback=0.1)
        self.label_correction = config_file.getboolean(
            'network', 'label_correction', fallback=False)

        self.overload_rcnn_predictor = config_file.getboolean(
            'network', 'overload_rcnn_predictor', fallback=False)

        self.label_correction_threshold = config_file.getfloat(
            'network', 'label_correction_threshold', fallback=0.9)

        self.psuedoLabelsGenerationEpoch = config_file.getint(
            'network', 'psuedoLabelsGenerationEpoch', fallback=3)
        self.mean_teacher_epoch = config_file.getint(
            'network', 'mean_teacher_epoch', fallback=3)
        self.num_workers = config_file.getint(
            'network', 'num_workers', fallback=0)

        self.val_batch_size = config_file.getint(
            'network', 'val_batch_size', fallback=16)

        self.generationLowerThreshold = config_file.getfloat(
            'network', 'generationLowerThreshold', fallback=0.05)
        self.generationHigherThreshold = config_file.getfloat(
            'network', 'generationHigherThreshold', fallback=0.02)

        if self.linux == True:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.linux_gpu_id

        self.backbone = config_file.get(
            'network', 'backbone', fallback='resnet34')
        self.max_iterations = config_file.getint(
            'network', 'max_iterations', fallback=30000)
        self.batch_size = config_file.getint(
            'network', 'batch_size', fallback=16)
        self.labelled_bs = config_file.getint(
            'network', 'labelled_bs', fallback=8)
        self.deterministic = config_file.getint(
            'network', 'deterministic', fallback=1)

        self.base_lr = config_file.getfloat(
            'network', 'base_lr', fallback=0.01)

        patch_size = config_file.get(
            'network', 'patch_size', fallback='[256, 256]')
        self.patch_size = [int(number)
                           for number in patch_size[1:-1].split(',')]

        self.seed = config_file.getint('network', 'seed', fallback=1337)
        self.num_classes = config_file.getint(
            'network', 'num_classes', fallback=2)
        self.in_channels = config_file.getint(
            'network', 'in_channels', fallback=1)
        if self.overload_rcnn_predictor:
            #overload blob detection
            self.model = self.create_mask_rcnn(2)
        else:
            self.model = self.create_mask_rcnn(self.num_classes)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model.to(self.device)

        if self.optim.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.base_lr,
                                       momentum=0.9, weight_decay=0.0001)
        elif self.optim.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.base_lr)
        else:
            raise Exception("Optimizer is not supported")

        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1, 10],
        #                                                          gamma=0.1)
        self.train_iou_values = []

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'max')
        # writers
        self.train_writer = None
        self.val_writer = None

        self.train_transform = self.get_transform(True)
        self.val_transform = self.get_transform(False)
        self.train_detections_transforms = self.get_transform_detection(True)
        self.val_detections_transforms = self.get_transform_detection(False)

        self.best_performance = 0
        self.start_epoch = 0

        if not self.train_mask:
            self.model.roi_heads.mask_predictor = None

        if self.load_model:
            print(self.model_path)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_performance = checkpoint['best_performance']
            self.train_iou_values = checkpoint['train_iou_values']
            self.need_label_correction = checkpoint['need_label_correction']

        if self.overload_rcnn_predictor:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.num_classes)

        self.need_label_correction = config_file.getboolean(
            'network', 'need_label_correction', fallback=False)

    def update_lr(self, iter_num):
        if self.optim.lower() == 'sgd':
            lr_ = self.base_lr * (1.0 - iter_num / self.max_iterations) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_

    def create_mask_rcnn(self, num_classes):

        model = maskrcnn_resnet50_fpn(pretrained_backbone=True, rpn_positive_fraction=0.5
                                      , rpn_fg_iou_thresh=0.7
                                      , rpn_bg_iou_thresh=0.3
                                      , box_nms_thresh=self.box_nms_thresh, box_score_thresh=self.box_score_thresh,
                                      min_size=self.min_size, max_size=self.max_size,
                                      box_detections_per_img=self.box_detections_per_img)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

        return model

    def get_transform(self, train):
        if train:
            transforms = A.Compose([
                A.Resize(self.patch_size[0], self.patch_size[1]),
                # A.RandomCrop(width=self.patch_size[0]//2, height=self.patch_size[1]//2),
                A.ChannelShuffle(),
                A.Blur(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=[0.5, 1.5], border_mode=0, value=0,
                                   mask_value=0),
                ToTensorV2(),
            ])
            #  ,bbox_params={'format':'pascal_voc', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['category_id']} )
        else:
            transforms = A.Compose(
                [A.Resize(self.patch_size[0], self.patch_size[1]),
                 ToTensorV2(),
                 ])
        return transforms

    def get_transform_detection(self, train):
        if train:
            transforms = A.Compose([
                A.Resize(self.patch_size[0], self.patch_size[1]),
                A.ChannelShuffle(),
                A.Blur(),
                # A.OneOf([
                #     A.GaussNoise(p=0.2, var_limit=0.01),
                # ], p=0.3),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # TODO scale parameter tuning (no zoom out just zoom in)
                # A.ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=[0.5, 1.5], border_mode=0, value=0,
                #                    mask_value=0),
                ToTensorV2(),
            ]
                , bbox_params={'format': 'pascal_voc', 'min_area': 0, 'min_visibility': 0,
                               'label_fields': ['category_id']})
        else:
            transforms = A.Compose(
                [A.Resize(self.patch_size[0], self.patch_size[1]),
                 ToTensorV2(),
                 ], bbox_params={'format': 'pascal_voc', 'min_area': 0, 'min_visibility': 0,
                                 'label_fields': ['category_id']})
        return transforms
