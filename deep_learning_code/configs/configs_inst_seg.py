# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:47:42 2021

@author: Prinzessin
"""
import configparser
import os
import torch.optim as optim
from monai.transforms.utility.dictionary import Lambdad

import reference.transforms as T
from torchvision_our.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision_our.models.detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn

from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityd,
    EnsureTyped,
    Resized,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Rand2DElasticd,
    RandAffined,
    OneOf,
    NormalizeIntensity,
    AsChannelFirstd,
    EnsureType,
    LabelToMaskd
)
from monai.data.image_reader import PILReader, NibabelReader
from monai.metrics import DiceMetric

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

        self.live_cells_root_path = config_file.get('path', 'live_cells_root_path', fallback='../data/FETA/')
        self.live_cells_img_path = config_file.get('path', 'live_cells_img_path', fallback='../data/FETA/')

        self.cell_pose_root_path = config_file.get('path', 'cell_pose_root_path', fallback='../data/FETA/')
        self.cell_pose_img_root_path = config_file.get('path', 'cell_pose_img_root_path', fallback='../data/FETA/')

        self.chrisi_cells_root_path = config_file.get('path', 'chrisi_cells_root_path', fallback='../data/FETA/')
        self.chrisi_cells_img_path = config_file.get('path', 'chrisi_cells_img_path', fallback='../data/FETA/')

        self.linux_gpu_id = config_file.get('path', 'linux_gpu_id', fallback=0)
        self.linux = config_file.getboolean('path', 'linux', fallback=False)
        self.model_path = config_file.get('path', 'model_path', fallback='')
        self.fold = config_file.getint('path', 'fold', fallback=0)

        self.load_model = config_file.getint('path', 'load_model', fallback=1)

        self.exp = config_file.get('path', 'exp', fallback='FETA/Mean_Teacher')
        self.model_name = config_file.get('path', 'model_name', fallback='unetResnet34')

        self.multi_class = config_file.getboolean('network', 'multi_class', fallback=True)

        self.optim = config_file.get('network', 'optim', fallback='adam')

        self.lr_step_size = config_file.getfloat('network', 'lr_step_size', fallback=8)
        self.lr_gamma = config_file.getfloat('network', 'lr_gamma', fallback=0.1)

        self.psuedoLabelsGenerationEpoch = config_file.getint('network', 'psuedoLabelsGenerationEpoch', fallback=3)
        self.mean_teacher_epoch = config_file.getint('network', 'mean_teacher_epoch', fallback=3)
        self.num_workers = config_file.getint('network', 'num_workers', fallback=0)

        self.val_batch_size = config_file.getint('network', 'val_batch_size', fallback=16)

        self.generationLowerThreshold = config_file.getfloat('network', 'generationLowerThreshold', fallback=0.05)
        self.generationHigherThreshold = config_file.getfloat('network', 'generationHigherThreshold', fallback=0.02)

        if self.linux == True:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.linux_gpu_id

        self.backbone = config_file.get('network', 'backbone', fallback='resnet34')
        self.max_iterations = config_file.getint('network', 'max_iterations', fallback=30000)
        self.batch_size = config_file.getint('network', 'batch_size', fallback=16)
        self.labelled_bs = config_file.getint('network', 'labelled_bs', fallback=8)
        self.deterministic = config_file.getint('network', 'deterministic', fallback=1)

        self.base_lr = config_file.getfloat('network', 'base_lr', fallback=0.01)

        patch_size = config_file.get('network', 'patch_size', fallback='[256, 256]')
        self.patch_size = [int(number) for number in patch_size[1:-1].split(',')]

        self.seed = config_file.getint('network', 'seed', fallback=1337)
        self.num_classes = config_file.getint('network', 'num_classes', fallback=2)
        self.in_channels = config_file.getint('network', 'in_channels', fallback=1)

        # Model

        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            # dropout=0.5,  # dropout ratio, default is None
            # activation='sigmoid',  # activation function, default is None
            classes=self.num_classes,  # define number of output labels
        )

        self.model = self.create_mask_rcnn(self.num_classes)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model.to(self.device)

        if self.optim.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.base_lr,
                                       momentum=0.9, weight_decay=0.0001)
        elif self.optim.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr)
        else:
            raise Exception("Optimizer is not supported")

        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1, 10],
        #                                                          gamma=0.1)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        # writers
        self.train_writer = None
        self.val_writer = None

        image_loader = None
        channel_transform = None
        lambda_transform_channel = None

        if self.in_channels == 1:
            # TODO force the image to have one dim
            image_loader = PILReader(converter=lambda image: image)
            # TODO instead of dividing by 255 to make the label mask between 0 and 1 ,make it more generic
            # TODO Scale intensity?
            lambda_transform_channel = Lambdad(keys=["label"], func=[lambda x: x / 255])

            # image_loader= NibabelReader()
            # lambda_transform_channel= Lambdad(keys=["image","label"],func=[lambda x: x[:,:,0],lambda x: x[:,:,0]])

            channel_transform = AddChanneld(keys=["image", "label"])
        elif self.in_channels == 3:
            # TODO force the image to have 3 dim
            image_loader = PILReader(converter=lambda image: image)
            # lambda_transform_channel = Lambdad(keys=["label"], func=[lambda x:x[:,:,0]])
            # TODO instead of dividing by 255 to make the label mask between 0 and 1 ,make it more generic
            # TODO Scale intensity?
            lambda_transform_channel = Lambdad(keys=["label"], func=[lambda x: x[:, :, 0] / 255])

            # image_loader= NibabelReader()
            # lambda_transform_channel= Lambdad(keys=["label"],func=[lambda x: x,lambda x: x[:,:,0]])
            channel_transform = AsChannelFirstd(keys=["image", "label"])
        else:
            raise Exception("input channel is not supported")

        deform = Rand2DElasticd(
            keys=["image", "label"],
            prob=0.5,
            spacing=(7, 7),
            magnitude_range=(1, 2),
            rotate_range=(np.pi / 6,),
            scale_range=(0.2, 0.2),
            translate_range=(20, 20),
            padding_mode="zeros",
            # device=self.device,
        )

        affine = RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(np.pi / 6),
            scale_range=(0.2, 0.2),
            translate_range=(20, 20),
            padding_mode="zeros",
            # device=self.device
        )

        # self.train_transform = Compose(
        #     [
        #         LoadImaged(keys=["image", "label"], reader=image_loader),
        #         lambda_transform_channel,
        #         channel_transform,
        #
        #         LabelToMaskd(keys=["label"], select_labels=[1]),
        #
        #         ScaleIntensityd(keys=["image", "label"]),
        #
        #         Resized(keys=["image", "label"], spatial_size=(self.patch_size[0], self.patch_size[1])),
        #         RandRotated(keys=["image", "label"], range_x=(-np.pi / 6, np.pi / 6), prob=0.5, keep_size=True),
        #
        #         RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        #         RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        #
        #         RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
        #
        #         RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
        #         RandGaussianNoised(keys=["image"], mean=0, std=0.1, prob=0.5),
        #
        #         OneOf(transforms=[affine, deform], weights=[0.8, 0.2]),
        #         # NormalizeIntensity(subtrahend=None, divisor=None, channel_wise=False),
        #
        #         EnsureTyped(keys=["image", "label"], ),
        #     ]
        # )

        # self.val_transform = Compose(
        #     [
        #         LoadImaged(keys=["image", "label"], reader=image_loader),
        #         lambda_transform_channel,
        #         channel_transform,
        #
        #         LabelToMaskd(keys=["label"], select_labels=[1]),
        #
        #         ScaleIntensityd(keys=["image", "label"]),
        #
        #         # NormalizeIntensity(subtrahend=None, divisor=None, channel_wise=False),
        #
        #         Resized(keys=["image", "label"], spatial_size=(self.patch_size[0], self.patch_size[1])),
        #         EnsureTyped(keys=["image", "label"])
        #     ])

        self.train_transform = self.get_transform(True)
        self.val_transform = self.get_transform(False)

        # self.y_pred_trans = Compose(
        #     [EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=self.num_classes)])

        # self.y_trans = AsDiscrete(threshold=0.1, to_onehot=self.num_classes)

        # self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_performance = 0
        self.start_epoch=0
        if self.load_model:
            print(self.model_path)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_performance = checkpoint['best_performance']

    def update_lr(self, iter_num):
        if self.optim.lower() == 'sgd':
            lr_ = self.base_lr * (1.0 - iter_num / self.max_iterations) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_

    def create_mask_rcnn(self, num_classes):

        model = maskrcnn_resnet50_fpn(pretrained=True, min_size=500, max_size=800)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

        return model

    def get_transform(self, train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
