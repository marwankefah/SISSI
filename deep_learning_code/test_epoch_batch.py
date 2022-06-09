import time

from configs.configs_inst_seg import Configs
from dataloaders.dataset import (ddsm_dataset_labelled, BaseFetaDataSets, RandomGenerator, ResizeTransform,
                                 TwoStreamBatchSampler)
from medpy import metric
from monai.data.utils import decollate_batch
from odach_our import oda
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
from PIL import Image
from reference.engine import train_one_epoch, evaluate, test, save_check_point
from dataloaders.instance_seg_dataset import chrisi_dataset
import os
from tensorboardX import SummaryWriter
import reference.utils as utils

from dataloaders.instance_seg_dataset import cell_pose_dataset

from reference.engine import test_time_augmentation

from reference.engine import correct_labels

configs = Configs('./configs/mask_rcnn_test.ini')
log_time = int(time.time())

db_chrisi_test = chrisi_dataset(configs.chrisi_cells_root_path, ['test_labelled'],
                                configs.val_detections_transforms, cache_labels=True, need_seam_less_clone=False)

db_train = cell_pose_dataset(configs.cell_pose_root_path, 'train', configs.train_transform)
db_test = cell_pose_dataset(configs.cell_pose_root_path, 'test', configs.val_transform)

weak_label_chrisi_dataset = chrisi_dataset(configs.chrisi_cells_root_path, ['alive', 'dead', 'inhib'],
                                           configs.val_detections_transforms,
                                           cache_labels=True, need_seam_less_clone=configs.need_seam_less_clone)

weak_label_chrisi_dataset_val = chrisi_dataset(configs.chrisi_cells_root_path, ['alive', 'dead', 'inhib'],
                                               configs.val_detections_transforms,
                                               cache_labels=True)

random.seed(10)
weak_label_chrisi_dataset.sample_list = random.sample(weak_label_chrisi_dataset.sample_list, 8)
weak_label_chrisi_dataset_val.sample_list = weak_label_chrisi_dataset.sample_list.copy()

initial_weak_labels_data_loader = torch.utils.data.DataLoader(
    weak_label_chrisi_dataset_val, batch_size=configs.labelled_bs, shuffle=False,
    num_workers=0,
    collate_fn=utils.collate_fn)

weak_label_chrisi_dataloader = torch.utils.data.DataLoader(
    weak_label_chrisi_dataset, batch_size=configs.labelled_bs, shuffle=False, num_workers=configs.num_workers,
    collate_fn=utils.collate_fn)


chrisi_test_data_loader = torch.utils.data.DataLoader(
    db_chrisi_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
    collate_fn=utils.collate_fn)


configs.model.to(configs.device)

if not configs.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(configs.seed)
np.random.seed(configs.seed)
torch.manual_seed(configs.seed)
torch.cuda.manual_seed(configs.seed)


epochs_dir ='/content/drive/MyDrive/IP_DL/model/cell/mix_final_labelled_labelled/faster_rcnn_resnet50_0.25/1654079041'
snapshot_path = os.path.join(epochs_dir, 'viz', str(log_time), 'test')
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

model_epoch_path_list = [epoch_model_path for epoch_model_path in os.listdir(epochs_dir) if
                         epoch_model_path.endswith(".pth")]

for idx, model_epoch_path in enumerate(model_epoch_path_list):
    print(model_epoch_path)
    checkpoint = torch.load(os.path.join(epochs_dir, model_epoch_path), map_location=configs.device)
    configs.model.load_state_dict(checkpoint['model'])
    configs.chrisi_test_writer = SummaryWriter(snapshot_path + '/log_chrisi_test/' + model_epoch_path)
    configs.inital_weak_writer = SummaryWriter(snapshot_path + '/log/' + model_epoch_path)
    configs.cum_weak_writer=SummaryWriter(snapshot_path + '/log_cum_weak/' + model_epoch_path)
    with torch.no_grad():
        # test_time_augmentation(configs, tta_model, alive_data_loader, configs.device,
        #                        writer=configs.alive_writer)
        #
        # test_time_augmentation(configs, tta_model, chrisi_test_data_loader, configs.device,
        #                        writer=configs.chrisi_test_writer)
        #
        # test_time_augmentation(configs, tta_model, dead_data_loader, configs.device,
        #                        writer=configs.dead_writer)

        evaluate(configs, idx, chrisi_test_data_loader, device=configs.device, writer=configs.chrisi_test_writer,
                 vis_every_iter=1, use_tta=True)

        # print(configs.device)
        train_iou, outputs_list_dict = evaluate(configs, idx, initial_weak_labels_data_loader, configs.device,
                                                configs.inital_weak_writer,
                                                vis_every_iter=1,use_tta=True)

        correct_labels(configs, weak_label_chrisi_dataset, outputs_list_dict, idx, 100)

        weak_label_chrisi_dataloader = torch.utils.data.DataLoader(
            weak_label_chrisi_dataset, batch_size=configs.labelled_bs, shuffle=False,
            num_workers=configs.num_workers,
            collate_fn=utils.collate_fn)

        evaluate(configs, idx, weak_label_chrisi_dataloader, configs.device,
                                                configs.cum_weak_writer,
                                                vis_every_iter=1, use_tta=True)

