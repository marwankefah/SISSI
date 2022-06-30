import os

from torch.backends import cudnn
import utils
import torch
import argparse
import logging
import os
import random
import shutil
import sys
import time
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.configs_inst_seg import Configs

from dataloaders.instance_seg_dataset import cell_pose_dataset,cell_lab_dataset

import reference.utils as utils
from reference.engine import train_one_epoch, evaluate, save_check_point



def train(configs, snapshot_path):
    configs.train_writer = SummaryWriter(snapshot_path + '/log')
    configs.val_writer = SummaryWriter(snapshot_path + '/log_val')
    configs.cell_lab_test_writer = SummaryWriter(snapshot_path + '/log_cell_lab_test')

    configs.model.to(configs.device)

    db_train = cell_pose_dataset(configs.cell_pose_root_path, 'train', configs.train_transform)
    db_test = cell_pose_dataset(configs.cell_pose_root_path, 'test', configs.val_transform)
    db_cell_lab_test = cell_lab_dataset(configs.cell_lab_root_path, ['test_labelled'],
                                    configs.val_detections_transforms, cache_labels=True)


    trainloader = torch.utils.data.DataLoader(
        db_train, batch_size=configs.labelled_bs, shuffle=True, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    valloader = torch.utils.data.DataLoader(
        db_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    cell_lab_test_data_loader = torch.utils.data.DataLoader(
        db_cell_lab_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
        collate_fn=utils.collate_fn)

    configs.model.train()

    writer = configs.train_writer
    writer_val = configs.val_writer

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0

    max_epoch = configs.max_iterations // len(trainloader) + 1
    iterator = tqdm(range(configs.start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:

        train_one_epoch(configs, trainloader, epoch_num, print_freq=10, writer=writer)

        _, _, val_losses_reduced = evaluate(configs, epoch_num, valloader, device=configs.device, writer=writer_val)

        evaluate(configs, epoch_num, cell_lab_test_data_loader, configs.device,
                                   configs.cell_lab_test_writer,
                                   vis_every_iter=1)

        save_check_point(configs, epoch_num, val_losses_reduced, snapshot_path)

        configs.lr_scheduler.step(val_losses_reduced)

        if iter_num >= configs.max_iterations:
            break
        configs.model.train()
        logging.info('{} epoch finished'.format(epoch_num + 1))

    writer.close()
    writer_val.close()
    return "Training Finished!"


if __name__ == "__main__":

    configs = Configs('./configs/mask_rcnn.ini')

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

    log_time = int(time.time())

    snapshot_path = configs.model_output_path + "model/{}_labelled/{}/{}".format(
        configs.exp, configs.model_name, log_time)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(vars(configs)))
    train(configs, snapshot_path)
