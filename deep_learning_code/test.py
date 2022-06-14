import time
from configs.configs_inst_seg import Configs
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

configs = Configs('./configs/mask_rcnn_mix_colab.ini')
log_time = int(time.time())

snapshot_path = os.path.join(configs.model_path.split('.pth')[0], str(log_time), 'test')

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

configs.chrisi_test_writer = SummaryWriter(snapshot_path + '/log_chrisi_test')

db_chrisi_test = chrisi_dataset(configs.chrisi_cells_root_path, ['test_labelled'],
                                configs.val_detections_transforms, cache_labels=True,need_seam_less_clone=False)

db_test = cell_pose_dataset(configs.cell_pose_root_path, 'test', configs.val_transform)

chrisi_test_data_loader = torch.utils.data.DataLoader(
    db_chrisi_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
    collate_fn=utils.collate_fn)
cell_pose_test_dataloader = torch.utils.data.DataLoader(
    db_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
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

medpy_dice_list = []

with torch.no_grad():

    evaluate(configs, 1, chrisi_test_data_loader, device=configs.device, writer=configs.chrisi_test_writer,
             vis_every_iter=1)
