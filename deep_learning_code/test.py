import time
from configs.configs_inst_seg import Configs
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
from reference.engine import evaluate
from dataloaders.instance_seg_dataset import cell_lab_dataset
import os
from tensorboardX import SummaryWriter
import reference.utils as utils
from dataloaders.instance_seg_dataset import cell_pose_dataset

configs = Configs('./configs/mask_rcnn_test.ini')
log_time = int(time.time())
snapshot_path = os.path.join(configs.model_path.split('.pth')[0], str(log_time), 'test')

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

configs.cell_lab_test_writer = SummaryWriter(snapshot_path + '/log_cell_lab_test')

cell_lab_db_test = cell_lab_dataset(configs.chrisi_cells_root_path, ['test_labelled'],
                                    configs.val_detections_transforms, cache_labels=True,
                                    need_seam_less_clone=False, is_test=True)

cell_lab_db_train = cell_lab_dataset(configs.chrisi_cells_root_path, ['inhib','dead'],
                                     configs.val_detections_transforms, cache_labels=True,
                                     need_seam_less_clone=False, is_test=True)
random.seed(9001)

cell_lab_db_train.sample_list = random.sample(cell_lab_db_train.sample_list, 50)

cell_pose_db_test = cell_pose_dataset(configs.cell_pose_root_path, 'test', configs.val_transform)

cell_lab_test_data_loader = torch.utils.data.DataLoader(
    cell_lab_db_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
    collate_fn=utils.collate_fn)
cell_pose_test_dataloader = torch.utils.data.DataLoader(
    cell_pose_db_test, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
    collate_fn=utils.collate_fn)

cell_lab_train_data_loader = torch.utils.data.DataLoader(
    cell_lab_db_train, batch_size=configs.val_batch_size, shuffle=False, num_workers=configs.num_workers,
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

print(configs.model_path)
checkpoint = torch.load(os.path.join(configs.model_path), map_location=configs.device)
configs.model.load_state_dict(checkpoint)
with torch.no_grad():
    evaluate(configs, 1, cell_lab_test_data_loader, device=configs.device, writer=configs.cell_lab_test_writer,
             vis_every_iter=1)

    # evaluate(configs, 200, cell_lab_test_data_loader, device=configs.device, writer=configs.cell_lab_test_writer,
    #          vis_every_iter=1, use_tta=True)

    # evaluate(configs, 200, cell_lab_train_data_loader, device=configs.device, writer=configs.cell_lab_test_writer,
    #          vis_every_iter=1, use_tta=True)