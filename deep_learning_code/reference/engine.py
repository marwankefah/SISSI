import math
import sys
import time
import torch
import numpy as np
import logging
from reference.coco_utils import get_coco_api_from_dataset
from reference.coco_eval import CocoEvaluator
import reference.utils as utils
from collections import Counter


def train_one_epoch(configs, data_loader, epoch, print_freq, writer):
    train_loss_list = []
    configs.model.train()

    header = 'Epoch: [{}]'.format(epoch)
    train_loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0,
                       'loss_rpn_box_reg': 0}
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(configs.optimizer, warmup_iters, warmup_factor)
    total_iter_per_epoch = len(data_loader)

    for iter_epoch, (images, targets) in enumerate(data_loader):
        images = list(image.to(configs.device) for image in images)


        targets = [{k: v.to(configs.device) for k, v in t.items()} for t in targets]

        loss_dict, outputs = configs.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        configs.optimizer.zero_grad()
        losses.backward()
        configs.optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        writer.add_scalar('info/lr', configs.optimizer.param_groups[0]["lr"], epoch)

        train_loss_dict = Counter(train_loss_dict) + Counter(loss_dict_reduced)

        loss_str = []
        for name, meter in loss_dict_reduced.items():
            loss_str.append(
                "{}: {}".format(name, str(round(float(meter), 5)))
            )

        logging.info('{} [{}/{}] loss:{} '.format(header, iter_epoch, total_iter_per_epoch,
                                                  round(loss_value, 4)) + "\t".join(loss_str))

        # TODO add images and predictions and masks to tensorboard
    train_losses_reduced = sum(loss for loss in loss_dict_reduced.values()) / total_iter_per_epoch
    loss_str = []
    for name, meter in train_loss_dict.items():
        writer.add_scalar('info/' + str(name), float(meter) / total_iter_per_epoch, epoch)
        loss_str.append(
            "{}: {}".format(name, str(round(float(meter) / total_iter_per_epoch, 5)))
        )

    logging.info('{}  finished [{}/{}] lr: {} loss:{} '.format(header, iter_epoch, total_iter_per_epoch,
                                                               configs.optimizer.param_groups[0]["lr"],
                                                               train_losses_reduced) + "\t".join(loss_str))

    return


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    iou_types.append("segm")
    return iou_types


@torch.no_grad()
def evaluate(model,epoch, data_loader, device, writer):
    n_threads = torch.get_num_threads()
    # FIXME (i need someone to fix me ) remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    header = 'Test: Epoch [{}]'.format(epoch)
    val_loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0,
                     'loss_rpn_box_reg': 0}
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    total_iter_per_epoch = len(data_loader)
    for iter_per_epoch, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        loss_dict, outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        coco_evaluator.update(res)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        val_loss_dict = Counter(val_loss_dict) + Counter(loss_dict_reduced)

    # gather the stats from all processes

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    val_losses_reduced = sum(loss for loss in loss_dict_reduced.values()) / total_iter_per_epoch
    loss_str = []
    for name, meter in val_loss_dict.items():
        writer.add_scalar('info/' + str(name), float(meter) / total_iter_per_epoch, epoch)
        loss_str.append(
            "{}: {}".format(name, str(round(float(meter) / total_iter_per_epoch, 5)))
        )

    logging.info('{}  finished [{}/{}] loss:{} '.format(header, iter_per_epoch, total_iter_per_epoch,
                                                               val_losses_reduced) + "\t".join(loss_str))

    torch.set_num_threads(n_threads)
    return coco_evaluator
