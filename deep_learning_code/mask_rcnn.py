from detection.engine import train_one_epoch, evaluate
from torch.backends import cudnn
import utils
import torch

from configs.configs_inst_seg import Configs

from dataloaders.instance_seg_dataset import PennFudanDataset

import detection.utils as utils

def main():
    configs = Configs('./configs/mask_rcnn.ini')

    if not configs.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('../data/PennFudanPed', configs.train_transform)
    dataset_test = PennFudanDataset('../data/PennFudanPed', configs.val_transform)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    # move model to the right device
    configs.model.to(configs.device)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(configs.model, configs.optimizer, data_loader, configs.device, epoch, print_freq=10)
        evaluate(configs.model, data_loader_test, device=configs.device)

    print("That's it!")



if __name__ == "__main__":
    main()