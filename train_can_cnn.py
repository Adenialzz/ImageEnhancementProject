import os
import os.path as osp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD

from SongUtils.MetricUtils import AverageMeter, accuracy
from SongUtils.MLUtils.BaseTrainers import BaseTrainer, BaseDistTrainer, init_dist
from SongUtils.MLUtils.BaseArgs import get_dist_base_parser
from tqdm import tqdm

from dataset import EnhanceDataset
from CAN_FCN import CAN

import argparse
def get_args():
    parser = get_dist_base_parser()
    cfg = parser.parse_args()
    return cfg

class CANTrainer(BaseDistTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(CANTrainer, self).__init__(cfg, model, dataset_list, metrics_list)

    def init_loss_func(self):
        self.loss_func = torch.nn.MSELoss()

    def epoch_forward(self, epoch):
        _loss = AverageMeter()
        for origin, target in self.train_loader:
            origin = origin.to(self.device)
            target = target.to(self.device)
            output = self.model(origin)
            loss = self.loss_func(output, target)
            _loss.update(loss.item())
            loss.backward()
            self.optimizer.step()
        return {'loss': _loss.avg}
    
    def forward(self):
        self.logger.info("Start Training")
        for epoch in range(self.cfg.epochs):
            self.logger.info(f"Epoch = {epoch}")
            train_metric_dict = self.epoch_forward(epoch)
            self.plot_epoch_metric(epoch, train_metric_dict, train_metric_dict)
            self.save_model(epoch)


def main_worker(local_rank, nprocs, cfg):
    init_dist(cfg.gpu_id, cfg.nprocs, local_rank)
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dset = EnhanceDataset("./test1w/target_images", "./test1w/origin1w", transform=pipeline)

    model = CAN(in_planes=5, d=10, w=32)
    trainer = CANTrainer(cfg, model, [dset, dset], ["loss", ])
    trainer.forward()

if __name__ == "__main__":
    cfg = get_args()
    import torch.multiprocessing as mp
    mp.spawn(main_worker, nprocs=cfg.nprocs, args=(cfg.nprocs, cfg))