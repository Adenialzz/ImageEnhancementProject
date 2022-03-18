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

from dataset import EnhanceDataset, AVADataset
from CAN_FCN import CAN
from sit import VisionTransformer_SiT

import argparse
def get_args():
    parser = get_dist_base_parser()
    cfg = parser.parse_args()
    return cfg

# class CANTrainer(BaseDistTrainer):
class CANTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(CANTrainer, self).__init__(cfg, model, dataset_list, metrics_list)

    def init_loss_func(self):
        self.loss_func = torch.nn.MSELoss()

    def epoch_forward(self, epoch, isTrain):
        if isTrain:
            loader = self.train_loader
            self.model.train()
        else:
            loader = self.val_loader
            self.model.eval()

        _loss = AverageMeter()
        for origin, target, filters in loader:
            origin = origin.to(self.device)
            filters = filters.to(self.device)
            target = target.to(self.device)

            inputs = torch.cat((origin, filters), dim=1)
            output = self.model(inputs)
            loss = self.loss_func(output, target)
            _loss.update(loss.item())
            loss.backward()
            self.optimizer.step()
        return {'loss': _loss.avg}

def main_worker(local_rank, nprocs, cfg):
    # init_dist(cfg.gpu_id, cfg.nprocs, local_rank)
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # dset = EnhanceDataset("./test1w/target_images", "./test1w/origin1w", transform=pipeline)
    ava_root = '/ssd1t/song/Datasets/AVA/shortEdge256'
    csv_root = '/home/song/JJ_Projects/FromSong/dsmAVA/csvFiles'
    train_set = AVADataset(osp.join(csv_root, 'train_mlsp.csv'), ava_root)
    val_set = AVADataset(osp.join(csv_file, 'val_mlsp.csv'), ava_root)


    # model = CAN(in_planes=6, d=10, w=32)
    model = VisionTransformer_SiT(in_chans=6)
    trainer = CANTrainer(cfg, model, [train_set, val_set], ["loss", ])
    trainer.forward()

if __name__ == "__main__":
    cfg = get_args()
    main_worker(None, None, cfg)
    # import torch.multiprocessing as mp
    # mp.spawn(main_worker, nprocs=cfg.nprocs, args=(cfg.nprocs, cfg))
