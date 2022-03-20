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
from models.nicer_models import CAN
from sit import VisionTransformer_SiT
from edit_transform import ImageEditor

import argparse
def get_args():
    parser = get_dist_base_parser()
    parser.add_argument("--resume", type=str, default=None)
    cfg = parser.parse_args()
    return cfg

# class CANTrainer(BaseDistTrainer):
class CANTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(CANTrainer, self).__init__(cfg, model, dataset_list, metrics_list)
        self.image_editor = ImageEditor()

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
        for epoch_step, data in enumerate(loader):
            original_images = data["image"].to(self.device)
            batchSize = original_images.shape[0]
            edited_images, filter_channels = self.image_editor(original_images, sinle_filter=True, polar_intensity=True)
            filter_channels = filter_channels.expand(batchSize, 5, 224, 224).to(self.device)
            edited_images = edited_images.to(self.device)
            inputs = torch.cat((original_images, filter_channels), dim=1)

            if isTrain:
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.loss_func(output, edited_images)
                loss.backward()
            else:
                output = self.model(inputs)
                loss = self.loss_func(output, edited_images)

            if isTrain:
                self.optimizer.step()
            _loss.update(loss.item())
            if epoch_step % self.cfg.log_freq == 0:
                print(f"Step: {epoch_step}, loss: {_loss.avg}")
        return {'loss': _loss.avg}

def main_worker(local_rank, nprocs, cfg):
    # init_dist(cfg.gpu_id, cfg.nprocs, local_rank)
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # dset = EnhanceDataset("./test1w/target_images", "./test1w/origin1w", transform=pipeline)
    ava_root = '/home/song/AVA/shortEdge256'
    csv_root = '/home/song/JJ_Projects/dsmAVA/csvFiles'
    train_set = AVADataset(osp.join(csv_root, 'train_mlsp.csv'), ava_root, transform=pipeline)
    val_set = AVADataset(osp.join(csv_root, 'val_mlsp.csv'), ava_root, transform=pipeline)


    model = CAN(in_planes=8)
    # model = VisionTransformer_SiT(in_chans=8)
    trainer = CANTrainer(cfg, model, [train_set, val_set], ["loss", ])
    trainer.forward()

def print_args(cfg):
    print('*'*21, " - Configs - ", '*'*21)
    for k, v in vars(cfg).items():
        print(k, ':', v)
    print('*'*56)

if __name__ == "__main__":
    cfg = get_args()
    print_args(cfg)
    main_worker(None, None, cfg)
    # import torch.multiprocessing as mp
    # mp.spawn(main_worker, nprocs=cfg.nprocs, args=(cfg.nprocs, cfg))
