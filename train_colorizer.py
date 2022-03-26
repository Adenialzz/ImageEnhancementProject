import os
import os.path as osp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD

from SongUtils.MetricUtils import AverageMeter, accuracy
from SongUtils.MLUtils.BaseArgs import get_dist_base_parser
from tqdm import tqdm

from utils.dataset import ColorizationDataset
from models.nicer_models import CAN
from models.vit_editors import ViT_Editor_Channels, ViT_Editor_Tokens
from trainers.colorization_trainer import ColorizationTrainer

import argparse
def get_args():
    parser = get_dist_base_parser()
    cfg = parser.parse_args()
    return cfg

def main_worker(local_rank, nprocs, cfg):
    # init_dist(cfg.gpu_id, cfg.nprocs, local_rank)
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    ava_root = '/ssd1t/song/Datasets/AVA/shortEdge256'
    csv_root = '/home/ps/JJ_Projects/FromSong/dsmAVA/csvFiles'
    train_set = ColorizationDataset(osp.join(csv_root, 'train_mlsp.csv'), ava_root)
    val_set = ColorizationDataset(osp.join(csv_root, 'val_mlsp.csv'), ava_root)


    # model = CAN(5)
    model = ViT_Editor_Channels(in_chans=3, depth=12)
    trainer = ColorizationTrainer(cfg, model, [train_set, val_set], ["loss", ])
    trainer.forward()


if __name__ == "__main__":
    cfg = get_args()
    main_worker(None, None, cfg)
    # import torch.multiprocessing as mp
    # mp.spawn(main_worker, nprocs=cfg.nprocs, args=(cfg.nprocs, cfg))
