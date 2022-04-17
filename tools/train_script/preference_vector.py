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

import sys
sys.path.append('/home/ps/JJ_Projects/ImageEnhancementProject')

from utils.dataset import FiveKDataset
from models.feat_extractor import FeatExtractor
from trainers.pv_triplet_trainer import PVTrainer
from utils.utils import save_tensor_image, load_weights

import argparse
def get_args():
    parser = get_dist_base_parser()
    parser.add_argument('--realbatchSize', type=int, default=64)
    cfg = parser.parse_args()
    return cfg

def main_worker(local_rank, nprocs, cfg):
    # init_dist(cfg.gpu_id, cfg.nprocs, local_rank)
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    data_root = '/ssd1t/song/Datasets/FiveK'
    train_image_name_list = os.listdir(osp.join(data_root, 'A'))[: 4500]
    val_image_name_list = os.listdir(osp.join(data_root, 'A'))[4500: ]
    train_set = FiveKDataset(data_root, target='A', image_name_list=train_image_name_list, transform=pipeline)
    val_set = FiveKDataset(data_root, target='A', image_name_list=val_image_name_list, transform=pipeline)

    model = FeatExtractor()
    trainer = PVTrainer(cfg, model, [train_set, val_set], ["loss", ])
    trainer.forward()

if __name__ == "__main__":
    cfg = get_args()
    main_worker(None, None, cfg)
    # import torch.multiprocessing as mp
    # mp.spawn(main_worker, nprocs=cfg.nprocs, args=(cfg.nprocs, cfg))
