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
sys.path.append(os.getcwd())

from utils.dataset import FiveKDataset4PV_Editor
from models.pie_cnn import PV_Editor
from models.SimpleHRNet import HRNet
from trainers.pv_editor_trainer import PV_Editor_Trainer
from utils.edit_transform import ImageEditor
from utils.utils import save_tensor_image, load_weights

import argparse
def get_args():
    parser = get_dist_base_parser()
    parser.add_argument('--pv_dir', type=str)
    parser.add_argument('--pv_epoch', type=int)
    parser.add_argument('--input_expert', type=str, default='I')
    parser.add_argument('--arch', type=str, default='can')
    parser.add_argument('--target_experts', type=str, default='ABCDEFGH')
    cfg = parser.parse_args()
    return cfg

def main_worker(local_rank, nprocs, cfg):
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    fivek_root = 'FiveK/'
    all_fivek_image_list = os.listdir(osp.join(fivek_root, 'I'))
    train_set = FiveKDataset4PV_Editor(fivek_root, cfg.target_experts, all_fivek_image_list[: 4500], cfg.input_expert, transform=pipeline)
    val_set = FiveKDataset4PV_Editor(fivek_root, cfg.target_experts, all_fivek_image_list[4500: ], cfg.input_expert, transform=pipeline)

    if cfg.arch == 'can':
        model = PV_Editor()
    elif cfg.arch == 'hrnet':
        model = HRNet(16, 3, 2, 0.1)
    else:
        print(f'Unknown Arch: {cfg.arch}')
        sys.exit(-1)

    trainer = PV_Editor_Trainer(cfg, model, [train_set, val_set], ["loss", ])
    trainer.forward()


if __name__ == "__main__":
    cfg = get_args()
    main_worker(None, None, cfg)
