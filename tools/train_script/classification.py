import os
import os.path as osp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD

import timm

from SongUtils.MetricUtils import AverageMeter, accuracy
from SongUtils.MLUtils.BaseTrainers import BaseTrainer, BaseDistTrainer, init_dist
from SongUtils.MLUtils.BaseArgs import get_dist_base_parser
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())

from torchvision.datasets import ImageFolder
from utils.dataset import EnhanceDataset, AVADataset
from trainers.cls_trainer import ClassificationTrainer
from utils.edit_transform import ImageEditor
from utils.utils import save_tensor_image, load_weights_resize_pos_embed, load_weights, load_timm_weights

import argparse
def get_args():
    parser = get_dist_base_parser()
    parser.add_argument('--n_classes', type=int, default=9)
    parser.add_argument('--dataset', type=str, default='t_inet', choices=['t_inet', 'fivek'])
    cfg = parser.parse_args()
    return cfg

def main_worker(local_rank, nprocs, cfg):
    # init_dist(cfg.gpu_id, cfg.nprocs, local_rank)
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    if cfg.dataset == 'fivek':
        trainpath = '/ssd1t/song/Datasets/FiveK/ImageFolders/'
        valpath = '/ssd1t/song/Datasets/FiveK/ImageFolders/'
    elif cfg.dataset == 't_inet':
        trainpath = 't_inet/'
        valpath = 't_inet/'
    train_set = ImageFolder(trainpath, transform=pipeline)
    val_set = ImageFolder(trainpath, transform=pipeline)

    model = timm.create_model('resnet18', num_classes=cfg.n_classes, pretrained=True)
    # model = load_timm_weights(model, '/home/ps/.cache/torch/hub/checkpoints/jx_vit_base_patch16_224_in21k-e5005f0a.pth')

    metrics_list = [
        "loss", "acc",
    ]
    trainer = ClassificationTrainer(cfg, model, [train_set, val_set], metrics_list)
    trainer.forward()


if __name__ == "__main__":
    cfg = get_args()
    main_worker(None, None, cfg)
