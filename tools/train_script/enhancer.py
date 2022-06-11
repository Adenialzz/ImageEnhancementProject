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

from utils.dataset import EnhanceDataset, AVADataset
# from models.vit_enhancers import ViT_Enhancer_Channels
from models.nicer_models import CAN
# from trainers.vit_enhancer_trainer import EnhancerTrainer
from trainers.editor_trainers import EditorTrainer_Channels
from utils.edit_transform import ImageEditor
from utils.utils import save_tensor_image, load_weights_resize_pos_embed

import argparse
def get_args():
    parser = get_dist_base_parser()
    parser.add_argument("--emd-gamma", type=float, default=1.)
    parser.add_argument("--mse-gamma", type=float, default=1.)
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
    train_set = AVADataset(osp.join(csv_root, 'train_mlsp.csv'), ava_root, transform=pipeline)
    val_set = AVADataset(osp.join(csv_root, 'val_mlsp.csv'), ava_root, transform=pipeline)


    # model = ViT_Enhancer_Channels(in_chans=8, depth=6)
    model = CAN(no_of_filters=5)
    # model = load_weights_resize_pos_embed(model, "/media/song/ImageEnhancingResults/weights/vit_editor_channels_d6_lr1e-1/model_42.pth")
    metrics_list = [
        "emd_loss", "mse_loss", "loss",
        "plcc_mean", "srcc_mean", "plcc_std", "srcc_std",
        "acc"
    ]
    trainer = EditorTrainer_Channels(cfg, model, [train_set, val_set], metrics_list)
    trainer.forward()


if __name__ == "__main__":
    cfg = get_args()
    main_worker(None, None, cfg)
