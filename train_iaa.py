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

from utils.dataset import EnhanceDataset, AVADataset
from models.nicer_models import CAN
from models.iaa_models import IAAModel
from trainers.iaa_trainer import IAATrainer
from utils.edit_transform import ImageEditor
from utils.utils import save_tensor_image, load_weights_resize_pos_embed, load_weights, load_timm_weights
from models.vit_components import Mlp, Attention, Block, PatchEmbed

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
    train_set = AVADataset(osp.join(csv_root, 'train_mlsp.csv'), ava_root, transform=pipeline)
    val_set = AVADataset(osp.join(csv_root, 'val_mlsp.csv'), ava_root, transform=pipeline)


    model = IAAModel(in_chans=3, depth=12)
    # model = load_weights_resize_pos_embed(model, './colorizer_weights/model_7.pth')

    # model = timm.create_model('vit_base_patch16_224_in21k', num_classes=10)
    # model = load_timm_weights(model, '/home/ps/.cache/torch/hub/checkpoints//jx_vit_base_patch16_224_in21k-e5005f0a.pth')

    metrics_list = [
        "loss", "acc",
        "plcc_mean", "srcc_mean", "plcc_std", "srcc_std"
    ]
    trainer = IAATrainer(cfg, model, [train_set, val_set], metrics_list)
    trainer.forward()


if __name__ == "__main__":
    cfg = get_args()
    main_worker(None, None, cfg)
