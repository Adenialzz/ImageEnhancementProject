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
from models.nicer_models import CAN
from models.vit_editors import ViT_Editor_Channels, ViT_Editor_Tokens
from trainers.editor_trainers import EditorTrainer_Channels, EditorTrainer_Tokens
from utils.edit_transform import ImageEditor
from utils.utils import save_tensor_image, load_weights

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


    # model = CAN(5)
    model = ViT_Editor_Channels(in_chans=8, depth=6)
    # model = ViT_Editor_Tokens(in_chans=3, num_filters=5, depth=6)
    trainer = EditorTrainer_Channels(cfg, model, [train_set, val_set], ["loss", ])
    # trainer = EditorTrainer_Tokens(cfg, model, [train_set, val_set], ["loss", ])
    trainer.forward()


if __name__ == "__main__":
    cfg = get_args()
    main_worker(None, None, cfg)
    # import torch.multiprocessing as mp
    # mp.spawn(main_worker, nprocs=cfg.nprocs, args=(cfg.nprocs, cfg))
