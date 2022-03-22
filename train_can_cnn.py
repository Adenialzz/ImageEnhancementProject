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
from utils.edit_transform import ImageEditor
from utils.utils import save_tensor_image

import argparse
def get_args():
    parser = get_dist_base_parser()
    cfg = parser.parse_args()
    return cfg

# class CANTrainer(BaseDistTrainer):
class EditorTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(EditorTrainer, self).__init__(cfg, model, dataset_list, metrics_list)
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

            edited_images, filter_tokens = self.image_editor(original_images, (768, ), single_filter=True, polar_intensity=True)
            edited_images = edited_images.to(self.device)
            filter_tokens = filter_tokens.to(self.device)
            self.model.add_filter_tokens(filter_tokens)
            # edited_images, filter_channels = self.image_editor(original_images, (224, 224), single_filter=True, polar_intensity=True)
            # filter_channels = filter_channels.expand(batchSize, 5, 224, 224).to(self.device)
            edited_images = edited_images.to(self.device).to(self.device)
            # inputs = torch.cat((original_images, filter_channels), dim=1)
            inputs = original_images

            if isTrain:
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.loss_func(output, edited_images)
                loss.backward()
                self.optimizer.step()

            else:
                output = self.model(inputs)
                loss = self.loss_func(output, edited_images)

            _loss.update(loss.item())
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info("Step: {}, loss: {:.4f}".format(epoch_step, _loss.avg))
        return {'loss': _loss.avg}

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
    # model = ViT_Editor_Channels(in_chans=8)
    model = ViT_Editor_Tokens(in_chans=3, num_filters=5)
    trainer = EditorTrainer(cfg, model, [train_set, val_set], ["loss", ])
    trainer.forward()


if __name__ == "__main__":
    cfg = get_args()
    main_worker(None, None, cfg)
    # import torch.multiprocessing as mp
    # mp.spawn(main_worker, nprocs=cfg.nprocs, args=(cfg.nprocs, cfg))
