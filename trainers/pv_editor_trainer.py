import torch
import os
import os.path as osp
from SongUtils.MetricUtils import AverageMeter, accuracy
from SongUtils.MLUtils.BaseTrainers import BaseTrainer, BaseDistTrainer, init_dist
from SongUtils.MLUtils.BaseArgs import get_dist_base_parser

from utils.edit_transform import ImageEditor

class PV_Editor_Trainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super().__init__(cfg, model, dataset_list, metrics_list)
        self.pv = self.get_preference_vectors(self.cfg.pv_dir, self.cfg.pv_epoch)

    def get_preference_vectors(self, model_dir, epoch):
        ckpt = torch.load(osp.join(model_dir, f'pv_k_{epoch}.pth'), map_location='cpu')
        pv = ckpt['preference_vector']
        for e, feat in pv.items():
            pv[e] = feat.detach()
        return pv

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
            original_image = data['input_image'].to(self.device)
            target_images = data['target_images']
            batchSize = original_image.shape[0]

            if isTrain:
                self.optimizer.zero_grad()

            loss = 0.
            for e in self.cfg.target_experts:
                target_image = target_images[e].to(self.device)
                pv = self.pv[e].unsqueeze(dim=2).unsqueeze(dim=3).expand(batchSize, 512, 1, 1).to(self.device)
                output_image = self.model(original_image, pv)
                loss += self.loss_func(output_image, target_image)
            loss /= len(self.cfg.target_experts)
            if isTrain:
                loss.backward()
                self.optimizer.step()
                
            _loss.update(loss.item())
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info("Epoch: {}, Step: {}, loss: {:.4f}".format(epoch, epoch_step, _loss.avg))
        return {'loss': _loss.avg}

