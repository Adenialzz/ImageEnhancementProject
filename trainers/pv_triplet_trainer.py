import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import os
import os.path as osp
import random 
import itertools

from SongUtils.MLUtils.BaseTrainers import BaseTrainer
from SongUtils.MetricUtils import AverageMeter

from utils.losses import TripletLoss

class PVTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        self.experts = list(cfg.experts)
        self.k_dict = {
            e: torch.nn.Parameter(torch.randn(1, cfg.feat_dim))
            for e in self.experts
        }
        super(PVTrainer, self).__init__(cfg, model, dataset_list, metrics_list)
    
    def save_model(self, epoch):
        if not osp.isdir(self.cfg.model_path):
            os.mkdir(self.cfg.model_path)
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        k_state = {
            'preference_vector': self.k_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, osp.join(self.cfg.model_path, f"model_{epoch}.pth"))
        torch.save(k_state, osp.join(self.cfg.model_path, f"pv_k_{epoch}.pth"))

    def init_loss_func(self):
        self.loss_func = TripletLoss(margin=self.cfg.margin)
    
    def update_margin(self):
        pass
    
    def init_optimizer(self):
        optim_params = [
            {'params': self.model.parameters()},
        ]
        for _, v in self.k_dict.items():
            optim_params.append(
                {'params': v}
            )
        
        self.optimizer = SGD(optim_params, lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)

    def epoch_forward(self, epoch, isTrain):
        if isTrain:
            loader = self.train_loader
            self.model.train()
        else:
            loader = self.val_loader
            self.model.eval()

        _loss = AverageMeter()
        for epoch_step, data in enumerate(loader):
            bs = data['I'].shape[0]
            
            feat_dict = {}
            # forward model
            for e, image in data.items():
                image = image.to(self.device)
                feat = self.model(image)
                feat_dict[e] = feat

            # calculate loss
            loss = 0.
            cal_cnt = 0
            for e_pos, feat_pos in feat_dict.items():
                feat_k = self.k_dict[e_pos].expand(bs, -1).to(self.device)
                for e_neg, feat_neg in feat_dict.items():
                    if e_pos == e_neg:
                        continue
                    loss += self.loss_func(feat_k, feat_pos, feat_neg)
                    cal_cnt += 1
            loss /= cal_cnt

            if isTrain :
                loss.backward()
                if (epoch_step % self.cfg.realbatchSize == 0):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            _loss.update(loss.item())
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info("Step: {}, loss: {:.4f}".format(epoch_step, _loss.avg))
            # break

        metrics_dict = {}
        for metric in self.metrics_list:
            metrics_dict[metric] = eval('_' + metric).avg
        return metrics_dict

class PVTrainer_Depc(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        self.experts = list(cfg.experts)
        self.k_dict = {
            # e: torch.nn.Parameter(torch.zeros(1, cfg.feat_dim))
            e: torch.nn.Parameter(torch.randn(1, cfg.feat_dim))
            for e in self.experts
        }
        super(PVTrainer_Depc, self).__init__(cfg, model, dataset_list, metrics_list)
    
    def save_model(self, epoch):
        if not osp.isdir(self.cfg.model_path):
            os.mkdir(self.cfg.model_path)
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.model_optimizer.state_dict(),
            'epoch': epoch
        }
        k_state = {
            'preference_vector': self.k_dict,
            'optimizer': self.pv_optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, osp.join(self.cfg.model_path, f"model_{epoch}.pth"))
        torch.save(k_state, osp.join(self.cfg.model_path, f"pv_k_{epoch}.pth"))

    def init_loss_func(self):
        self.loss_func = TripletLoss()
    
    def init_optimizer(self):
        pv_params = []
        for e, feat_k in self.k_dict.items():
            pv_params.append(
                {'params': feat_k}
            )
        
        self.model_optimizer = SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)
        self.pv_optimizer = SGD(pv_params, lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)

    def epoch_forward(self, epoch, isTrain):
        if isTrain:
            loader = self.train_loader
            self.model.train()
        else:
            loader = self.val_loader
            self.model.eval()

        _loss = AverageMeter()
        for epoch_step, (data, pos) in enumerate(zip(loader, itertools.cycle(self.experts))):
            data_pos = data[pos].to(self.device)
            e = self.experts.copy()
            e.remove(pos)
            neg = random.choice(e)
            # self.logger.info(f"pos: {pos}, neg: {neg}")
            data_neg = data[neg].to(self.device)
            
            bs = data_pos.shape[0]

            feat_pos = self.model(data_pos)
            feat_neg = self.model(data_neg)
            feat_k = self.k_dict[pos].expand(bs, -1).to(self.device)
            loss = self.loss_func(feat_k, feat_pos, feat_neg)

            if isTrain :
                loss.backward()
                if (epoch_step % self.cfg.realbatchSize == 0):
                    # self.optimizer.step()
                    # self.optimizer.zero_grad()
                    self.model_optimizer.step()
                    self.model_optimizer.zero_grad()
                    self.pv_optimizer.step()
                    self.pv_optimizer.zero_grad()

            _loss.update(loss.item())
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info("Step: {}, loss: {:.4f}".format(epoch_step, _loss.avg))
            # break

        metrics_dict = {}
        for metric in self.metrics_list:
            metrics_dict[metric] = eval('_' + metric).avg
        return metrics_dict