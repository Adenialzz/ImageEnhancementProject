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
        self.k = torch.nn.Parameter(torch.zeros(1, 512))
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
            'preference_vector': self.k,
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, osp.join(self.cfg.model_path, f"model_{epoch}.pth"))
        torch.save(k_state, osp.join(self.cfg.model_path, f"pv_k_{epoch}.pth"))

    def init_loss_func(self):
        self.loss_func = TripletLoss()
    
    def init_optimizer(self):
        optim_params = [
            {'params': self.model.parameters()},
            {'params': self.k}
        ]
        
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
            data_pos = data['positive'].to(self.device)
            data_neg = data['negative'].to(self.device)
            bs = data_pos.shape[0]

            feat_pos = self.model(data_pos)
            feat_neg = self.model(data_neg)
            feat_k = self.k.expand(bs, -1).to(self.device)
            loss = self.loss_func(feat_k, feat_pos, feat_neg)

            if isTrain :
                loss.backward()
                if (epoch_step % self.cfg.realbatchSize == 0):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            _loss.update(loss.item())
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info("Step: {}, loss: {:.4f}".format(epoch_step, _loss.avg))

        metrics_dict = {}
        for metric in self.metrics_list:
            metrics_dict[metric] = eval('_' + metric).avg
        return metrics_dict

class AllPVTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        self.experts = ['A', 'B', 'C', 'D', 'E']
        self.kA = torch.nn.Parameter(torch.zeros(1, 512))
        self.kB = torch.nn.Parameter(torch.zeros(1, 512))
        self.kC = torch.nn.Parameter(torch.zeros(1, 512))
        self.kD = torch.nn.Parameter(torch.zeros(1, 512))
        self.kE = torch.nn.Parameter(torch.zeros(1, 512))
        self.k_dict = {'A': self.kA, 'B': self.kB, 'C': self.kC, 'D': self.kD, 'E': self.kE}
        super(AllPVTrainer, self).__init__(cfg, model, dataset_list, metrics_list)
    
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
        self.loss_func = TripletLoss()
    
    def init_optimizer(self):
        optim_params = [
            {'params': self.model.parameters()},
            # {'params': self.k}
        ]
        for _, v in self.k_dict.items():
            optim_params.append( {'params': v} )
        
        self.optimizer = SGD(optim_params, lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)

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
            data_neg = data[neg].to(self.device)
            bs = data_pos.shape[0]

            feat_pos = self.model(data_pos)
            feat_neg = self.model(data_neg)
            feat_k = self.k_dict[pos].expand(bs, -1).to(self.device)
            loss = self.loss_func(feat_k, feat_pos, feat_neg)

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