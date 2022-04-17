import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from SongUtils.MLUtils.BaseTrainers import BaseTrainer
from SongUtils.MetricUtils import AverageMeter

from utils.losses import TripletLoss
 
class PVTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        self.k = torch.nn.Parameter(torch.zeros(1, 512))
        super(PVTrainer, self).__init__(cfg, model, dataset_list, metrics_list)

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
            # print(feat_k.shape, feat_pos.shape, feat_neg.shape)
            loss = self.loss_func(feat_k, feat_pos, feat_neg)

            if isTrain :
                loss.backward()
                if (epoch_step % self.cfg.realbatchSize == 0):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            _loss.update(loss.item())
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info("Step: {}, loss: {:.4f}".format(epoch_step, _loss.avg))

 
if __name__ == "__main__":
    loss_func = TripletLoss()
    a = torch.ones(4, 512)
    p = torch.ones(4, 512)
    n = torch.ones(4, 512)
    loss = loss_func(a, p, n)
    print(loss)
    print(loss.shape)
