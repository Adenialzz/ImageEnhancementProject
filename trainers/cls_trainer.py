import sys
import torch
from SongUtils.MLUtils.BaseTrainers import BaseTrainer
from SongUtils.MetricUtils import AverageMeter, accuracy

class ClassificationTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super().__init__(cfg, model, dataset_list, metrics_list)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def init_loss_func(self):
        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def epoch_forward(self, isTrain, epoch):
        _loss = AverageMeter()
        _acc = AverageMeter()

        if isTrain:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader
        
        for epoch_step, data in enumerate(loader):
            image = data[0].to(self.device)
            label = data[1].to(self.device)
            if isTrain:
                self.optimizer.zero_grad()
            output = self.model(image)
            output = self.softmax(output)
            loss = self.loss_func(output, label)
            acc = accuracy(output, label, [1, ])[0]
            if isTrain:
                loss.backward()
                self.optimizer.step()
            
            _loss.update(loss.item())
            _acc.update(acc)
            if (epoch_step + 1) % self.cfg.log_freq == 0:
                self.logger.info(f"Epoch: {epoch}/{self.cfg.epochs}, Step: {epoch_step}/{len(loader)}")
                for metric in self.metrics_list:
                    self.logger.info(f"\t {metric}: {eval(f'_{metric}.avg'):.2f}")

        metrics_dict = {}
        for metric in self.metrics_list:
            metrics_dict[metric] = eval('_' + metric).avg
        return metrics_dict
