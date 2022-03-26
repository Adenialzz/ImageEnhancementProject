import torch
from SongUtils.MetricUtils import AverageMeter, accuracy
from SongUtils.MLUtils.BaseTrainers import BaseTrainer, BaseDistTrainer, init_dist
from SongUtils.MLUtils.BaseArgs import get_dist_base_parser

class ColorizationTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(ColorizationTrainer, self).__init__(cfg, model, dataset_list, metrics_list)

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
            gray_image = data["gray_image"].to(self.device)
            original_image = data["original_image"].to(self.device)
            batchSize = original_image.shape[0]

            if isTrain:
                self.optimizer.zero_grad()
                output = self.model(gray_image)
                loss = self.loss_func(output, original_image)
                loss.backward()
                self.optimizer.step()

            else:
                output = self.model(gray_image)
                loss = self.loss_func(output, original_image)

            _loss.update(loss.item())
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info("Step: {}, loss: {:.4f}".format(epoch_step, _loss.avg))
        return {'loss': _loss.avg}