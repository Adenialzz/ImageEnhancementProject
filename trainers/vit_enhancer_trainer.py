import torch
import numpy as np

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn import metrics

from SongUtils.MetricUtils import AverageMeter, accuracy
from SongUtils.MLUtils.BaseTrainers import BaseTrainer, BaseDistTrainer, init_dist
from SongUtils.MLUtils.BaseArgs import get_dist_base_parser

from utils.edit_transform import ImageEditor
from utils.loss import emd_loss
from utils.utils import get_score

# class CANTrainer(BaseDistTrainer):
class EnhancerTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(EnhancerTrainer, self).__init__(cfg, model, dataset_list, metrics_list)
        self.image_editor = ImageEditor()

    def init_loss_func(self):
        self.loss_func_mse = torch.nn.MSELoss()
        self.loss_func_emd = emd_loss

    def epoch_forward(self, epoch, isTrain):
        if isTrain:
            loader = self.train_loader
            self.model.train()
        else:
            loader = self.val_loader
            self.model.eval()
        
        _emd_loss = AverageMeter()
        _mse_loss = AverageMeter()
        _loss = AverageMeter()
        _plcc_mean = AverageMeter()
        _srcc_mean = AverageMeter()
        _plcc_std = AverageMeter()
        _srcc_std = AverageMeter()
        _acc = AverageMeter()
        for epoch_step, data in enumerate(loader):
            original_images = data["image"].to(self.device)
            distri_label = data["annotations"].squeeze(dim=2).to(self.device)
            bin_label = data["bin_cls"]
            batchSize = original_images.shape[0]

            edited_images, filter_channels = self.image_editor(original_images)

            filter_channels = filter_channels.expand(batchSize, 5, 224, 224).to(self.device)
            edited_images = edited_images.to(self.device)
            inputs = torch.cat((original_images, filter_channels), dim=1)

            if isTrain:
                self.optimizer.zero_grad()
            enhanced_image, aes_output = self.model(inputs)
            emd_loss = self.loss_func_emd(aes_output, distri_label)
            mse_loss = self.loss_func_mse(enhanced_image, edited_images)
            loss = self.cfg.emd_gamma * emd_loss + self.cfg.mse_gamma * mse_loss
            if isTrain:
                loss.backward()
            
            _emd_loss.update(emd_loss.item())
            _mse_loss.update(mse_loss.item())
            _loss.update(loss.item())

            # calculate the cc of mean score
            pscore_np = get_score(aes_output, self.device).cpu().detach().numpy()
            tscore_np = get_score(distri_label, self.device).cpu().detach().numpy()

            plcc_mean = pearsonr(pscore_np, tscore_np)[0]
            srcc_mean = spearmanr(pscore_np, tscore_np)[0]

            # calculate the cc of std.dev
            pstd_np = torch.std(aes_output, dim=1).cpu().detach().numpy()
            tstd_np = torch.std(distri_label, dim=1).cpu().detach().numpy()

            plcc_std = pearsonr(pstd_np, tstd_np)[0]
            srcc_std = spearmanr(pstd_np, tstd_np)[0]

            _plcc_mean.update(plcc_mean)
            _srcc_mean.update(srcc_mean)
            _plcc_std.update(plcc_std)
            _srcc_std.update(srcc_std)


            # calculate the classification result of emd
            emd_class_pred = torch.zeros((batchSize), dtype=float)
            for idx in range(batchSize):
                if pscore_np[idx] < 5:
                    emd_class_pred[idx] = 0.0
                elif pscore_np[idx] >= 5:
                    emd_class_pred[idx] = 1.0

            acc = metrics.accuracy_score(bin_label, emd_class_pred)
            _acc.update(acc)
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info(
                    "step: {} | loss: {:.4f}, emd: {:.2f}, mse: {:.2f}, cc: {:.4f}, {:.4f}, {:.4f}, {:.4f}, acc: {:.2f}"
                .format(epoch_step, _loss.avg, _emd_loss.avg, _mse_loss.avg, 
                _plcc_mean.avg, _srcc_mean.avg, _plcc_std.avg, _srcc_std.avg,
                _acc.avg))

            # break
        metrics_result = {}
        for metric in self.metrics_list:
            metrics_result[metric] = eval('_' + metric).avg

        return metrics_result

