import torch
from SongUtils.MetricUtils import AverageMeter, accuracy
from SongUtils.MLUtils.BaseTrainers import BaseTrainer, BaseDistTrainer, init_dist
from SongUtils.MLUtils.BaseArgs import get_dist_base_parser

from utils.edit_transform import ImageEditor

# class CANTrainer(BaseDistTrainer):
class BaseEditorTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(BaseEditorTrainer, self).__init__(cfg, model, dataset_list, metrics_list)
        self.image_editor = ImageEditor()

    def init_loss_func(self):
        self.loss_func = torch.nn.MSELoss()

    def epoch_forward(self, epoch, isTrain):
        # implement in children class for channels or tokens
        pass

class EditorTrainer_Tokens(BaseEditorTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(EditorTrainer_Tokens, self).__init__(cfg, model, dataset_list, metrics_list)

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

class EditorTrainer_Channels(BaseEditorTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(EditorTrainer_Channels, self).__init__(cfg, model, dataset_list, metrics_list)

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

            edited_images, filter_channels = self.image_editor(original_images, (224, 224), single_filter=True, polar_intensity=True)
            filter_channels = filter_channels.expand(batchSize, 5, 224, 224).to(self.device)
            edited_images = edited_images.to(self.device).to(self.device)
            inputs = torch.cat((original_images, filter_channels), dim=1)

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


class EditorTrainer_midChannels(BaseEditorTrainer):
    def __init__(self, cfg, model, dataset_list, metrics_list):
        super(EditorTrainer_midChannels, self).__init__(cfg, model, dataset_list, metrics_list)

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

            edited_images, filter_channels = self.image_editor(original_images, (14, 14), single_filter=True, polar_intensity=True)
            filter_channels = filter_channels.expand(batchSize, 5, 14, 14).to(self.device)
            edited_images = edited_images.to(self.device).to(self.device)

            if isTrain:
                self.optimizer.zero_grad()
                output = self.model(original_images, filter_channels)
                loss = self.loss_func(output, edited_images)
                loss.backward()
                self.optimizer.step()

            else:
                output = self.model(original_images, filter_channels)
                loss = self.loss_func(output, edited_images)

            _loss.update(loss.item())
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info("Step: {}, loss: {:.4f}".format(epoch_step, _loss.avg))
        return {'loss': _loss.avg}