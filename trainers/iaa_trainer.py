import torch
import torchvision
import torchvision.transforms as transforms
import timm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from SongUtils.MLUtils.BaseTrainers import BaseTrainer
from SongUtils.MLUtils.BaseArgs import get_base_parser
from SongUtils.MetricUtils import AverageMeter
from sklearn import metrics
from utils.utils import get_score
from utils.loss import emd_loss
from utils.dataset import AVADataset
from scipy.stats import pearsonr
from scipy.stats import spearmanr

class IAATrainer(BaseTrainer):
    def __init__(self, cfg, model, dataloader_list, metrics_list):
        super(IAATrainer, self).__init__(cfg, model, dataloader_list, metrics_list)
        self.softmax = torch.nn.Softmax(dim=1)

    def init_loss_func(self):
        self.loss_func = emd_loss
    
    def epoch_forward(self, isTrain, epoch):
        if isTrain:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader
        
        _loss = AverageMeter()
        _acc = AverageMeter()
        _plcc_mean = AverageMeter()
        _srcc_mean = AverageMeter()
        _plcc_std = AverageMeter()
        _srcc_std = AverageMeter()
        for epoch_step, data in enumerate(loader):
            image = data["image"].to(self.device)
            label = data["annotations"].to(self.device).squeeze(dim=2)
            bin_label = data["bin_cls"]
            batch_size = image.shape[0]

            if isTrain:
                self.optimizer.zero_grad()
            output = self.model(image)
            # output = self.softmax(output)

            loss = self.loss_func(output, label)
            if isTrain:
                loss.backward()
                self.optimizer.step()
            
            # calculate the cc of mean score
            pscore_np = get_score(output, self.device).cpu().detach().numpy()
            tscore_np = get_score(label, self.device).cpu().detach().numpy()

            plcc_mean = pearsonr(pscore_np, tscore_np)[0]
            srcc_mean = spearmanr(pscore_np, tscore_np)[0]

            # calculate the cc of std.dev
            pstd_np = torch.std(output, dim=1).cpu().detach().numpy()
            tstd_np = torch.std(label, dim=1).cpu().detach().numpy()

            plcc_std = pearsonr(pstd_np, tstd_np)[0]
            srcc_std = spearmanr(pstd_np, tstd_np)[0]

            _loss.update(loss.item())
            _plcc_mean.update(plcc_mean)
            _srcc_mean.update(srcc_mean)
            _plcc_std.update(plcc_std)
            _srcc_std.update(srcc_std)


            # calculate the classification result of emd
            emd_class_pred = torch.zeros((batch_size), dtype=float)
            for idx in range(batch_size):
                if pscore_np[idx] < 5:
                    emd_class_pred[idx] = 0.0
                elif pscore_np[idx] >= 5:
                    emd_class_pred[idx] = 1.0

            acc = metrics.accuracy_score(bin_label, emd_class_pred)
            _acc.update(acc)
            if epoch_step % self.cfg.log_freq == 0:
                self.logger.info("step: {} | loss: {:.4f}, acc: {:.2f}, cc: {:.4f}, {:.4f}, {:.4f}, {:.4f}"
                .format(epoch_step, _loss.avg, _acc.avg, _plcc_mean.avg, _srcc_mean.avg, _plcc_std.avg, _srcc_std.avg))

            # break
        metrics_result = {}
        for metric in self.metrics_list:
            metrics_result[metric] = eval('_' + metric).avg

        return metrics_result

def get_args():
    parser = get_base_parser()
    parser.add_argument("--timm-model", type=str, default='resnet50')
    cfg = parser.parse_args()
    return cfg

def main(cfg):
    pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_set = AVADataset(csv_file="/home/song/JJ_Projects/dsmAVA/csvFiles/train_mlsp.csv", root_dir="/home/song/AVA/shortEdge256", transform=pipeline)
    val_set = AVADataset(csv_file="/home/song/JJ_Projects/dsmAVA/csvFiles/val_mlsp.csv", root_dir="/home/song/AVA/shortEdge256", transform=pipeline)
    train_loader = DataLoader(train_set, batch_size=cfg.batchSize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batchSize, shuffle=False)

    model = timm.create_model(cfg.timm_model, pretrained=True, num_classes=10)
    metrics_list = ["loss", "acc", "plcc_mean", "srcc_mean", "plcc_std", "srcc_std"]
    trainer = NIMATrainer(cfg, model, [train_loader, val_loader], metrics_list)
    trainer.forward()


if __name__ == "__main__":
    cfg = get_args()
    main(cfg)
