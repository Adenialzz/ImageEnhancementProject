import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
import os.path as osp
import argparse
import random
from tqdm import tqdm
from PIL import Image
import numpy as np

from run_tsne import run_tsne

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('job_name', help='job name')
    parser.add_argument('--epoch', default=16, help='which epoch of results to load')
    parser.add_argument('--n_classes', default=9)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--experts', default=None)
    parser.add_argument('--out', default='cls_tsne.jpg')
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--arch', default='resnet34')
    cfg = parser.parse_args()
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    # setup_seed(46)
    data_root = '/ssd1t/song/Datasets/FiveK/'

    pipeline = transforms.Compose( [transforms.Resize((224, 224)), transforms.ToTensor()] )
    image_list_all = os.listdir(osp.join(data_root, 'I'))
    feats_list = []

    model = models.resnet18(num_classes=cfg.n_classes, pretrained=False)
    ckpt = torch.load(osp.join(cfg.base_dir, cfg.job_name, f"model_{cfg.epoch}.pth"), map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(cfg.device)

    cfg.experts = "IABCDEFGH"

    for e in cfg.experts:
        random.shuffle(image_list_all)
        image_list = image_list_all[: cfg.n_samples]
        base_expert_dir = osp.join(data_root, e)
        for image in tqdm(image_list):
            image_path = osp.join(base_expert_dir, image)
            img = Image.open(image_path).convert('RGB')
            img = pipeline(img).unsqueeze(dim=0)
            img = img.to(cfg.device)
            feat = model(img).squeeze()
            feat = feat.cpu().detach().numpy()
            feats_list.append(feat)
    image_feats = np.array(feats_list)
    print(image_feats.shape)
    run_tsne(image_feats, cfg.n_samples, cfg.experts, cfg.out)
    