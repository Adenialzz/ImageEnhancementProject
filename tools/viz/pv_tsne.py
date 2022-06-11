import torch
import torchvision.transforms as transforms
import os
import os.path as osp
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm
import random

import sys
sys.path.append(os.getcwd())
from SongUtils.MiscUtils import setup_seed
from models.feat_extractor import FeatExtractor, FeatExtractor_ViT

from run_tsne import run_tsne

def get_k(path):
    ckpt = torch.load(path, map_location='cpu')
    pv = ckpt['preference_vector']
    return pv

def load_feat_extractor(weights_path, model):
    ckpt = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='model_dir')
    parser.add_argument('--epoch', default=30, help='which epoch of results to load')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--experts', default=None)
    parser.add_argument('--out', default='pv_tsne.png')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--arch', default='resnet34')
    cfg = parser.parse_args()
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    setup_seed(46)
    data_root = '/ssd1t/song/Datasets/FiveK/'
    pv = get_k(osp.join(cfg.model_dir, f'pv_k_{cfg.epoch}.pth'))
    if cfg.experts is None:
        cfg.experts = list(pv.keys())

    pipeline = transforms.Compose( [transforms.Resize((224, 224)), transforms.ToTensor()] )
    image_list_all = os.listdir(osp.join(data_root, 'I'))
    feats_list = []

    for e in cfg.experts:
        random.shuffle(image_list_all)
        image_list = image_list_all[: cfg.n_samples]
        base_expert_dir = osp.join(data_root, e)
        # model = FeatExtractor(arch=cfg.arch, pretrained=False)
        model = FeatExtractor_ViT(arch='vit_tiny_patch16_224', pretrained=False)
        model = load_feat_extractor(osp.join(cfg.model_dir, f"model_{cfg.epoch}.pth"), model)
        model = model.to(cfg.device)
        for image in tqdm(image_list):
            image_path = osp.join(base_expert_dir, image)
            img = Image.open(image_path).convert('RGB')
            img = pipeline(img).unsqueeze(dim=0)
            img = img.to(cfg.device)
            feat = model(img)
            feat = feat.squeeze().cpu().detach().numpy()
            feats_list.append(feat)
    image_feats = np.array(feats_list)
    print(image_feats.shape)
    
    expert_feats = np.zeros((0, 512))
    for e, feat in pv.items():
        if e in cfg.experts:
            expert_feats = np.concatenate((expert_feats, feat.detach().numpy()), axis=0)
    simi = np.dot(expert_feats[0], expert_feats[1])
    print(simi)
    feats = np.concatenate((image_feats, expert_feats), axis=0)
    print(f"Num of Samples: {image_feats.shape[0]}, Num of Experts: {expert_feats.shape[0]}")
    run_tsne(feats, cfg.n_samples, cfg.experts, cfg.out)
