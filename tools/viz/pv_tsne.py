import torch
import torchvision.transforms as transforms
import os
import os.path as osp
import argparse

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from PIL import Image
from tqdm import tqdm
import random

import sys
sys.path.append(os.getcwd())
from SongUtils.MiscUtils import setup_seed
from models.feat_extractor import FeatExtractor


def run_tsne(data, n_examples, experts_list, out_pic_name):
    # data.shape: bs * dim
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_result = tsne.fit_transform(data)
    color_list = list(pltcolors.get_named_colors_mapping().keys())
    random.shuffle(color_list)
    for i, coord in enumerate(tsne_result):
        if i < n_examples * len(experts_list):
            idx = i // n_examples
            if i % n_examples == 0:
                plt.scatter(coord[0], coord[1], marker='x', c=color_list[idx], label=experts_list[idx])
            else:
                plt.scatter(coord[0], coord[1], marker='x', c=color_list[idx])
        elif i >= n_examples * len(experts_list):
            idx += 1
            plt.scatter(coord[0], coord[1], marker='o', c=color_list[idx], label=experts_list[idx-len(experts_list)])
        
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.title(f't-SNE of \'{experts_list}\' Samples & PreferenceVector')
    plt.savefig(out_pic_name)


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
    parser.add_argument('job_name', help='job name')
    parser.add_argument('epoch', help='which epoch of results to load')
    parser.add_argument('--n_samples', default=50)
    parser.add_argument('--experts', default='IABCDE')
    parser.add_argument('--out', default='tsne.png')
    parser.add_argument('--base_dir', default='/media/song/ImageEnhancingResults/weights/train_pv')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--arch', default='resnet18')
    cfg = parser.parse_args()
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    setup_seed(46)
    data_root = '/ssd1t/song/Datasets/FiveK/'
    pv = get_k(osp.join(cfg.base_dir, cfg.job_name, f'pv_k_{cfg.epoch}.pth'))

    pipeline = transforms.Compose( [transforms.Resize((224, 224)), transforms.ToTensor()] )
    image_list_all = os.listdir(osp.join(data_root, 'I'))
    feats_list = []

    for e in cfg.experts:
        random.shuffle(image_list_all)
        image_list = image_list_all[: cfg.n_samples]
        base_expert_dir = osp.join(data_root, e)
        model = FeatExtractor(arch=cfg.arch, pretrained=False)
        model = load_feat_extractor(osp.join(cfg.base_dir, cfg.job_name, f"model_{cfg.epoch}.pth"), model)
        model = model.to(cfg.device)
        for image in tqdm(image_list):
            image_path = osp.join(base_expert_dir, image)
            img = Image.open(image_path).convert('RGB')
            img = pipeline(img).unsqueeze(dim=0)
            img = img.to(cfg.device)
            feat = model(img)
            feat = feat.cpu().detach().numpy()
            feats_list.append(feat)
    image_feats = np.array(feats_list)
    
    expert_feats = np.zeros((0, 512))
    for e, feat in pv.items():
        if e in cfg.experts:
            expert_feats = np.concatenate((expert_feats, feat.detach().numpy()), axis=0)

    feats = np.concatenate((image_feats, expert_feats), axis=0)
    print(f"Num of Samples: {image_feats.shape[0]}, Num of Experts: {expert_feats.shape[0]}")
    run_tsne(feats, cfg.n_samples, cfg.experts, cfg.out)
