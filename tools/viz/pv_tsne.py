import torch
import torchvision.transforms as transforms
import os
import os.path as osp

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random

import sys
sys.path.append('/home/ps/JJ_Projects/ImageEnhancementProject')
from SongUtils.MiscUtils import setup_seed

from models.feat_extractor import FeatExtractor

def run_tsne(data, n_examples, n_experts=2):
    # data.shape: bs * dim
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_result = tsne.fit_transform(data)
    color_list = ['r', 'b', 'g', 'orange', 'm', 'gold', 'c', 'lightseagreen', 'k', 'darkorchid']
    experts_list = 'IABCDE'
    for i, coord in enumerate(tsne_result):
        # if i < (n_experts * n_examples - 2)
        if i < n_examples * n_experts:
            idx = i // n_examples
            if i % n_examples == 0:
                plt.scatter(coord[0], coord[1], marker='x', c=color_list[idx], label=experts_list[idx])
            else:
                plt.scatter(coord[0], coord[1], marker='x', c=color_list[idx])
        elif i >= n_examples * n_experts:
            idx += 1
            plt.scatter(coord[0], coord[1], marker='o', c=color_list[idx], label=experts_list[idx-n_experts])
        
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.title(f't-SNE of \'{experts_list[: n_experts]}\' Samples & PreferenceVector')
    plt.savefig('tsne.png')


def get_k(path):
    ckpt = torch.load(path, map_location='cpu')
    pv = ckpt['preference_vector']
    for k, v in pv.items():
        print(k, v.shape)
    return pv

if __name__ == "__main__":
    setup_seed(46)
    base_dir = '/ssd1t/song/Datasets/FiveK/'
    epoch = 99
    experts = 'IAB'
    job_name = f'triplet_{experts}_exp0'
    results_dir = '/media/song/ImageEnhancingResults/weights/train_pv'
    pv = get_k(osp.join(results_dir, job_name, f'pv_k_{epoch}.pth'))

    pipeline = transforms.Compose( [transforms.ToTensor()] )
    image_list_all = os.listdir(osp.join(base_dir, 'I'))
    feats_list = []
    N = 200
    device = "cuda:2"
    for e in experts:
        random.shuffle(image_list_all)
        image_list = image_list_all[: N]
        base_expert_dir = osp.join(base_dir, e)
        model = FeatExtractor()
        model.load_state_dict(torch.load(osp.join(results_dir, job_name, f'model_{epoch}.pth'))['state_dict'])
        model = model.to(device)
        # feat_list = []
        for image in tqdm(image_list):
            image_path = osp.join(base_expert_dir, image)
            img = Image.open(image_path).convert('RGB')
            img = pipeline(img).unsqueeze(dim=0)
            img = img.to(device)
            feat = model(img)
            feat = feat.cpu().detach().numpy()
            feats_list.append(feat)
    np_feats_list = np.array(feats_list)
    for k, v in pv.items():
        np_feats_list = np.concatenate((np_feats_list, v.detach().numpy()), axis=0)
    print(np_feats_list.shape)
    run_tsne(np_feats_list, N, len(experts))
