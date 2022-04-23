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

from models.feat_extractor import FeatExtractor

def run_tsne(data):
    # data.shape: bs * dim
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_result = tsne.fit_transform(data)
    print(tsne_result.shape)
    color_list = ['r', 'b', 'g', 'orange', 'yellow']
    for i, coord in enumerate(tsne_result):
        if i < 50: c = color_list[0]
        elif i > 50 and i < 100: c = color_list[1]
        elif i > 100 and i < 150: c = color_list[2]
        elif i > 150 and i < 200: c = color_list[3]
        elif i > 200 and i < 250: c = color_list[4]
        plt.scatter(coord[0], coord[1], c=c)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('tsne.png')


if __name__ == "__main__":
    base_dir = '/ssd1t/song/Datasets/FiveK/'

    pipeline = transforms.Compose( [transforms.ToTensor()] )
    feats_list = []
    for e in 'ABCDE':
        image_list_all = os.listdir(osp.join(base_dir, 'A'))
        random.shuffle(image_list_all)
        image_list = image_list_all[: 50]
        base_expert_dir = osp.join(base_dir, e)
        model = FeatExtractor()
        model.load_state_dict(torch.load('/media/song/ImageEnhancingResults/weights/train_pv/all_lr1e-4/model_26.pth')['state_dict'])
        # feat_list = []
        for image in tqdm(image_list):
            image_path = osp.join(base_expert_dir, image)
            img = Image.open(image_path).convert('RGB')
            img = pipeline(img).unsqueeze(dim=0)
            feat = model(img).detach().numpy()
            feats_list.append(feat)
    np_feats_list = np.array(feats_list)
    print(np_feats_list.shape)
    run_tsne(np_feats_list)
