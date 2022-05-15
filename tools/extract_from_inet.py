import os
import os.path as osp
import shutil
import random
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('target_base_dir', type=str)
    parser.add_argument('n_classes', type=int)
    parser.add_argument('--r2n', action='store_true', help='rename to number')
    parser.add_argument('--inet_base_dir', type=str, default='/ssd1t/song/Datasets/ImageNet/ImageNet_2012_DataSets/ILSVRC2012_img_train/')
    cfg = parser.parse_args()
    return cfg

def main(cfg):
    target_cls_list = os.listdir(cfg.inet_base_dir)
    random.shuffle(target_cls_list)
    if not osp.isdir(cfg.target_base_dir):
        os.mkdir(cfg.target_base_dir)
    for idx, clsdir in tqdm(enumerate(target_cls_list[ :cfg.n_classes]), total=cfg.n_classes):
        originpath = osp.join(cfg.inet_base_dir, clsdir)
        if cfg.rename2number:
            targetpath = osp.join(cfg.target_base_dir, str(idx))
        else:
            targetpath = osp.join(cfg.target_base_dir, clsdir)
        shutil.copytree(originpath, targetpath)
    
if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)