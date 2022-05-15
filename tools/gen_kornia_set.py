import torch
import kornia.enhance as ke
import cv2
import os
import os.path as osp
import sys
sys.path.append(os.getcwd())
from utils.utils import save_tensor_image, load_tensor_image, kornia_edit
import argparse


# outputs = ke.adjust_brightness(input_tensor_image, inten)   # [-0.5, 0.5], origin: 0.
# outputs = ke.adjust_contrast(inputs, inten)      # [0.2, 1.8], 1.
# outputs = ke.adjust_hue(inputs, -2.)            # [-3., 3.], 0.
# outputs = ke.adjust_saturation(inputs, 10.)      # [0., 5.]. 1.
# outputs = ke.adjust_gamma(inputs, 1.8)             # [0.2, 1.8], 1.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_set', type=str, default='I')
    parser.add_argument('--dst_set', type=str, default='F')
    parser.add_argument('--method', type=str, default='brightness')
    parser.add_argument('--inten', type=float, default=0.5)
    cfg = parser.parse_args()
    return cfg

def main(cfg):
    data_root = '/ssd1t/song/Datasets/FiveK/'

    src_path = osp.join(data_root, cfg.src_set)
    dst_path = osp.join(data_root, cfg.dst_set)
    if not osp.isdir(dst_path):
        os.mkdir(dst_path)
    path = osp.join(data_root, cfg.src_set)

    from tqdm import tqdm
    for image in tqdm(os.listdir(src_path)):
        src_image_path = osp.join(src_path, image)
        dst_image_path = osp.join(dst_path, image)
        inputs = load_tensor_image(src_image_path)
        outputs = kornia_edit(inputs, cfg.method, cfg.inten)
        save_tensor_image(outputs, dst_image_path)

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
