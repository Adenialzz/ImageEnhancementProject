import torch
from torch.utils.data import DataLoader
from vit import VisionTransformer
from sit import VisionTransformer_SiT
from SongUtils.ImgUtils import *
import shutil
import os
import os.path as osp

base_dir = "/ssd1t/song/Datasets/AVA/filters/test1w/origin1w/"
target_dir = "/ssd1t/song/Datasets/AVA/filters/test1w/target_images/"
images_list = os.listdir(target_dir)

for image in images_list:
    if '_' not in image:
        os.remove(osp.join(target_dir, image))