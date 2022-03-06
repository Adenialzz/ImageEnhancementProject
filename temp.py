import torch
from musiq import MUSIQ
import os
import os.path as osp
from tqdm import tqdm
from SongUtils import ImgUtils 

import warnings
warnings.filterwarnings('ignore')

def forward_all_ava(model, output_file):
    image_dir = "/ssd1t/song/Datasets/AVA/shortEdge256"
    image_list = os.listdir(image_dir)

    fd = open(output_file, 'w')
    model.eval()
    for image in image_list:
        image_path = osp.join(image_dir, image)
        inputs = ImgUtils.readTensorImage(image_path).cuda()
        with torch.no_grad():
            outputs = model(inputs)
        fd.write(f"{image},{outputs.item()}\n")

if __name__ == "__main__":
    model = MUSIQ(num_class=10, pretrained=False)
    image_path = "/ssd1t/song/Datasets/AVA/shortEdge256/125.jpg"
    inputs = ImgUtils.readTensorImage(image_path)
    print(inputs.shape)
    model(inputs)

    # forward_all_ava(model, "musiq.txt")


