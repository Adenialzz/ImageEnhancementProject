import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from models.nicer_models import CAN
from utils.utils import load_weights

def test_can():
    can = CAN(8) 
    can = load_weights(can, 'weights/can8_epoch10_final.pt')

    # strings = ['Sat', 'Con', 'Bri', 'Sha', 'Hig', 'LLF', 'NLD', 'EXP']
    filter_intensities = [0., 0., 0., 0., 0., 0., 0., 0.]
    filter_channels = None
    for inten in filter_intensities:
        channel = (torch.ones(1, 1, 224, 224) * inten)
        if filter_channels is None:
            filter_channels = channel
        else:
            filter_channels = torch.cat((filter_channels, channel), dim=1)
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open("imgs/125.jpg").convert('RGB')
    tensor_img = pipeline(img)
    inputs = torch.cat((tensor_img.unsqueeze(dim=0), filter_channels), dim=1)

    outputs = can(inputs).squeeze(dim=0)
    edited = outputs.detach().permute(1, 2, 0).numpy()[:, :, ::-1]
    edited_clipped = (np.clip(edited, 0., 1.) * 255.).astype(np.uint8)
    cv2.imwrite("imgs/nicer_edited.jpg", edited_clipped)


if __name__ == "__main__":
    test_can()