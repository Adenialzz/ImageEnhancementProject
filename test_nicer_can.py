import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from models.nicer_models import CAN
from utils.utils import load_weights, save_tensor_image
mse_loss_func = torch.nn.MSELoss()

def test_can():
    num_filters = 5
    can = CAN(num_filters) 
    # weights_path = 'weights/nicer/can8_epoch10_final.pt'
    weights_path = '/media/song/ImageEnhancingResults/model/can_cnn/model_3.pth'
    can = load_weights(can, weights_path)

    # strings = ['Sat', 'Con', 'Bri', 'Sha', 'Hig', 'LLF', 'NLD', 'EXP']
    filter_intensities = [0., 0., 0., 0., 0., 0., 0., 0.][: num_filters]
    filter_channels = None
    for inten in filter_intensities:
        channel = (torch.ones(1, 1, 224, 224) * inten)
        # print(filter_channels.shape, channel.shape)
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
    print(inputs.shape)

    outputs = can(inputs).detach()
    outputs = outputs.squeeze(dim=0)
    print(mse_loss_func(tensor_img, outputs).item())
    save_tensor_image(outputs, "imgs/nicer_edited.jpg")


if __name__ == "__main__":
    test_can()