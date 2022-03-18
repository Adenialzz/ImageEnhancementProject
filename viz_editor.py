import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import SongUtils.ImgUtils as imu
from utils import *
from CAN_FCN import CAN
# from nicer_models import CAN
from sit import VisionTransformer_SiT

pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_path = './imgs/125.jpg'

original = imu.readTensorImage(image_path, transform=pipeline).unsqueeze(dim=0)
intensity_str = "224"
print(f"Number of filters: {len(intensity_str)}")

print(original.shape)
_125 = imu.tensor2cv(original.squeeze(dim=0))[:, :, ::-1]
cv2.imwrite("125_224.jpg", _125)

filter_channels = get_filter_tokens(intensity_str, 'cnn', 224).unsqueeze(dim=0)
# print(filter_channels)

inputs = torch.cat((original, filter_channels), dim=1)

# model = CAN(in_planes=5, d=10, w=32)
model = CAN(no_of_filters=len(intensity_str))
# model = VisionTransformer_SiT(in_chans=6)
model = load_weights(model, "/media/song/IE_Models/can_vit_res_weights/model_2.pth")

outputs = model(inputs)

outputs += original
# print(outputs)

edited_img = outputs.squeeze(dim=0).detach().permute(1, 2, 0).numpy()[:, :, ::-1] * 255.
print(edited_img.shape)
cv2.imwrite("edited_image.jpg", edited_img)

