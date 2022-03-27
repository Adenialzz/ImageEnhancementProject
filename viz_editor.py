import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import SongUtils.ImgUtils as imu
from utils import *
from models.CAN_FCN import CAN
# from nicer_models import CAN
from models.sit import VisionTransformer_SiT
from utils.edit_transform import ImageEditor
from utils.utils import load_weights

pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_path = './imgs/125.jpg'

original = Image.open(image_path).convert('RGB')
original = pipeline(original)
intensity_str = "22222"
image_editor = ImageEditor()
factors = [int(i) for i in intensity_str]
target_edited_image, filter_channels = image_editor(original, factors=factors)

print(f"Number of filters: {len(intensity_str)}")


inputs = torch.cat((original.unsqueeze(dim=0), filter_channels.unsqueeze(dim=0)), dim=1)

# model = CAN(in_planes=5, d=10, w=32)
# model = CAN(no_of_filters=len(intensity_str))
model = VisionTransformer_SiT(in_chans=len(intensity_str) + 3, num_filters=0)
model = load_weights(model, "/media/song/ImageEnhancingResults/model/can_vit/model_18.pth")

print(inputs.shape)
outputs = model(inputs)

outputs += original
# print(outputs)

edited_img = outputs.squeeze(dim=0).detach().permute(1, 2, 0).numpy()[:, :, ::-1] * 255.
target_edited_image = target_edited_image.squeeze(dim=0).detach().permute(1, 2, 0).numpy()[:, :, ::-1] * 255.
cv2.imwrite("imgs/vit_edited_0.0.jpg", edited_img)
# cv2.imwrite("imgs/target_edited_image.jpg", target_edited_image)

