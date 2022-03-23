import torch
import kornia.enhance as ke
import cv2
from utils.utils import *


from PIL import Image
import torchvision.transforms as transforms
pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
inten = -0.5
inputs = pipeline(Image.open('imgs/125.jpg').convert('RGB'))
outputs = ke.adjust_brightness(inputs, inten)   # [-0.5, 0.5], origin: 0.
# outputs = ke.adjust_contrast(inputs, 1.)      # [0.2, 1.8], 1.
# outputs = ke.adjust_hue(inputs, -2.)            # [-3., 3.], 0.
# outputs = ke.adjust_saturation(inputs, 10.)      # [0., 5.]. 1.
# outputs = ke.adjust_gamma(inputs, 1.8)             # [0.2, 1.8], 1.

save_tensor_image(outputs, f"imgs/kornia_test-{inten}.jpg")

