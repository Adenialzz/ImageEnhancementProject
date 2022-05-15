import torch
import cv2
from utils.utils import save_tensor_image, load_tensor_image, kornia_edit

image_path = '/ssd1t/song/Datasets/FiveK/C/a0001.jpg'
out_path = "imgs/kornia_test.jpg"
method = "contrast"
inten = 0.2

inputs = load_tensor_image(image_path)
outputs = kornia_edit(inputs, method, inten)

save_tensor_image(outputs, out_path)

