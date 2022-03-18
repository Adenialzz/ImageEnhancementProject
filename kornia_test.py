import torch
import kornia.enhance as ke
import cv2
from SongUtils import IOUtils as iout


inputs = iout.readTensorImage("/ssd1t/song/Datasets/AVA/shortEdge256/125.jpg", through='opencv')
# outputs = ke.adjust_brightness(inputs, 0.8)   # [-0.5, 0.5], origin: 0.
# outputs = ke.adjust_contrast(inputs, 1.)      # [0.2, 1.8], 1.
# outputs = ke.adjust_hue(inputs, -2.)            # [-3., 3.], 0.
# outputs = ke.adjust_saturation(inputs, 0.)      # [0., 5.]. 1.
# outputs = ke.adjust_gamma(inputs, 1.8)             # [0.2, 1.8], 1.

img = iout.tensor2cv(outputs)
print(img.shape)
cv2.imwrite('edited.jpg', img[:,:,::-1])

