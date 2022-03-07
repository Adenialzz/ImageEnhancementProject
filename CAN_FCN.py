
   
import torchvision
import numpy as np
import torch
from torch import nn
import PIL

norm=nn.BatchNorm2d
act_fn=lambda inplace:nn.LeakyReLU(negative_slope=0.2,inplace=inplace)
criterion=nn.MSELoss()

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,dilation=1):
        padding = dilation * (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,dilation=dilation, groups=groups, bias=False),
            norm(out_planes),
            act_fn(inplace=True)
        )

class CAN(nn.Module):
    def __init__(self,d=9,w=24):
        super().__init__()
        l=[]
        in_planes=3
        dilate=1
        for i in range(0,d-2):
            l.append(ConvBNReLU(in_planes,w,dilation=dilate))
            in_planes=w
            dilate*=2
        l.append(ConvBNReLU(w,w))
        l.append(nn.Conv2d(w,3,1))
        self.conv=nn.Sequential(*l)

    def forward(self,x):
        return self.conv(x)

if __name__ =='__main__':
    model=CAN(d=10,w=32)
    image_path = "/ssd1t/song/Datasets/AVA/shortEdge256/125.jpg"
    from SongUtils import ImgUtils as imutils
    inputs = imutils.readTensorImage(image_path, expand=True)
    print(inputs.shape)
    outputs = model(inputs)
    print(outputs.shape)
