import torch
import torch.nn as nn
import torch.nn.functional as F

class PV_Editor(nn.Module):
    def __init__(self):
        super(PV_Editor, self).__init__()
        in_count = 3
        pv_fmap_channel = 4
        self.conv1 = nn.Conv2d(in_channels=in_count, out_channels=24, kernel_size=(3, 3), dilation=1,
                               padding=(1, 1))  # weight shape [24,8,3,3]
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=2,
                               padding=(2, 2))  # weight shape [24,24,3,3]
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=4,
                               padding=(4, 4))  # weight shape     ""
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=8,
                               padding=(8, 8))  # weight shape     ""

        self.conv5 = nn.Conv2d(in_channels=24+pv_fmap_channel, out_channels=24+pv_fmap_channel, kernel_size=(3, 3), dilation=16,
                               padding=(16, 16))  # weight shape     ""
        self.conv6 = nn.Conv2d(in_channels=24+pv_fmap_channel, out_channels=24+pv_fmap_channel, kernel_size=(3, 3), dilation=32,
                               padding=(32, 32))  # weight shape     ""
        self.conv7 = nn.Conv2d(in_channels=24+pv_fmap_channel, out_channels=24+pv_fmap_channel, kernel_size=(3, 3), dilation=64,
                               padding=(64, 64))  # weight shape     ""
        self.conv9 = nn.Conv2d(in_channels=24+pv_fmap_channel, out_channels=24+pv_fmap_channel, kernel_size=(3, 3), dilation=1,
                               padding=(1, 1))  # weight shape     ""
        self.conv10 = nn.Conv2d(in_channels=24+pv_fmap_channel, out_channels=3, kernel_size=(1, 1),
                                dilation=1)  # weight shape [3,24,1,1]
        # self.pv_conv = nn.Conv2d(512, out_channels=pv_fmap_channel, kernel_size=(1, 1), dilation=8) # 512, 1, 1 ---> 4, 1, 1
        self.pv_conv = nn.Conv2d(512, out_channels=pv_fmap_channel, kernel_size=(1, 1)) # 512, 1, 1 ---> 4, 1, 1
        self.pv_upsample = nn.Upsample(scale_factor=224)   # 4, 1, 1 ---> 4, 32, 32

    def forward(self, x, pv):
        inshape = x.shape
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)

        pv = self.pv_conv(pv)
        pv = self.pv_upsample(pv)
        pv = F.leaky_relu(pv, negative_slope=0.2)

        x = torch.cat((x, pv), dim=1)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv6(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv7(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv9(x), negative_slope=0.2)
        x = self.conv10(x)  # no activation in last layer

        return x

if __name__ == "__main__":
    import torch
    model = PV_Editor()
    bs = 4
    inputs = torch.ones(bs, 3, 224, 224)
    pv = torch.randn(bs, 512).unsqueeze(dim=2).unsqueeze(dim=3)
    outputs = model(inputs, pv)
    print(outputs.shape)
