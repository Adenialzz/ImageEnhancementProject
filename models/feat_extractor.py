import torch
import timm

class FeatExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatExtractor, self).__init__()
        self.base_model = timm.create_model('resnet34', pretrained=True, features_only=True, out_indices=[4, ])
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        feat_map = self.base_model(x)[0]
        feat = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1)).squeeze()
        fear = self.sigmoid(feat)
        return feat
