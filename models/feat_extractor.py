import torch
import timm

class FeatExtractor(torch.nn.Module):
    def __init__(self, arch, pretrained):
        super(FeatExtractor, self).__init__()
        assert arch in ['resnet18', 'resnet34']
        self.base_model = timm.create_model(arch, pretrained=pretrained, features_only=True, out_indices=[4, ])
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        feat_map = self.base_model(x)[0]
        feat = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1)).squeeze()
        feat = self.sigmoid(feat)
        return feat

class FeatExtractor_ViT(torch.nn.Module):
    def __init__(self, arch, pretrained):
        super(FeatExtractor_ViT, self).__init__()
        assert arch in ['vit_tiny_patch16_224']
        self.base_model = timm.create_model(arch, pretrained=pretrained, num_classes=512)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        feat = self.base_model(x)
        feat = self.sigmoid(feat)
        return feat

class FeatExtractor_FC(torch.nn.Module):
    def __init__(self, arch, pretrained):
        super(FeatExtractor_FC, self).__init__()
        assert arch in ['resnet18', 'resnet34']
        self.base_model = timm.create_model(arch, pretrained=pretrained, features_only=True, out_indices=[4, ])
        self.fc = torch.nn.Linear(512, 128)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        feat_map = self.base_model(x)[0]
        feat = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1)).squeeze()
        feat = self.relu(feat)
        feat = self.fc(feat)
        feat = self.sigmoid(feat)
        return feat
