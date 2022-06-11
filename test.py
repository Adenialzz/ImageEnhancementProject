import timm
import torch


model = timm.create_model('vit_tiny_patch16_224', num_classes=512)
inputs = torch.randn(4, 3, 224, 224)
outputs = model(inputs)
print(outputs.shape)
