import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import os.path as osp
from PIL import Image

class EnhanceDataset(Dataset):
    def __init__(self, target_image_folder, original_image_folder, model_type="cnn", transform=None):
        self.intensity_map = {
            "-100": 0,
            "-50": 1,
            "0": 2,
            "50": 3,
            "100": 4
        }
        self.intensity2value_map = {
            '0': -1.,
            '1': -0.5,
            '2': 0.,
            '3': 0.5,
            '4': 1.
        }
        self.method_idx_map = {       
            "brightness": 0,
            "contrast": 1,
            "saturation": 2,
            "?": 3,
            "??": 4
        }
        # for example: 125_03221.jpg means: 
        #   brightness: -100,
        #   constrast: 50,
        #   saturation: 0,
        #   ?: 0,
        #   ??: -50
        self.target_image_folder = target_image_folder
        self.original_image_folder = original_image_folder
        self.target_image_list = os.listdir(self.target_image_folder)
        self.model_type = model_type
        self.transform = transform
    
    def __len__(self):
        return len(self.target_image_list)
    
    def __getitem__(self, idx):
        image_name = self.target_image_list[idx]
        target_img = Image.open(osp.join(self.target_image_folder, image_name)).convert("RGB")

        h, w = target_img.size
        image_prefix = image_name.split('.')[0]
        original_image_prefix, intensity_str = image_prefix.split('_')  # for example: 03221 
        original_img = Image.open(osp.join(self.original_image_folder, original_image_prefix + '.jpg')).convert("RGB")

        if self.transform is not None:
            target_image_tensor = self.transform(target_img)
            original_image_tensor = self.transform(original_img)
            h, w = 224, 224
        else:
            target_image_tensor = transforms.functional.to_tensor(target_img)
            original_image_tensor = transforms.functional.to_tensor(original_img)
        
        assert len(intensity_str) == 5
        if self.model_type == "cnn":
            filter_channels = None
            for i, intensity in enumerate(intensity_str):
                if i == 3: break
                filter_channel = torch.ones(1, w, h) * self.intensity2value_map[intensity]
                if filter_channels is None:
                    filter_channels = filter_channel
                else:
                    filter_channels = torch.cat((filter_channels, filter_channel), dim=0)
                # original_image_tensor = torch.cat((original_image_tensor, intensity_channel), dim=0)

            return original_image_tensor, target_image_tensor, filter_channels

        elif self.model_type == "vit":
            filter_tokens = None
            for i, intensity in enumerate(intensity_str):
                if i == 3: break 
                filter_token = torch.ones(1, 768) * self.intensity2value_map[intensity]
                if filter_tokens is None:
                    filter_tokens = filter_token
                else:
                    filter_tokens = torch.cat((filter_tokens, filter_token), dim=0)
        else:
            print(f"Unknown model_type: {self.model_type}")
        return original_image_tensor, target_image_tensor, filter_tokens