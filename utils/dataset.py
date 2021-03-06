import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import os.path as osp
import random
import pandas as pd
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

class AVADataset(Dataset):
    """AVA dataset

    Args:
        # csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        csv_file: a 12-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings, column 12 contains the bin_cls infomation
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        annotations = self.annotations.iloc[idx, 1:11].to_numpy()
        annotations = annotations.astype('float').reshape(-1, 1)
        tri_cls = int(self.annotations.iloc[idx, 11])
        bin_cls = int(self.annotations.iloc[idx, 12])
        sample = {'img_name': img_name.split('/')[-1], 'image': image, 'annotations': annotations, 'tri_cls': tri_cls, 'bin_cls': bin_cls}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class ColorizationDataset(Dataset):
    """AVA dataset

    Args:
        # csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        csv_file: a 12-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings, column 12 contains the bin_cls infomation
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.gray_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        sample = {'gray_image': self.gray_transform(image), 'original_image': self.transform(image)}
        return sample

class FiveKDataset(Dataset):
    def __init__(self, root_dir, image_name_list, same_image, transform=None, experts=['A', 'B', 'C', 'D', 'E']):
        self.root_dir = root_dir
        self.experts = experts
        self.same_image = same_image
        self.transform = transform
        self.image_name_list = image_name_list
    
    def __len__(self):
        return len(self.image_name_list)
    
    def get_random_idx(self, num, max=4999):
        random_list = [random.randint(0, max) for _ in range(num)]
        return random_list

    def __getitem__(self, idx):
        if self.same_image:
            idx_list = [idx for _ in range(len(self.experts))]
        else:
            idx_list = [idx]
            idx_list += self.get_random_idx(len(self.experts) - 1, max=len(self.image_name_list) - 1)

        paths = [osp.join(self.root_dir, e, self.image_name_list[idx]) for e, idx in zip(self.experts, idx_list)]

        data = {}
        for e, path in zip(self.experts, paths):
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            data[e] = img

        return data

class FiveKDataset4PV_Editor(Dataset):
    def __init__(self, root_dir,target_experts, image_name_list, input_expert='I', transform=None):
        assert input_expert not in target_experts
        self.input_expert = input_expert
        self.root_dir = root_dir
        self.image_name_list = image_name_list
        self.transform = transform
        self.target_experts = target_experts

    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self, idx):
        target_images = {}
        for e in self.target_experts:
            imagepath = osp.join(self.root_dir, e, self.image_name_list[idx])
            img = Image.open(imagepath).convert('RGB')
            if self.transform is not None:
                tensor_img = self.transform(img)
            else:
                tensor_img = transforms.ToTensor()(img)
            target_images[e] = tensor_img
        input_image_path = osp.join(self.root_dir, self.input_expert, self.image_name_list[idx])
        img = Image.open(input_image_path).convert('RGB')
        if self.transform is not None:
            input_image = self.transform(img)
        else:
            input_image = transforms.ToTensor()(img)
        data = {
            'input_image': input_image,
            'target_images': target_images
        }
        return data


     
if __name__ == "__main__":
    data_root = '/ssd1t/song/Datasets/FiveK'
    val_image_name_list = os.listdir(osp.join(data_root, 'inputs'))[4500: ]
    pipeline = transforms.Compose( [transforms.ToTensor()] )
    dataset = FiveKDataset(data_root, val_image_name_list, same_image=False, transform=pipeline)
    dataloader = DataLoader(dataset, batch_size=1)
    for data in dataloader:
        print('*'*42)
        for k, v in data.items():
            print(k, v.shape)










