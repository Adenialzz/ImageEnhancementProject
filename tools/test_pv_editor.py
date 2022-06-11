import torch
import numpy as np
import cv2
import os
import os.path as osp
import sys
sys.path.append(os.getcwd())

from models.pie_cnn import PV_Editor
from utils.utils import save_tensor_image

def read_tensor_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(dim=0).float()
    img.div_(255.)
    return img

class PVEditorViz():
    def __init__(self, editor_model_path, pv_path):
        self.model = PV_Editor()
        ckpt = torch.load(editor_model_path, map_location='cpu')
        self.model.load_state_dict(ckpt['state_dict'])

        self.all_pv = torch.load(pv_path, map_location='cpu')['preference_vector']

    def run(self, image_path, expert, out_pic='pv_editor_viz.jpg'):
        input_image = read_tensor_img(image_path)
        save_tensor_image(input_image.squeeze(), 'input_pv_editor_viz.jpg')
        pv = self.all_pv[expert].unsqueeze(dim=2).unsqueeze(dim=3)
        output = self.model(input_image, pv)
        save_tensor_image(output.squeeze(), out_pic)

if __name__ == '__main__':
    model_path = '/media/song/ImageEnhancingResults/weights/can_editors/can_base_lr1e-1_C_IFGH_pvepoch20/model_49.pth'
    pv_path = '/media/song/ImageEnhancingResults/weights/train_pv/triplet_ICFGH_m0.2_lr1e-2_vit_tiny_patch16_224_rand/pv_k_20.pth'
    vizer = PVEditorViz(model_path, pv_path)

    fivek_root = 'FiveK/'
    input_expert = 'C'
    image_name = sys.argv[1]    # a0001.jpg
    target_expert = sys.argv[2]
    image_path = osp.join(fivek_root, input_expert, image_name)
    vizer.run(image_path, target_expert)
    target_img = read_tensor_img(osp.join(fivek_root, target_expert, image_name))
    save_tensor_image(target_img.squeeze(), 'target_pv_editor_viz.jpg')








