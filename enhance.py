import torch
import numpy as np
from SongUtils.MiscUtils import load_weights
from models.nicer_models import CAN
import timm

class Enhancer:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.filters = torch.zeros((8, 224, 224), dtype=np.float32, requires_grad=True).to(self.device)
        self.optimizer = torch.optim.Adam(params=[self.filters], lr=0.1)
        
        pass
    
    def init_editor(self):
        editor = CAN()
        self.editor = load_weights(editor, 'can_weights.pth')
        self.editor.eval()
    
    def init_assessor(self):
        assessor = timm.create_model('vit_base_patch16_224_in21k', num_classes=10)
        self.assessor = load_weights(assessor, 'nima/vit_weights/model_10.pth')
        self.assessor.eval()
    
    def run(self, image):
        filter_tensors = torch.zeros((8, 224, 224), dtype=np.float32).to(self.device)
        return  filter_tensors


def main():

    for i in range(8):
        pass
    inputs_image = torch.cat(img)

if __name__ == "__main__":
    main()
