import torch
import cv2
import numpy as np

def get_filter_tokens(intensity_str, model_type, size=224):
    intensity2value_map = {
        '0': -1.,
        '1': -0.5,
        '2': 0.,
        '3': 0.5,
        '4': 1.
    }
    # intensity2value_map = { str(i): -100 + i * 20 for i in range(10) }
    if model_type == "cnn":
        filter_channels = None
        for i, intensity in enumerate(intensity_str):
            # if i == 3: break
            filter_channel = torch.ones(1, size, size) * intensity2value_map[intensity]
            if filter_channels is None:
                filter_channels = filter_channel
            else:
                filter_channels = torch.cat((filter_channels, filter_channel), dim=0)
        return filter_channels

    elif model_type == "vit":
        filter_tokens = None
        for i, intensity in enumerate(intensity_str):
            if i == 3: break 
            filter_token = torch.ones(1, size) * intensity2value_map[intensity]
            if filter_tokens is None:
                filter_tokens = filter_token
            else:
                filter_tokens = torch.cat((filter_tokens, filter_token), dim=0)

    else:
        print(f"Unknown model_type: {self.model_type}")
    return filter_tokens

def load_weights(model, weights_path):
    ckpt = torch.load(weights_path, map_location='cpu')
    model_weights = ckpt['state_dict']
    model.load_state_dict(model_weights)
    return model

def get_score(score_distri, device):
    '''
    score_distri shape:  batch_size * 10
    '''
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor).to(device)
    w_batch = w.repeat(score_distri.size(0), 1)

    score = (score_distri * w_batch).sum(dim=1)

    return score

def load_weights_resize_pos_embed(model, weights_path):
    ckpt = torch.load(weights_path, map_location='cpu')
    model_weights = ckpt["state_dict"]
    add = torch.ones(1, 1, 768)
    model_weights["pos_embed"] = torch.cat((model_weights["pos_embed"], add), dim=1)
    
    msg = model.load_state_dict(model_weights, strict=False)
    print(msg)
    return model

def save_tensor_image(tensor, path):
    img = tensor.permute(1, 2, 0).numpy()[:, :, ::-1] * 255.
    cv2.imwrite(path, img)