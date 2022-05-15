import torch
import cv2
import numpy as np
import kornia.enhance as ke

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

def load_timm_weights(model, weights_path):
    model_weights = torch.load(weights_path, map_location='cpu')
    new_model_weights = {}
    for name, weight in model_weights.items():
        if 'head' not in name:
            new_model_weights[name] = weight

    msg = model.load_state_dict(new_model_weights, strict=False)
    print(msg)
    return model


def kornia_edit(input_tensor_image, method, inten):
    func = eval(f"ke.adjust_{method}")
    outputs = func(input_tensor_image, inten)
    # [-0.5, 0.5], origin: 0.
    # outputs = ke.adjust_contrast(inputs, inten)      # [0.2, 1.8], 1.
    # outputs = ke.adjust_hue(inputs, -2.)            # [-3., 3.], 0.
    # outputs = ke.adjust_saturation(inputs, 10.)      # [0., 5.]. 1.
    # outputs = ke.adjust_gamma(inputs, 1.8)             # [0.2, 1.8], 1.

    return outputs



def save_tensor_image(tensor, save_path):
    img = tensor.permute(1, 2, 0).numpy()[:, :, ::-1] * 255.
    cv2.imwrite(save_path, img)

def load_tensor_image(image_path):
    array = cv2.imread(image_path)[:, :, ::-1].transpose(2, 0, 1) / 255.
    tensor = torch.from_numpy(array)
    return tensor

def _load_tensor_imag_pil(image_path):
    pipeline = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    tensor = pipeline(Image.open(image_path).convert('RGB'))
    return tensor