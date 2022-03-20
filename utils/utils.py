import torch
import cv2

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

def save_tensor_image(tensor, path):
    img = tensor.permute(1, 2, 0).numpy()[:, :, ::-1] * 255.
    cv2.imwrite(path, img)