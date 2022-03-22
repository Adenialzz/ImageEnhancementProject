import random
import torch
import kornia.enhance as ke
import cv2


class ImageEditor:
    def __init__(self):
        self.num_intensities = 5
        brightness_factors = [-0.5, -0.25, 0., 0.5, 0.5]
        gamma_factors = [0.2, 0.6, 1., 1.4, 1.8]
        contrast_factors = [0.2, 0.6, 1., 1.4, 1.8]
        # satruration_factors = [0., 0.5, 1., 2.5, 5.]
        satruration_factors = [0., 0.5, 1., 1.5, 2.]
        hue_factors = [-3., -1.5, 0., 1.5, 3.]

        brightness_editors_list = [ke.AdjustBrightness(f) for f in brightness_factors]
        gamma_editors_list = [ke.AdjustGamma(f) for f in gamma_factors]
        contrast_editors_list = [ke.AdjustContrast(f) for f in contrast_factors]
        satruration_editors_list = [ke.AdjustSaturation(f) for f in satruration_factors]
        hue_editors_list = [ke.AdjustHue(f) for f in hue_factors]

        self.editors = [brightness_editors_list, gamma_editors_list, contrast_editors_list, satruration_editors_list, hue_editors_list]

    def get_random_factors(self, single_filter=False, polar_intensity=False):
        factors = [2 for _ in range(5)]
        if single_filter:
            chosen_filter = random.choice([i for i in range(5)])
            if polar_intensity:
                factors[chosen_filter] = random.choice([0, 2, 4])
            else:
                factors[chosen_filter] = random.randint(0, 4)

        else:
            for i in range(5):
                if polar_intensity:
                    factors[i] = random.choice([0, 2, 4])
                else:
                    factors[i] = random.randint(0, 4)

        return factors
    
    def gen_filter_params(self, factors, shape=(224, 224)):
        factor2value_map = {
            0: -1.,
            1: -0.5,
            2: 0.,
            3: 0.5,
            4: 1.
        }

        filter_channels = None
        for f in factors:
            value = factor2value_map[f]
            channel = torch.ones(shape).unsqueeze(dim=0) * value
            if filter_channels is not None:
                filter_channels = torch.cat((filter_channels, channel), dim=0)
            else:
                filter_channels = channel
        return filter_channels
    
    def __call__(self, img, filters_shape=(224, 224), single_filter=False, polar_intensity=False, factors=None):
        if factors is None:
            factors = self.get_random_factors(single_filter, polar_intensity)
        for f, editor in zip(factors, self.editors):
            img = editor[f](img)
        
        filter_channels = self.gen_filter_params(factors, filters_shape)

        return img, filter_channels

            

if __name__ == "__main__":
    editor = ImageEditor()
    # from SongUtils import IOUtils as iout
    # inp = iout.readTensorImage('imgs/125_224.jpg', through='opencv')
    inp = torch.ones(3, 224, 224)
    # out, factors = editor(inp, [0, 3, 2, 2, 4])
    for i in range(100):
        out, filter_channels = editor(inp, (768, ), single_filter=False, polar_intensity=False)
        print(filter_channels.shape)
        # l = [filter_channels[i, 0, 0].item() for i in range(5)]
        # print(l, sum(l))
    # img = iout.tensor2cv(out)
    # cv2.imwrite("edited.jpg", img)

