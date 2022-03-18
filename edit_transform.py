import random
import kornia.enhance as ke
import torchvision.transforms as transforms
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

    def get_random_factors(self):
        factors = []
        for i in range(5):
            factors.append(random.randint(0, 4))
        return factors
    
    def __call__(self, img):
        factors = self.get_random_factors()
        for f, editor in zip(factors, self.editors):
            img = editor[f](img)

        return img, factors

            

if __name__ == "__main__":
    editor = ImageEditor()
    from SongUtils import IOUtils as iout
    inp = iout.readTensorImage('125_224.jpg', through='opencv')
    out, factors = editor(inp)
    print(factors)
    img = iout.tensor2cv(out)
    cv2.imwrite("edited.jpg", img[:, :, ::-1])

