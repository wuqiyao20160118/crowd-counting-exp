import torch
import numpy as np
import cv2


class RandomStretch(object):
    def __init__(self, max_stretch=0.05):
        """Random resize image according to the stretch
        Args:
            max_stretch(float): 0 to 1 value
        """
        self.max_stretch = max_stretch

    def __call__(self, sample):
        """
        Args:
            sample(numpy array): 3 or 1 dim image
        """
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w = sample.shape[:2]
        shape = int(w * scale_w), int(h * scale_h)
        return cv2.resize(sample, shape, cv2.INTER_LINEAR)


class ColorAug(object):
    def __init__(self, type_in='z'):
        if type_in == 'z':
            rgb_var = np.array([[3.2586416e+03, 2.8992207e+03, 2.6392236e+03],
                                [2.8992207e+03, 3.0958174e+03, 2.9321748e+03],
                                [2.6392236e+03, 2.9321748e+03, 3.4533721e+03]])
        if type_in == 'x':
            rgb_var = np.array([[2.4847285e+03, 2.1796064e+03, 1.9766885e+03],
                                [2.1796064e+03, 2.3441289e+03, 2.2357402e+03],
                                [1.9766885e+03, 2.2357402e+03, 2.7369697e+03]])
        self.v, _ = np.linalg.eig(rgb_var)
        self.v = np.sqrt(self.v)

    def __call__(self, sample):
        return sample + 0.1 * self.v * np.random.randn(3)


class RandomBlur(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, sample):
        if np.random.rand(1) < self.ratio:
            # random kernel size
            kernel_size = np.random.choice([3, 5, 7])
            # random gaussian sigma
            sigma = np.random.rand() * 5
            return cv2.GaussianBlur(sample, (kernel_size, kernel_size), sigma)
        else:
            return sample


class Normalize(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, sample):
        return (sample / 255. - self.mean) / self.std


class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose(2, 0, 1)
        return torch.from_numpy(sample.astype(np.float32))
