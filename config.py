import numpy as np


class Config(object):
    epoches = 50
    start_lr = 3e-3
    end_lr = 1e-5
    warm_lr = 1e-3
    warm_scale = warm_lr / start_lr
    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoches)[0]
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoches)[1] / \
            np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoches)[0]
    momentum = 0.9
    weight_decay = 0.0005

    clip = 100  # grad clip

    max_inter = 80
    gray_ratio = 0.25


config = Config()
