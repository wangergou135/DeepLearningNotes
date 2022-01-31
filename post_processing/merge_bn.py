
import torch
import os
from collections import OrderedDict
import cv2
import numpy as np
import torchvision.transforms as transforms


"""  Functions  """
def merge(params, name, layer):
    # global variables
    global weights, bias
    global bn_param

    if layer == 'Convolution':
        # save weights and bias when meet conv layer
        if 'weight' in name:
            weights = params.data
            bias = torch.zeros(weights.size()[0])
        elif 'bias' in name:
            bias = params.data
        bn_param = {}

    elif layer == 'BatchNorm':
        # save bn params
        bn_param[name.split('.')[-1]] = params.data

        # running_var is the last bn param in pytorch
        if 'running_var' in name:
            # let us merge bn ~
            tmp = bn_param['weight'] / torch.sqrt(bn_param['running_var'] + 1e-5)
            weights = tmp.view(tmp.size()[0], 1, 1, 1) * weights
            bias = tmp*(bias - bn_param['running_mean']) + bn_param['bias']

            return weights, bias

    return None, None
