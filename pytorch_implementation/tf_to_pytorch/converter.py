import dnnlib
import dnnlib.tflib
import pickle
import torch
import numpy as np
import collections
from collections import OrderedDict

"""
dnnlib.tflib.init_tf()
_, _, D, _ = pickle.load(open('./styleganinv_face_256.pkl', 'rb'))
weights_pt = collections.OrderedDict((k, torch.from_numpy(v.value().eval())) \
                                       for k, v in D.trainables.items())
"""

weights = torch.load('./weights/styleganinv_face_256_discriminator.pt')[0]


def key_translate(k):
    """Translates the keys from the tensorflow weights to weights
    which are compatible with the pytorch model"""

    k = k.lower().split('/')
    if k[0][0] == 'f':
        idx = k[0][-1]
        k[0] = k[0].replace(k[0], 'input' + idx)
        k = '.'.join(k)
    else:
        resolution = int(k[0].split('x')[0])
        log2_res = int(np.log2(resolution))
        if log2_res == 2:
            idx = 2 * (8 - log2_res)
            if k[1][0] == 'd':
                idx += int(k[1][5]) + 1
            idx = str(idx)
        else:
            idx = str(2 * (8 - log2_res) + int(k[1][4]))
        k = '.'.join(['layer' + idx, k[2]])

    return k


def weight_translate(k, w):
    """Translates the weights"""

    k = key_translate(k)
    if k.endswith('.weight'):
        if w.dim() == 2:
            w = w.t()
        elif w.dim() == 1:
            pass
        else:
            assert w.dim() == 4
            w = w.permute(3, 2, 0, 1)
    return w
