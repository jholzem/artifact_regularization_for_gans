import dnnlib
import dnnlib.tflib
import pickle
import torch
import collections
from collections import OrderedDict

"""
dnnlib.tflib.init_tf()
_, _, D, _ = pickle.load(open('./styleganinv_face_256.pkl', 'rb'))
weights_pt = collections.OrderedDict((k, torch.from_numpy(v.value().eval())) \
                                       for k, v in D.trainables.items())
"""

weights = torch.load('./weights/styleganinv_face_256_discriminator.pt')[0]
