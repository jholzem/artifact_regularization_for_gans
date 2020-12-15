import os
import subprocess
import numpy as np
from PIL import Image

import torch

from models import MODEL_ZOO
from models import build_model
from models import build_discriminator


def postprocess(images):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
    images = images.detach().cpu().numpy()
    images = (images + 1) * 255 / 2
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images


def build(model_name='stylegan_ffhq256', generator=True, discriminator=True, encoder=True, gpu=False):
    """Builds generator and discriminator and loads pre-trained weights."""
    
    assert isinstance(model_name, str), "Input must be string"    
    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Build requested parts
    model = []
    description = []
    print(f'Building requested parts of the model `{model_name}`:')
    if generator:
        print(f'Building generator')
        model.append(build_model(**model_config, module='generator'))
        description.append('generator')
        print(f'Finish building generator.')
    if discriminator:
        print(f'Building discriminator')
        model.append(build_model(**model_config, module='discriminator'))
        description.append('discriminator')
        print(f'Finish building discriminator.')
    if encoder:
        print(f'Building encoder')
        model.append(build_model(**model_config, module='encoder'))
        description.append('encoder')
        print(f'Finish building emcoder.')

    # Load pre-trained weights.
    checkpoint_stylegan_path = os.path.join('pretrained_models', model_name + '.pth')
    checkpoint_encoder_path = os.path.join('pretrained_models', 'styleganinv_ffhq256_encoder.pth')
    print(f'Loading weights ...')
    checkpoint_stylegan = torch.load(checkpoint_stylegan_path, map_location='cpu')
    checkpoint_encoder = torch.load(checkpoint_encoder_path, map_location='cpu')

    for part, desc in zip(model, description):
        if desc == 'encoder':
            part.load_state_dict(checkpoint_encoder)
        else:
            if desc + '_smooth' in checkpoint_stylegan:
                part.load_state_dict(checkpoint_stylegan[desc + '_smooth'])
            else:
                part.load_state_dict(checkpoint_stylegan[desc])
            if gpu and desc == 'generator':
                part = part.cuda()
        part.eval()

    print(f'Finish loading checkpoint.')

    return model


def synthesize(generator, num, synthesis_kwargs=None, batch_size=1, seed=0):
    """Synthesize images."""
    assert num > 0 and batch_size > 0
    synthesis_kwargs = synthesis_kwargs or dict()

    # Set random seed.
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sample and synthesize.
    outputs = []
    for idx in range(0, num, batch_size):
        batch = min(batch_size, num - idx)
        # code = torch.randn(batch, generator.z_space_dim).cuda()
        code = torch.randn(batch, generator.z_space_dim)
        with torch.no_grad():
            images = generator(code, **synthesis_kwargs)['image']
            images = postprocess(images)
        outputs.append(images)
    return np.concatenate(outputs, axis=0)
