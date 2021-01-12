import argparse
import os
import numpy as np
import torch

from models import stylegan_generator_idinvert
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('generator', type=str, default='', help='choice of generator')
    parser.add_argument('syn_dir', type=str, default ='', help='choice of directions')
    parser.add_argument('latent_dir', type=str, default ='', help='choice of latents')
    parser.add_argument('latent_name', type=str, default ='', help='choice of latents')

    return parser.parse_args()

def main():
    args = parse_args()
    generator = stylegan_generator_idinvert.StyleGANGeneratorIdinvert(args.generator)

    if os.path.isdir(args.syn_dir) == 0:
        os.mkdir(args.syn_dir)


    latent = np.genfromtxt(os.path.join(args.latent_dir, args.latent_name), delimiter=',')
    latent = torch.from_numpy(latent)
    latent = latent.float()
    latent.unsqueeze_(0)

    image = generator.net.synthesis(latent)
    image = image.squeeze(0)

    image = image.permute(1, 2, 0)

    image = image.detach().numpy()
    image = (image + 1) * 128
    image = image.astype(int)
    image = np.float32(image)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{args.syn_dir}/{args.generator[16:-4]}.png', img_rgb)

    print(f'saved image')

if __name__ == '__main__':
    main()
