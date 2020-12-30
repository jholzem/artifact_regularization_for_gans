import argparse
import math
import os
import time
import numpy as np
from models import stylegan_generator_idinvert
import cv2

# test['models']['generator']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, default = 10000 , help='nr of images to be created.')
    parser.add_argument('generator', type=str, default = 'styleganinv_ffhq256' , help='choice of generator')

    return parser.parse_args()

def main():
    args = parse_args()

    n = args.n
    generator = stylegan_generator_idinvert.StyleGANGeneratorIdinvert(args.generator)

    #model_weights = torch.load('models/pretrain/styleganinv_ffhq256_generator.pth')
    #generator.net.load_state_dict(model_weights)
    #generator.net.eval()

    os.mkdir('test_syn')
    os.mkdir('test_syn' + '/0_real')
    os.mkdir('test_syn' + '/1_fake')
    # print(f'folder {folder_name} created')

    group_size = 10
    start = time.time()
    for k in range(math.ceil(n/group_size)):

        latent_codes = generator.sample(group_size, seed = k)
        # latent_codes = generator.preprocess(latent_codes)
        images = generator.synthesize(latent_codes)

        # save stuff
        print(f'save nr {str(k)}')
        for i, img, in enumerate(images.get('image')):
            img_reshape = np.moveaxis(img, 0, -1)
            img_reshape =( img_reshape + 1) * 128
            img_rgb = cv2.cvtColor(img_reshape, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'test_syn/1_fake/img{str(k).zfill(2)}{str(i).zfill(6)}.png',img_rgb)
        print(f'saved nr {str(k)}')
    print(f'saved {str(k)} x {str(group_size)} images')
    end = time.time()
    print(f'time elapsed: {end - start}')

if __name__ == '__main__':
    main()
