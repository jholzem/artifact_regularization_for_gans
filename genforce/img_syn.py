import math
import os
import time
import numpy as np
from models import stylegan_generator_idinvert
import cv2

def main():
    n = int(input("How many pictures do you want to produce: "))
    folder_name = input("Enter folder name: ")

    generator = stylegan_generator_idinvert.StyleGANGeneratorIdinvert('styleganinv_ffhq256')

    #model_weights = torch.load('models/pretrain/styleganinv_ffhq256_generator.pth')
    #generator.net.load_state_dict(model_weights)
    #generator.net.eval()

    os.mkdir('img_syn_genforce/' + folder_name)

    group_size = 500
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
            cv2.imwrite(f'img_syn_genforce/{folder_name}/img{str(k).zfill(2)}{str(i).zfill(6)}.png',img_rgb)
        print(f'saved nr {str(k)}')
    print(f'saved all')
    end = time.time()
    print(f'time elapsed: {end - start}')

if __name__ == '__main__':
    main()
