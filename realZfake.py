import os
import torch
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import idinvert_pytorch.utils.inverter as inv
from idinvert_pytorch.models.stylegan_generator_idinvert import StyleGANGeneratorIdinvert
from shutil import copyfile


def preprocess(image):
    image = image.astype(np.float32)
    image = image[:, :, :3]
    if image.shape[:2] != [256, 256]:
        image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32)
    image = image * 2 - 1
    return image.astype(np.float32).transpose(2, 0, 1)


def main():

    # settings
    path_save = 'data/reproduced/'
    path_images = 'data/reproduced/FFHQ_256/'
    n_iter = 200
    n_outer = 11
    n_inner = 2

    if not os.path.isdir(os.path.join(path_save, 'real')):
        os.mkdir(os.path.join(path_save, 'real'))

    if not os.path.isdir(os.path.join(path_save, 'latent')):
        os.mkdir(os.path.join(path_save, 'latent'))

    if not os.path.isdir(os.path.join(path_save, 'fake')):
        os.mkdir(os.path.join(path_save, 'fake'))

    # initialize generator & invertor
    G = StyleGANGeneratorIdinvert('styleganinv_ffhq256')
    Inverter = inv.StyleGANInverter(G, 'styleganinv_ffhq256', iteration=n_iter)

    latents = []
    fakes = []
    losses = []

    for i in range(n_outer):

        # read .png files
        real_list = []
        for j in range(n_inner):
            file = path_images + str(i * n_inner + j).zfill(5) + '.png'
            real_list.append(preprocess(plt.imread(file)))

        real = torch.from_numpy(np.array(real_list))

        # create optimized latent code & fake images
        for k in range(real.shape[0]):
            latent, fake, loss = Inverter.invert_offline(image=(real[k].type(torch.cuda.FloatTensor)).unsqueeze(0))
            latents.append(latent.squeeze().detach().cpu().numpy())
            fakes.append(fake.squeeze().detach().cpu().numpy())
            losses.append(loss)

    latents = np.array(latents)
    fakes = np.array(fakes)
    losses = np.array(losses)

    ids = np.arange(n_inner*n_outer)

    loss_threshold = 0.3

    index_keep = np.arange(n_inner * n_outer)[losses < loss_threshold]
    print('Accepting', str(np.round(1000 * len(index_keep) / len(ids)) / 10), '% of examples.')

    counter = 1
    start = time.time()
    stamp = time.time()

    for idx in index_keep:

        # real
        copyfile(path_images + str(ids[idx]).zfill(5) + '.png', path_save + 'real/' + str(ids[idx]).zfill(5) + '.png')

        # latent
        np.savetxt(path_save + 'latent/' + str(ids[idx]).zfill(5) + '.csv', latents[idx], delimiter=',')

        # fake
        plt.imsave(path_save + 'fake/' + str(ids[idx]).zfill(5) + '.png', (fakes[idx].transpose(1, 2, 0) + 1) / 2)

        if time.time() - stamp > 10:
            stamp = time.time()
            print('ETA:', str(np.round((len(index_keep) - counter) * (time.time() - start) / counter / 60)),
                  'min left.')

        counter += 1


if __name__ == '__main__':
    main()
