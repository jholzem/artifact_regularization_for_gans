import torch
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils.inverter as inv
from models.stylegan_generator_idinvert import StyleGANGeneratorIdinvert

# settings
path_images = 'genforce/data/FFHQ_256'
n_iter = 200

# initialize generator & invertor
G = StyleGANGeneratorIdinvert('styleganinv_ffhq256')
Inverter = inv.StyleGANInverter(G, 'styleganinv_ffhq256', iteration=n_iter)


def preprocess(image):
    image = image.astype(np.float32)
    image = image[:, :, :3]
    if image.shape[:2] != [256, 256]:
        image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32)
    image = image * 2 - 1
    return image.astype(np.float32).transpose(2, 0, 1)


# process images in packages of 1000


for i in range(6, 9):

    # read .png files
    real_list = []
    for j in range(1000):
        file = path_images + '/' + str(i * 1000 + j).zfill(5) + '.png'
        real_list.append(preprocess(plt.imread(file)))

    real = torch.from_numpy(np.array(real_list))

    latents = []
    fakes = []
    losses = []

    # create optimized latent code & fake images
    for k in range(real.shape[0]):
        latent, fake, loss = Inverter.invert_offline(image=(real[k].type(torch.cuda.FloatTensor)).unsqueeze(0))
        latents.append(latent.squeeze().detach().cpu().numpy())
        fakes.append(fake.squeeze().detach().cpu().numpy())
        losses.append(loss)

    latents = np.array(latents)
    fakes = np.array(fakes)
    losses = np.array(losses)

    # save
    pickle.dump(latents, open('latA' + str(i).zfill(2) + '.p', 'wb'))
    pickle.dump(fakes, open('fakA' + str(i).zfill(2) + '.p', 'wb'))
    pickle.dump(losses, open('losA' + str(i).zfill(2) + '.p', 'wb'))
