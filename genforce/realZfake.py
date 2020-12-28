import torch
import pickle
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils.inverter as inv
from models.stylegan_generator_idinvert import StyleGANGeneratorIdinvert

# settings
path_images = '/cluster/scratch/FFHQ_256'
n_iter = 100

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

latents = []
fakes = []
losses = []

counter = 1
start = time.time()

for i in range(2):

    # read .png files
    real_list = []
    for j in range(5):
        file = path_images + '/' + str(i * 1000 + j).zfill(5) + '.png'
        real_list.append(preprocess(plt.imread(file)))

    real = torch.from_numpy(np.array(real_list))

    # create optimized latent code & fake images
    for k in range(real.shape[0]):
        latent, fake, loss = Inverter.invert_offline(image=real[k].unsqueeze(0))
        latents.append(latent.squeeze().detach().numpy())
        fakes.append(fake.squeeze().detach().numpy())
        losses.append(loss)

        print('ETA:', (11000-counter)*(time.time()-start)/counter / 3600, 'hours')
        counter += 1

latents = np.array(latents)
fakes = np.array(fakes)
losses = np.array(losses)

print((time.time() - start) / 60, 'min')
print('=', (time.time() - start) / 3600, 'h')

# save
pickle.dump(latents, open('lat.p', 'wb'))
pickle.dump(fakes, open('fak.p', 'wb'))
pickle.dump(losses, open('los.p', 'wb'))
