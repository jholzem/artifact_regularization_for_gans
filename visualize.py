import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from tensorboard.backend.event_processing import event_accumulator
from fourier import fourier_dissimilarity
from idinvert_pytorch.models import stylegan_generator_idinvert


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('conf_F', type=str, default='0 1e3 cos 1e-4', help='configuration Fourier loss only')
    parser.add_argument('ep_F', type=int, default=17, help='interesting number of epochs')

    parser.add_argument('conf_FA', type=str, default='1 1e3 cos 1e-5', help='configuration Fourier & adv. loss')
    parser.add_argument('ep_FA', type=int, default=10, help='interesting number of epochs')

    parser.add_argument('conf_A', type=str, default='1 0 cos 1e-6', help='configuration adv. loss only')
    parser.add_argument('ep_A', type=int, default=5, help='interesting number of epochs')

    parser.add_argument('res_dir', type=str, default='', help='results folder')

    return parser.parse_args()


def main():
    args = parse_args()

    random_indices = a_priori(100) # 1000
    syn(args.res_dir, args.conf_F, args.conf_FA, args.conf_A, random_indices)
    a_posteriori(args.conf_F+'_'+str(args.ep_F), args.conf_FA+'_'+str(args.ep_FA), args.conf_A+'_'+str(args.ep_A), random_indices)

    plot_FL(args.res_dir, args.conf_F, args.conf_FA, args.conf_A)
    plot_AL(args.res_dir, args.conf_F, args.conf_FA, args.conf_A)
    plot_ACC(args.res_dir, args.conf_F, args.conf_FA, args.conf_A)


def syn(res_dir, conf_F, conf_FA, conf_A, random_indices):

    latent_files = sorted(os.listdir('data/latent'))

    for conf in [conf_F, conf_FA, conf_A]:

        for n_ep in range(1, 21):

            generator_path = os.path.join(res_dir, conf + '_' + str(n_ep) + '_generator.pth')
            generator = stylegan_generator_idinvert.StyleGANGeneratorIdinvert(generator_path)

            for idx in random_indices:

                latent = np.genfromtxt(os.path.join('data/latent/', latent_files[idx]), delimiter=',')
                latent = torch.from_numpy(latent)
                latent = latent.float()
                latent.unsqueeze_(0)

                image = generator.net.synthesis(latent.type(torch.cuda.FloatTensor))
                image = image.squeeze(0)

                image = image.permute(1, 2, 0)

                image = image.detach().cpu().numpy()
                image = (image + 1) * 128
                image = image.astype(int)
                image = np.float32(image)
                img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                start = generator_path.rfind('/')+1
                cv2.imwrite(f'visualization/img/{generator_path[start:-14]}_{latent_files[idx][:-4]}.png', img_rgb)


def a_priori(n_pairs):

    path = 'data/'

    #plot settings
    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 20}

    matplotlib.rc('font', **font)
    linewidth = 3

    # load images
    real_list = []
    fake_list = []

    real_files = sorted(os.listdir(path + 'real'))
    fake_files = sorted(os.listdir(path + 'fake'))

    for idx in range(min(len(real_files), n_pairs)):
        real_list.append(plt.imread(path + 'real/' + real_files[idx]))
        fake_list.append(plt.imread(path + 'fake/' + fake_files[idx]))

    real = torch.from_numpy(np.array(real_list))
    fake = torch.from_numpy(np.array(fake_list)[:, :, :, :3])

    # compute fourier dissimilarity for different frequency thresholds
    fd_2_list = []
    fd_cos_list = []

    thresholds = np.arange(1, 128)

    for thres in thresholds:
        fd_2_list.append(fourier_dissimilarity(real, fake, '2', thres).numpy())
        fd_cos_list.append(fourier_dissimilarity(real, fake, 'cos', thres).numpy())

    fd_2 = np.array(fd_2_list) / 2e-4
    fd_cos = np.array(fd_cos_list)

    # visualize fourier representations
    np.random.seed(9)
    random_indices = np.random.choice(n_pairs, size=3)
    print(random_indices)
    real_sample = real[random_indices]
    fake_sample = fake[random_indices]

    real_sample_ft = torch.norm(torch.rfft(rgb2gray(real_sample), signal_ndim=2), dim=3)
    fake_sample_ft = torch.norm(torch.rfft(rgb2gray(fake_sample), signal_ndim=2), dim=3)

    f = plt.figure(figsize=(18, 12))

    for idx in range(3):

        ax_real = f.add_subplot(3, 4, idx * 4 + 1)
        ax_real_ft = f.add_subplot(3, 4, idx * 4 + 2)
        ax_fake_ft = f.add_subplot(3, 4, idx * 4 + 3)
        ax_fake = f.add_subplot(3, 4, idx * 4 + 4)

        ax_real.imshow(real_sample[idx])
        ax_real_ft.imshow(np.minimum(real_sample_ft[idx, 128:].numpy(), 2 * np.mean(real_sample_ft[idx, 128:].numpy())))
        ax_fake_ft.imshow(np.minimum(fake_sample_ft[idx, 128:].numpy(), 2 * np.mean(fake_sample_ft[idx, 128:].numpy())))
        ax_fake.imshow(fake_sample[idx])

        for ax in [ax_real, ax_real_ft, ax_fake_ft, ax_fake]:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

    plt.savefig('visualization/spectra_priori.pdf')

    # plot fourier dissimilarity values
    f = plt.figure(figsize=(16, 12))
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)

    plot_range = np.arange(10, 110)

    ax1.plot(thresholds[plot_range], np.mean(fd_2[plot_range], axis=1),
             linewidth=linewidth)
    ax1.set_ylim([0, 1200])
    ax1.set_xticklabels([])
    color = 'tab:orange'
    ax_diff = ax1.twinx()
    ax_diff.set_ylabel('change of Frobenius norm diss.', color=color)
    ax_diff.plot(thresholds[plot_range][:-1], -np.diff(np.mean(fd_2[plot_range], axis=1)), color=color,
                 linewidth=linewidth)
    ax_diff.tick_params(axis='y', labelcolor=color)
    ax_diff.set_ylim([0, 120])
    ax1.grid()
    ax1.set_ylabel('Frobenius norm diss.')

    ax2.plot(thresholds[plot_range], np.mean(fd_cos[plot_range], axis=1), linewidth=linewidth)
    ax2.grid()
    ax2.set_xlabel('truncation threshold')
    ax2.set_ylabel('cosine diss.')

    plt.savefig('visualization/truncation.pdf')

    return random_indices


def a_posteriori(stem_F, stem_FA, stem_A, random_indices):
    # load and visualize Fourier respresentation of images after training

    path_real = 'data/real/'
    path_fake = 'data/fake/'
    path_trained = 'visualization/img/'

    real_files = sorted(os.listdir('data/real'))

    f = plt.figure(figsize=(14, 16))

    for idx, rand_idx in enumerate(random_indices):

        # read images
        im_real = torch.from_numpy(plt.imread(path_real + real_files[rand_idx])[:, :, :3]).unsqueeze(0)
        im_fake = torch.from_numpy(plt.imread(path_fake + real_files[rand_idx])[:, :, :3]).unsqueeze(0)
        im_four = torch.from_numpy(
            plt.imread(path_trained + stem_F + '_' + real_files[rand_idx])[:, :, :3]).unsqueeze(0)
        im_mix = torch.from_numpy(
            plt.imread(path_trained + stem_FA + '_' + real_files[rand_idx])[:, :, :3]).unsqueeze(0)
        im_adv = torch.from_numpy(
            plt.imread(path_trained + stem_A + '_' + real_files[rand_idx])[:, :, :3]).unsqueeze(0)

        # compute spectra
        ft_real = torch.norm(torch.rfft(rgb2gray(im_real), signal_ndim=2), dim=3)
        ft_fake = torch.norm(torch.rfft(rgb2gray(im_fake), signal_ndim=2), dim=3)
        ft_four = torch.norm(torch.rfft(rgb2gray(im_four), signal_ndim=2), dim=3)
        ft_mix = torch.norm(torch.rfft(rgb2gray(im_mix), signal_ndim=2), dim=3)
        ft_adv = torch.norm(torch.rfft(rgb2gray(im_adv), signal_ndim=2), dim=3)

        # plot
        ax_real = f.add_subplot(6, 5, idx * 10 + 1)
        ax_real_ft = f.add_subplot(6, 5, idx * 10 + 6)
        ax_fake = f.add_subplot(6, 5, idx * 10 + 2)
        ax_fake_ft = f.add_subplot(6, 5, idx * 10 + 7)
        ax_four = f.add_subplot(6, 5, idx * 10 + 3)
        ax_four_ft = f.add_subplot(6, 5, idx * 10 + 8)
        ax_mix = f.add_subplot(6, 5, idx * 10 + 4)
        ax_mix_ft = f.add_subplot(6, 5, idx * 10 + 9)
        ax_adv = f.add_subplot(6, 5, idx * 10 + 5)
        ax_adv_ft = f.add_subplot(6, 5, idx * 10 + 10)

        ax_real.imshow(im_real.squeeze())
        ax_real_ft.imshow(np.minimum(ft_real[0, 128:].numpy(), 2 * np.mean(ft_real[0, 128:].numpy())))
        ax_fake.imshow(im_fake.squeeze())
        ax_fake_ft.imshow(np.minimum(ft_fake[0, 128:].numpy(), 2 * np.mean(ft_fake[0, 128:].numpy())))
        ax_four.imshow(im_four.squeeze())
        ax_four_ft.imshow(np.minimum(ft_four[0, 128:].numpy(), 2 * np.mean(ft_four[0, 128:].numpy())))
        ax_mix.imshow(im_mix.squeeze())
        ax_mix_ft.imshow(np.minimum(ft_mix[0, 128:].numpy(), 2 * np.mean(ft_mix[0, 128:].numpy())))
        ax_adv.imshow(im_adv.squeeze())
        ax_adv_ft.imshow(np.minimum(ft_adv[0, 128:].numpy(), 2 * np.mean(ft_adv[0, 128:].numpy())))

        for ax in [ax_real, ax_real_ft, ax_fake, ax_fake_ft, ax_four, ax_four_ft, ax_mix, ax_mix_ft, ax_adv, ax_adv_ft]:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

    plt.savefig('visualization/spectra_posteriori.pdf')


# utility function to convert RGB into gray-scale images
def rgb2gray(images):
    return 0.299*images[:, :, :, 0] + 0.587*images[:, :, :, 1] + 0.114*images[:, :, :, 2]


def plot_FL(res_dir, conf_F, conf_FA, conf_A):

    data_0 = tb2array(os.path.join(res_dir, conf_A + '_workdir', 'events'), 'fourier_loss')
    data_c = tb2array(os.path.join(res_dir, conf_FA + '_workdir', 'events'), 'fourier_loss')
    data_inf = tb2array(os.path.join(res_dir, conf_F + '_workdir', 'events'), 'fourier_loss')

    fl_0 = data_0
    fl_c = data_c
    fl_inf = data_inf

    n = 2###

    std_0 = np.std(rolling_window(fl_0, n), 1)
    std_c = np.std(rolling_window(fl_c, n), 1)
    std_inf = np.std(rolling_window(fl_inf, n), 1)

    x_ax = np.linspace(0.0, 20.0, num=1000 - n + 1)
    x_tic = list(range(0, 21, 5))

    fl_0_ma = np.convolve(fl_0, np.ones(n), 'valid') / n
    fl_c_ma = np.convolve(fl_c, np.ones(n), 'valid') / n
    fl_inf_ma = np.convolve(fl_inf, np.ones(n), 'valid') / n

    plt.figure(figsize=(7, 5.5))

    p_0, = plt.plot(x_ax, fl_0_ma, 'r', markersize=2)
    p_c, = plt.plot(x_ax, fl_c_ma, 'g', markersize=2)
    p_inf, = plt.plot(x_ax, fl_inf_ma, 'b', markersize=2)

    plt.fill_between(x_ax, fl_0_ma - std_0, fl_0_ma + std_0, where=fl_0_ma + std_0 >= fl_0_ma - std_0, facecolor='r', alpha=0.2, interpolate=True)
    plt.fill_between(x_ax, fl_c_ma - std_c, fl_c_ma + std_c, where=fl_c_ma + std_c >= fl_c_ma - std_c, facecolor='g', alpha=0.2, interpolate=True)
    plt.fill_between(x_ax, fl_inf_ma - std_inf, fl_inf_ma + std_inf, where=fl_inf_ma + std_inf >= fl_inf_ma - std_inf, facecolor='b', alpha=0.2, interpolate=True)

    plt.grid()
    plt.xticks(x_tic)
    plt.xlim([0, 20])
    plt.ylim([0.205, 0.25])
    plt.xlabel('$N_\mathrm{epochs}$')
    plt.ylabel('Fourier loss $\ell_\mathrm{F,cos}$')
    plt.legend([p_inf, p_c, p_0], ['Fourier loss only, $\eta = 10^{-4}$', 'Fourier & adv. loss, $\eta = 10^{-5}$',
                                   'adv. loss only, $\eta = 10^{-6}$'], loc='upper left')

    plt.savefig('visualization/Fourier_loss.pdf')

    print("saved plot of Fourier loss")


def plot_AL(res_dir, conf_F, conf_FA, conf_A):

    data_0 = tb2array(os.path.join(res_dir, conf_A + '_workdir', 'events'), 'g_loss')
    data_c = tb2array(os.path.join(res_dir, conf_FA + '_workdir', 'events'), 'g_loss')
    data_inf = tb2array(os.path.join(res_dir, conf_F + '_workdir', 'events'), 'g_loss')

    fl_0 = data_0
    fl_c = data_c
    fl_inf = data_inf

    n = 2###

    std_0 = np.std(rolling_window(fl_0, n), 1)
    std_c = np.std(rolling_window(fl_c, n), 1)
    std_inf = np.std(rolling_window(fl_inf, n), 1)

    x_ax = np.linspace(0.0, 20.0, num=1000-n+1)
    x_tic = list(range(0,21,5))

    fl_0_ma = np.convolve(fl_0, np.ones(n), 'valid') / n
    fl_c_ma = np.convolve(fl_c, np.ones(n), 'valid') / n
    fl_inf_ma = np.convolve(fl_inf, np.ones(n), 'valid') / n

    plt.figure(figsize=(7, 5.5))

    p_0, = plt.plot(x_ax, fl_0_ma, 'r', markersize=2)
    p_c, = plt.plot(x_ax, fl_c_ma, 'g', markersize=2)
    p_inf, = plt.plot(x_ax, fl_inf_ma, 'b',  markersize=2)

    plt.fill_between(x_ax, fl_0_ma - std_0, fl_0_ma + std_0, where=fl_0_ma + std_0 >= fl_0_ma - std_0, facecolor='r',alpha=0.2, interpolate=True)
    plt.fill_between(x_ax, fl_c_ma - std_c, fl_c_ma + std_c, where=fl_c_ma + std_c >= fl_c_ma - std_c, facecolor='g',alpha=0.2, interpolate=True)
    plt.fill_between(x_ax, fl_inf_ma - std_inf, fl_inf_ma + std_inf, where=fl_inf_ma + std_inf >= fl_inf_ma - std_inf, facecolor='b',alpha=0.2, interpolate=True)

    plt.grid()
    plt.xticks(x_tic)
    plt.xlim([0, 20])
    plt.xlabel('$N_\mathrm{epochs}$')
    plt.ylabel('adversarial loss')
    plt.ylim([0, 0.38])
    plt.legend([p_inf, p_c, p_0],['Fourier loss only, $\eta = 10^{-4}$', 'Fourier & adv. loss, $\eta = 10^{-5}$', 'adv. loss only, $\eta = 10^{-6}$'], loc='lower left')

    plt.savefig('visualization/adversarial_loss.pdf')

    print("saved plot of adversarial loss")


def plot_ACC(res_dir, conf_F, conf_FA, conf_A):

    data_0 = np.genfromtxt(os.path.join(res_dir, conf_A + '_accuracies.txt'), delimiter=',')
    data_c = np.genfromtxt(os.path.join(res_dir, conf_FA + '_accuracies.txt'), delimiter=',')
    data_inf = np.genfromtxt(os.path.join(res_dir, conf_F + '_accuracies.txt'), delimiter=',')

    baseline = 0.9015

    data_0 = np.append(np.array(baseline), data_0[0:20])
    data_c = np.append(np.array(baseline), data_c[0:20])
    data_inf = np.append(np.array(baseline), data_inf[0:20])

    x_ax = list(range(0,21))
    x_tic = list(range(0,21,5))

    plt.figure(figsize=(7, 5.5))

    p_base, = plt.plot(np.arange(0,21), np.arange(0,21)*0 + baseline, 'k--')
    p_0, = plt.plot(x_ax, data_0, 'r-o', markersize=5)
    p_c, = plt.plot(x_ax, data_c, 'g-o', markersize=5)
    p_inf, = plt.plot(x_ax, data_inf, 'b-o', markersize=5)

    plt.grid()
    plt.xticks(x_tic)
    plt.xlim([0, 20])
    plt.xlabel('$N_\mathrm{epochs}$')
    plt.ylabel('detection accuracy')
    plt.ylim([0,1])
    plt.yticks(np.arange(0,1.1,0.1))
    plt.legend([p_base, p_inf, p_c, p_0], ['baseline','Fourier loss only, $\eta = 10^{-4}$', 'Fourier & adv. loss, $\eta = 10^{-5}$', 'adv. loss only, $\eta = 10^{-6}$'], loc='lower left')

    plt.savefig('visualization/detection_accuracy.pdf')

    print('saved accuracy plot')


def tb2array(events_folder, signal):

    events_files = os.listdir(events_folder)
    ea = event_accumulator.EventAccumulator(os.path.join(events_folder, events_files[0]))
    ea.Reload()

    return np.array(ea.Scalars(signal))[:, 2]


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


if __name__ == '__main__':
    main()
