import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('savepath', type=str, default='', help='savepath')
    parser.add_argument('acc1', type=str, default='', help='acc1')
    parser.add_argument('acc2', type=str, default ='', help='acc2')
    parser.add_argument('acc3', type=str, default ='', help='acc3')

    return parser.parse_args()

def main():
    args = parse_args()
    acc_0 = np.genfromtxt(args.acc1, delimiter=',')
    acc_c = np.genfromtxt(args.acc2, delimiter=',')
    acc_inf = np.genfromtxt(args.acc3, delimiter=',')

    if os.path.isdir(args.savepath) == 0:
        os.mkdir(args.savepath)

    fl_0 = acc_0[1:,2]
    fl_c = acc_c[1:,2]
    fl_inf = acc_inf[1:,2]

    n = 50

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

    # p_0_2, = plt.plot(x_ax[a:-a], fl_0_ma + std_0, 'r', alpha=0.2, markersize=2)
    # p_c_2, = plt.plot(x_ax[a:-a], fl_c_ma + std_c, 'g', alpha=0.2,markersize=2)
    # p_inf_2, = plt.plot(x_ax_2[a:-a], fl_inf_ma + std_inf , 'b', alpha=0.2, markersize=2)

    # p_0_3, = plt.plot(x_ax[a:-a], fl_0_ma - std_0, 'r', alpha=0.2, markersize=2)
    # p_c_3, = plt.plot(x_ax[a:-a], fl_c_ma - std_c, 'g', alpha=0.2,markersize=2)
    # p_inf_3, = plt.plot(x_ax_2[a:-a], fl_inf_ma - std_inf , 'b', alpha=0.2, markersize=2)

    fill_0 = plt.fill_between(x_ax, fl_0_ma - std_0, fl_0_ma + std_0, where=fl_0_ma + std_0 >= fl_0_ma - std_0, facecolor='r',alpha=0.2, interpolate=True)
    fill_c = plt.fill_between(x_ax, fl_c_ma - std_c, fl_c_ma + std_c, where=fl_c_ma + std_c >= fl_c_ma - std_c, facecolor='g',alpha=0.2, interpolate=True)
    fill_inf = plt.fill_between(x_ax, fl_inf_ma - std_inf, fl_inf_ma + std_inf, where=fl_inf_ma + std_inf >= fl_inf_ma - std_inf, facecolor='b',alpha=0.2, interpolate=True)


    plt.grid()
    # plt.axvline(x=5, linestyle=(0, (1, 10)), color='k', markersize=2)
    plt.xticks(x_tic)
    plt.xlim([0, 20])
    plt.xlabel('$N_\mathrm{epochs}$')
    plt.ylabel('adversarial loss')
    plt.ylim([0, 0.38])
    # plt.title('Accuracy comparison after different epochs of training w/o regularization, with Frobenius and cosine')
    plt.legend([p_inf, p_c, p_0],['Fourier loss only, $\eta = 10^{-4}$', 'Fourier & adv. loss, $\eta = 10^{-5}$', 'adv. loss only, $\eta = 10^{-6}$'], loc='lower left')

    plt.savefig(args.savepath+'plot_AL.pdf')

    #plt.show()

    print("saved plot of adversary loss")

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

if __name__ == '__main__':
    main()
