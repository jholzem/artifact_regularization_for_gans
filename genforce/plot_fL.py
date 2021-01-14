import numpy as np
import matplotlib.pyplot as plt
import pandas


def main():
    acc_0 = np.genfromtxt('models/pretrain/0_cos_1e-6_fourier_loss.csv', delimiter=',')
    acc_c = np.genfromtxt('models/pretrain/1e3_cos_1e-5_fourier_loss.csv', delimiter=',')
    acc_inf = np.genfromtxt('models/pretrain/inf_cos_1e-4_fourier_loss.csv', delimiter=',')

    fl_0 = acc_0[1:,2]
    fl_c = acc_c[1:,2]
    fl_inf = acc_inf[1:401,2]

    n = 50
    a = int(n/2 - 0.5)

    std_0 = np.std(rolling_window(fl_0, n), 1)
    std_c = np.std(rolling_window(fl_c, n), 1)
    std_inf = np.std(rolling_window(fl_inf, n), 1)

    x_ax = np.linspace(0.0, 20.0, num=1000-n+1)
    x_ax_2 = np.linspace(0.0, 20.0, num=400-n+1)
    x_tic = list(range(0,21,5))

    fl_0_ma = np.convolve(fl_0, np.ones(n), 'valid') / n
    fl_c_ma = np.convolve(fl_c, np.ones(n), 'valid') / n
    fl_inf_ma = np.convolve(fl_inf, np.ones(n), 'valid') / n

    plt.figure(figsize=(7, 5.5))

    p_0, = plt.plot(x_ax, fl_0_ma, 'r', markersize=2)
    p_c, = plt.plot(x_ax, fl_c_ma, 'g', markersize=2)
    p_inf, = plt.plot(x_ax_2, fl_inf_ma, 'b',  markersize=2)

    # p_0_2, = plt.plot(x_ax[a:-a], fl_0_ma + std_0, 'r', alpha=0.2, markersize=2)
    # p_c_2, = plt.plot(x_ax[a:-a], fl_c_ma + std_c, 'g', alpha=0.2,markersize=2)
    # p_inf_2, = plt.plot(x_ax_2[a:-a], fl_inf_ma + std_inf , 'b', alpha=0.2, markersize=2)

    # p_0_3, = plt.plot(x_ax[a:-a], fl_0_ma - std_0, 'r', alpha=0.2, markersize=2)
    # p_c_3, = plt.plot(x_ax[a:-a], fl_c_ma - std_c, 'g', alpha=0.2,markersize=2)
    # p_inf_3, = plt.plot(x_ax_2[a:-a], fl_inf_ma - std_inf , 'b', alpha=0.2, markersize=2)

    fill_0 = plt.fill_between(x_ax, fl_0_ma - std_0, fl_0_ma + std_0, where=fl_0_ma + std_0 >= fl_0_ma - std_0, facecolor='r',alpha=0.2, interpolate=True)
    fill_c = plt.fill_between(x_ax, fl_c_ma - std_c, fl_c_ma + std_c, where=fl_c_ma + std_c >= fl_c_ma - std_c, facecolor='g',alpha=0.2, interpolate=True)
    fill_inf = plt.fill_between(x_ax_2, fl_inf_ma - std_inf, fl_inf_ma + std_inf, where=fl_inf_ma + std_inf >= fl_inf_ma - std_inf, facecolor='b',alpha=0.2, interpolate=True)

    plt.grid()
    # plt.axvline(x=5, linestyle=(0, (1, 10)), color='k', markersize=2)
    plt.xticks(x_tic)
    plt.xlim([0, 20])
    plt.ylim([0.205, 0.25])
    plt.xlabel('$N_\mathrm{epochs}$')
    plt.ylabel('Fourier loss $\ell_\mathrm{F,cos}$')
    # plt.title('Accuracy comparison after different epochs of training w/o regularization, with Frobenius and cosine')
    plt.legend([p_inf, p_c, p_0],['Fourier loss only, $\eta = 10^{-4}$', 'Fourier & adv. loss, $\eta = 10^{-5}$', 'adv. loss only, $\eta = 10^{-6}$'], loc='upper left')

    plt.savefig('FL.pdf')

    plt.show()

    print("finished")

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

if __name__ == '__main__':
    main()
