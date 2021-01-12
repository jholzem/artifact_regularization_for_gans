import numpy as np
import matplotlib.pyplot as plt

def main():
    acc_0 = np.genfromtxt('models/pretrain/0_cos_1e-6_accuracies.txt', delimiter=',')
    acc_c = np.genfromtxt('models/pretrain/1e3_cos_1e-5_accuracies.txt', delimiter=',')
    acc_inf = np.genfromtxt('models/pretrain/inf_cos_1e-4_accuracies.txt', delimiter=',')

    acc_inf = acc_inf[0:20]

    x_ax = list(range(1,21))
    x_tic = list(range(0,21,5))

    p_0, = plt.plot(x_ax, acc_0, 'r', markersize=2)
    p_2, = plt.plot(x_ax, acc_c, 'g', markersize=2)
    p_c, = plt.plot(x_ax, acc_inf, 'b', markersize=2)
    #p_0, = plt.plot(x_ax[2:21], acc_0[2:21], linestyle='dashed', color='b', markersize=2)
    #p_2, = plt.plot(x_ax[2:21], acc_c[2:21], linestyle='dashed', color='r', markersize=2)
    #p_c, = plt.plot(x_ax[2:21], acc_inf[2:21], linestyle='dashed', color='g', markersize=2)

    plt.grid()
    # plt.axvline(x=5, linestyle=(0, (1, 10)), color='k', markersize=2)
    plt.xticks(x_tic)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy of being detected by CNN-detection')
    # plt.title('Accuracy comparison after different epochs of training w/o regularization, with Frobenius and cosine')
    plt.legend(['Adv. loss only', 'Fourier & adv. loss', 'Fourier loss only'], loc='lower left')
    plt.show()

if __name__ == '__main__':
    main()
