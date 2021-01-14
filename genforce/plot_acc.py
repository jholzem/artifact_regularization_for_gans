import numpy as np
import matplotlib.pyplot as plt

def main():

    baseline = 0.9015

    acc_0 = np.genfromtxt('models/pretrain/0_cos_1e-6_accuracies.txt', delimiter=',')
    acc_c = np.genfromtxt('models/pretrain/1e3_cos_1e-5_accuracies.txt', delimiter=',')
    acc_inf = np.genfromtxt('models/pretrain/inf_cos_1e-4_accuracies.txt', delimiter=',')

    acc_0 = np.append(np.array(baseline), acc_0[0:20])
    acc_c = np.append(np.array(baseline), acc_c[0:20])
    acc_inf = np.append(np.array(baseline), acc_inf[0:20])

    x_ax = list(range(0,21))
    x_tic = list(range(0,21,5))

    plt.figure(figsize=(7, 5.5))

    p_base, = plt.plot(np.arange(0,21), np.arange(0,21)*0 + baseline, 'k--')
    p_0, = plt.plot(x_ax, acc_0, 'r-o', markersize=5)
    p_c, = plt.plot(x_ax, acc_c, 'g-o', markersize=5)
    p_inf, = plt.plot(x_ax, acc_inf, 'b-o', markersize=5)
    #p_0, = plt.plot(x_ax[2:21], acc_0[2:21], linestyle='dashed', color='b', markersize=2)
    #p_2, = plt.plot(x_ax[2:21], acc_c[2:21], linestyle='dashed', color='r', markersize=2)
    #p_c, = plt.plot(x_ax[2:21], acc_inf[2:21], linestyle='dashed', color='g', markersize=2)

    plt.grid()
    # plt.axvline(x=5, linestyle=(0, (1, 10)), color='k', markersize=2)
    plt.xticks(x_tic)
    plt.xlim([0, 20])
    plt.xlabel('$N_\mathrm{epochs}$')
    plt.ylabel('detection accuracy')
    plt.ylim([0,1])
    plt.yticks(np.arange(0,1.1,0.1))
    # plt.title('Accuracy comparison after different epochs of training w/o regularization, with Frobenius and cosine')
    plt.legend([p_base, p_inf, p_c, p_0], ['baseline','Fourier loss only, $\eta = 10^{-4}$', 'Fourier & adv. loss, $\eta = 10^{-5}$', 'adv. loss only, $\eta = 10^{-6}$'], loc='lower left')

    plt.savefig('acc.pdf')

    plt.show()



    print('saved')

if __name__ == '__main__':
    main()
