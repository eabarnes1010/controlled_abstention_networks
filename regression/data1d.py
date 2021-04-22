import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "22 April 2021"


# -----------------------------------------------------------------------------
def get_data(EXPINFO, to_plot=False):

    # make the data
    slope = EXPINFO['slope']
    yint = EXPINFO['yint']
    noise = EXPINFO['noise']
    x_sigma = EXPINFO['x_sigma']
    n_samples = EXPINFO['n_samples']

    y_data = []
    x_data = []
    tr_data = []

    for i in np.arange(0, len(slope)):
        if i == 0:
            x = np.random.normal(4., x_sigma[i], n_samples[i])
            tr = np.zeros(np.shape(x))
        else:
            x = np.random.normal(0, x_sigma[i], n_samples[i])
            tr = np.ones(np.shape(x))
        y = slope[i] * x + yint[i] + np.random.normal(0, noise[i], n_samples[i])

        x_data = np.append(x_data, x)
        y_data = np.append(y_data, y)
        tr_data = np.append(tr_data, tr)

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    tr_data = np.asarray(tr_data)
    print('\nnoise fraction = ' + str(np.round(100*(1.-len(x)/len(x_data)))) + '%')

    if to_plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title('all data')
        plt.plot(
            x_data,
            y_data,
            '.',
            color='tab:blue',
            markeredgecolor=None,
            linewidth=.1,
            markersize=.2
        )
        reg = stats.linregress(x=x_data, y=y_data)
        plt.plot(
            x_data,
            x_data*reg.slope+reg.intercept,
            '--',
            color='tab:gray',
            linewidth=.5,
            label='OLS best fit'
        )
        plt.legend()

    # mix the data
    shuffler = np.random.permutation(len(x_data))
    x_data = x_data[shuffler]
    y_data = y_data[shuffler]
    tr_data = tr_data[shuffler]

    # grab training and validation
    n = np.shape(x_data)[0]
    n_val = 1000
    n_test = 1000
    n_train = n - (n_val + n_test)

    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    tr_train = tr_data[:n_train]

    x_val = x_data[n_train:n_train+n_val]
    y_val = y_data[n_train:n_train+n_val]
    tr_val = tr_data[n_train:n_train+n_val]

    x_test = x_data[n_train+n_val:n_train+n_val+n_test]
    y_test = y_data[n_train+n_val:n_train+n_val+n_test]
    tr_test = tr_data[n_train+n_val:n_train+n_val+n_test]

    print('training samples shapes = ' + str(np.shape(x_train)))
    print('validation samples shapes = ' + str(np.shape(x_val)))
    print('testing samples shapes = ' + str(np.shape(x_test)))

    # standardize the data
    print('not actually standardizing the data\n')
    xmean = 0.      # np.nanmean(x_train)
    xstd = 1.       # np.nanstd(x_train)
    X_train_std = (x_train[:, np.newaxis] - xmean)/xstd
    X_val_std = (x_val[:, np.newaxis] - xmean)/xstd
    X_test_std = (x_test[:, np.newaxis] - xmean)/xstd

    # create one hot vectors
    N_CLASSES = EXPINFO['numClasses']
    onehot_train = np.zeros((len(y_train), N_CLASSES))
    onehot_train[:, 0] = y_train
    onehot_val = np.zeros((len(y_val), N_CLASSES))
    onehot_val[:, 0] = y_val
    onehot_test = np.zeros((len(y_test), N_CLASSES))
    onehot_test[:, 0] = y_test

    if to_plot:
        plt.subplot(1, 2, 2)
        plt.plot(x_train, y_train, '.', markersize=.5, label='training')
        plt.plot(x_val, y_val, 'o', markerfacecolor='None', markersize=.5, label='validation')
        plt.title('training and validation data')
        plt.legend()
        plt.show()

    return(
        X_train_std,
        onehot_train,
        X_val_std,
        onehot_val,
        X_test_std,
        onehot_test,
        xmean,
        xstd,
        tr_train,
        tr_val,
        tr_test
    )
