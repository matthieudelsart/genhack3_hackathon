import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import Image
import imageio
import numpy as np
from scipy import stats
import tensorflow as tf
from matplotlib.gridspec import GridSpec



# Gaussian

def viz_gaussian_train(uniform, gaussian_data):
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    uniform_idx_sorted = uniform.argsort(axis=0)
    uniform_sorted, data_sorted = uniform[uniform_idx_sorted].ravel(), gaussian_data[uniform_idx_sorted].ravel()

    ax1.plot(uniform_sorted, data_sorted)
    ax1.set_title("Quantile function $\Phi^{-1}(u)$")

    sns.histplot(gaussian_data, ax=ax2, kde=True, stat='density', label="$\mu$={:.2f}, $\sigma^2$={:.2f}".format(gaussian_data.mean(),                                                                         gaussian_data.std()))
    ax2.set_title("Empirical distribution")

    ax2.legend()
    plt.tight_layout()
    sns.despine()
    return

def viz_gaussian_gan(uniform, gaussian_data, generator, discriminator,  list_loss_G, list_loss_D, epoch):
    dict_arguments = locals()
    dict_viz_functions = {"unidim": viz_gaussian_gan_unidim, "multidim": viz_gaussian_gan_multidim}
    noise_dim = uniform.shape[1]

    if noise_dim == 1:
        dict_viz_functions["unidim"](**dict_arguments)
    else:
        dict_viz_functions["multidim"](**dict_arguments)
    return

def viz_gaussian_gan_unidim(uniform, gaussian_data, generator, discriminator,  list_loss_G, list_loss_D, epoch):
    """
    Visualization of statistics
    Parameters
    ----------
    uniform: input data
    gaussian_data: real or simulated data
    generator: model
    discriminator: model
    epoch: visualization at a specific epoch
    -------
    """
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # sort data with uniform
    uniform_sorted=tf.sort(uniform, axis=0)
    generated_data = generator(uniform_sorted)

    # Function inverse cdf and generator
    data_inverse_cdf = stats.norm.ppf(uniform_sorted).astype(np.float32)
    ax1.plot(uniform_sorted, data_inverse_cdf, label="réelle")
    ax1.plot(uniform_sorted, generated_data, label="GAN")
    ax1.set_title("Quantile function and GAN generator")
    ax1.legend()

    # PDF
    sns.histplot(gaussian_data, stat='density',ax=ax2, kde=True,  palette=["#1f77b4"],
        label="réelle ($\mu$={:.2f}, $\sigma$={:.2f})".format(
        gaussian_data.mean(), gaussian_data.std()))
    sns.histplot(generated_data.numpy(), stat='density',ax=ax2, kde=True, palette=["orange"],
        label="GAN ($\mu$={:.2f}, $\sigma$={:.2f})".format(
        generated_data.numpy().mean(), generated_data.numpy().std()))
    ax2.set_title("Distribution")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0., 0.41)
    
    # QQ-plot
    data_sorted = np.sort(gaussian_data, axis=0)
    generated_data_sorted = tf.sort(generated_data, axis=0)
    ax3.scatter(data_sorted, generated_data_sorted)
    ax3.set_xlabel("real data")
    ax3.set_ylabel("simulated data")
    ax3.set_ylim(-5., 5.)
    ax3.set_title("QQ-plot")

    # Discriminator Score
    scores = tf.nn.sigmoid(discriminator(data_sorted))

    ax4.plot(data_sorted, scores)
    ax4.fill_between(data_sorted.ravel(), 0, scores.numpy().ravel(), alpha=.3)
    ax4.axhline(y=0.5, color="k")
    ax4.set_xlabel("real Gaussian variables")
    ax4.set_ylabel("Discriminator score")
    ax4.set_title("Discriminator")
    ax4.set_ylim(0, 1.01)

    # # Loss functions
    epochs = np.arange(len(list_loss_G))
    if epoch == 0:
        ax5.scatter(epochs * 10, list_loss_G, label="Loss Générator")
        ax5.scatter(epochs * 10, list_loss_D, label="Loss Discriminator")
    else:
        ax5.plot(epochs * 10, list_loss_G, label="Loss Générator")
        ax5.plot(epochs * 10, list_loss_D, label="Loss Discriminator")
    ax5.set_xlabel("epochs")
    ax5.set_title("Loss functions")
    ax5.legend()

    fig.suptitle("GAN training (epoch={})".format(epoch), size=14, y=1.)
    plt.tight_layout()
    sns.despine()
    return

def viz_gaussian_gan_multidim(uniform, gaussian_data, generator, discriminator, list_loss_G, list_loss_D, epoch):
    fig = plt.figure(figsize=(15, 10))

    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    generated_data = generator(uniform)

    # PDF
    sns.histplot(gaussian_data, stat="density", ax=ax1, kde=True, palette=["#1f77b4"], 
        label="réelle ($\mu$={:.2f}, $\sigma$={:.2f})".format(
        gaussian_data.mean(), gaussian_data.std()))
    sns.histplot(generated_data.numpy(), stat="density", ax=ax1, kde=True, palette=["orange"], 
        label="GAN ($\mu$={:.2f}, $\sigma$={:.2f})".format(
        generated_data.numpy().mean(), generated_data.numpy().std()))
    ax1.set_title("Distribution")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 0.41)


    # Discriminator Score
    data_sorted = np.sort(gaussian_data, axis=0)
    scores = tf.nn.sigmoid(discriminator(data_sorted))

    ax2.plot(data_sorted, scores)
    ax2.fill_between(data_sorted.ravel(), 0, scores.numpy().ravel(), alpha=.3)
    ax2.axhline(y=0.5, color="k")
    ax2.set_xlabel("Real Gaussian variables")
    ax2.set_ylabel("Discriminator score")
    ax2.set_title("Discriminator")
    ax2.set_ylim(0, 1.01)

    plt.tight_layout()
    sns.despine()
    fig.suptitle("GAN training(epoch={})".format(epoch), size=14, y=1.)

    # # Loss functions
    epochs = np.arange(len(list_loss_G))
    if epoch == 0:
        ax3.scatter(epochs * 10, list_loss_G, label="Loss Générator")
        ax3.scatter(epochs * 10, list_loss_D, label="Loss Discriminator")
    else:
        ax3.plot(epochs * 10, list_loss_G, label="Loss Générator")
        ax3.plot(epochs * 10, list_loss_D, label="Loss Discriminator")
    ax3.set_xlabel("epochs")
    ax3.set_title("Loss functions")
    ax3.legend()

    fig.suptitle("GAN Training (epoch={})".format(epoch), size=14, y=1.)
    plt.tight_layout()
    sns.despine()
    return





