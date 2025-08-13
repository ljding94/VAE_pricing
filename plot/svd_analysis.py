import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../pricing")))
from american_put_pricer import read_vol_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_svd_analysis():
    '''
    3X3 grid of plots:
    1 SVD value vs rank
    8 singular vectors as heatmaps
    '''
    volmin=0.09
    volmax=0.66

    # 1 read in all vol data
    folder = "../data_process/data_pack"
    label = "post_vol_"
    train_vol_data_path = f"../data_process/data_pack/{label}grid_train.npz"
    # 1. read all vol data
    data, quote_dates, vol_surfaces, K_grid, T_grid = read_vol_data(train_vol_data_path, label)
    train_vol_surfaces = vol_surfaces

    test_vol_data_path = f"../data_process/data_pack/{label}grid_test.npz"
    data, quote_dates, vol_surfaces, K_grid, T_grid = read_vol_data(train_vol_data_path, label)
    test_vol_surfaces = vol_surfaces

    k_grid = data["k_grid"]
    print(f"train_vol_surfaces shape: {train_vol_surfaces.shape}")
    print(f"test_vol_surfaces shape: {test_vol_surfaces.shape}")

    all_vol_surfaces = np.concatenate([train_vol_surfaces, test_vol_surfaces], axis=0)
    print("all_vol_surfaces.shape", all_vol_surfaces.shape)

    all_vol_surfaces_flatten = all_vol_surfaces.reshape(all_vol_surfaces.shape[0], -1)
    print("all_vol_surfaces.shape", all_vol_surfaces.shape)
    print("all_vol_surfaces_flatten.shape", all_vol_surfaces_flatten.shape)


    fig = plt.figure(figsize=(10 / 3 * 1.0, 10 / 3 * 1.0))

    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333, sharex=ax2, sharey=ax2)
    ax4 = fig.add_subplot(334, sharex=ax2, sharey=ax2)
    ax5 = fig.add_subplot(335, sharex=ax2, sharey=ax2)
    ax6 = fig.add_subplot(336, sharex=ax2, sharey=ax2)
    ax7 = fig.add_subplot(337, sharex=ax2, sharey=ax2)
    ax8 = fig.add_subplot(338, sharex=ax2, sharey=ax2)
    ax9 = fig.add_subplot(339, sharex=ax2, sharey=ax2)

    vol_axs = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]


    # Perform SVD
    U, s, Vt = np.linalg.svd(all_vol_surfaces_flatten, full_matrices=False)

    # Plot singular values
    idx = np.arange(1, len(s)+1)
    ax1.plot(idx, s, "o", ms=2, mfc="none")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.set_ticks_position('top')
    ax1.set_xlabel("SVR", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$\Sigma$", fontsize=9, labelpad=0)
    ax1.tick_params(which="both", direction="in", labelsize=7, pad=0)

    # Unflatten the grid for plotting
    for i in range(8):
        print(f"Vt[{i}] min: {Vt[i].min():.4f}, max: {Vt[i].max():.4f}")
    overall_min = Vt[:8].min()
    overall_max = Vt[:8].max()
    print(f"Overall min: {overall_min:.4f}, Overall max: {overall_max:.4f}")
    vol_min=-0.2
    vol_max=0.2

    for i in range(len(vol_axs)):
        # Vt contains the right singular vectors (modes of variation in volatility surfaces)
        vec = Vt[i]
        # Reshape the vector back to the 2D grid
        vec_2d = vec.reshape(len(k_grid), len(T_grid))

        # Use pcolormesh for more control and correct axis orientation
        T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)
        vol_axs[i].pcolormesh(k_mesh, T_mesh, vec_2d, cmap="rainbow", shading="auto", vmin=vol_min, vmax=vol_max, linewidth=0,rasterized=True)
        # Optionally add a colorbar for each subplot
        # plt.colorbar(img, ax=vol_axs[i], fraction=0.046, pad=0.04)
        vol_axs[i].tick_params(which="both", direction="in", labelsize=7, labelleft=False, labelbottom=False, right=True, top=True)
        if i in [2, 5]:
            vol_axs[i].set_ylabel(r"$T$", fontsize=9, labelpad=-1)
            vol_axs[i].tick_params(labelleft=True)
            vol_axs[i].yaxis.set_major_locator(plt.MultipleLocator(0.4))
            vol_axs[i].yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        if i in [5, 6, 7]:
            vol_axs[i].set_xlabel(r"$k$", fontsize=9, labelpad=0)
            vol_axs[i].tick_params(labelbottom=True)
            vol_axs[i].xaxis.set_major_locator(plt.MultipleLocator(0.2))
            vol_axs[i].xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # Add an inset colorbar above ax2 without shrinking the main plot
    img = ax2.collections[0]
    # You can use absolute location by specifying a new axes with fig.add_axes and giving [left, bottom, width, height] in figure coordinates (0-1).
    # Example: place the colorbar at the bottom of the figure, spanning 60% width and 10% height
    cax = fig.add_axes([0.5, 0.95, 0.3, 0.01])  # [left, bottom, width, height]
    cbar = plt.colorbar(img, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(which="both", direction="in", labelsize=7, pad=0)

    # add annotation in a for loop
    axs = [ax1] + vol_axs
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$", r"$(g)$", r"$(h)$", r"$(i)$"]
    for i, ax in enumerate(axs):
        ax.text(0.7, 0.8, annos[i], transform=ax.transAxes, fontsize=9)


    plt.tight_layout(pad=0.2)
    plt.savefig("./figures/vol_surface_svd_analysis.png", dpi=300)
    plt.savefig("./figures/vol_surface_svd_analysis.pdf", dpi=500, format='pdf')

    plt.show()

    return U, s, Vt
