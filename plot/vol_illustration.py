import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_vol_illustration():

    fig = plt.figure(figsize=(10 / 3 * 1.0, 10 / 3 * 1))

    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222, projection="3d")

    # Make ax3 and ax4 smaller by adjusting their position
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, sharey=ax3)


    # Shrink ax3 and ax4 by reducing their height and width
    for ax in [ax3, ax4]:
        box = ax.get_position()
        ax.set_position([box.x0 + 0.02, box.y0 + 0.02, box.width * 0.8, box.height * 0.8])



    folders = ["../optionsdx_data/2020", "../optionsdx_data/2023q4"]
    quote_dates = ["2020-03-10", "2023-11-29"]

    for i in range(2):
        folder = folders[i]
        quote_date = quote_dates[i]

        ax = ax1 if i == 0 else ax2
        ax.set_box_aspect([1, 1, 1.05])
        vol_csv_file = f"{folder}/volatility_surface_{quote_date}.csv"
        post_process_vol_file = f"{folder}/post_vol_grid_data_{quote_date}.npz"

        # get market vol data
        df_vol = pd.read_csv(vol_csv_file)

        # filter out YTE < 0.05
        df_vol = df_vol[df_vol["YTE"] >= 0.05]

        print(f"{quote_date} min vol: {df_vol['BS_VOL'].min():.4f}, max vol: {df_vol['BS_VOL'].max():.4f}")

        ax.scatter(df_vol["SPOT_LOG_MONEYNESS"], df_vol["YTE"], df_vol["BS_VOL"], c="k", marker=".", s=0.1, alpha=0.3)

        post_vol_data = np.load(post_process_vol_file)
        k_grid = post_vol_data["k_grid"]
        T_grid = post_vol_data["T_grid"]
        vol_grid = post_vol_data["vol_grid"]

        print(vol_grid.shape, k_grid.shape, T_grid.shape)

        T, K = np.meshgrid(T_grid, k_grid)
        ax.plot_surface(K, T, vol_grid, cmap="rainbow", alpha=0.6, edgecolor="none", vmin=0.09, vmax=0.66)

        print(f"{quote_date} vol_grid min: {vol_grid.min():.4f}, max: {vol_grid.max():.4f}")

        ax.set_xlabel(r"$k$", fontsize=9, labelpad=-10)
        ax.set_ylabel(r"$T$", fontsize=9, labelpad=-10)
        ax.set_zlabel(r"$\sigma_{BS}$", fontsize=9, labelpad=-8)
        ax.view_init(elev=25, azim=50)
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([0, 1.4])
        ax.set_zlim([0.1, 0.6])
        ax.tick_params(axis="x", which="both", labelsize=7, pad=-6)
        ax.tick_params(axis="y", which="both", labelsize=7, pad=-4)
        ax.tick_params(axis="z", which="both", labelsize=7, pad=-3)
        ax.text2D(0, 0.9, f"{quote_date}", transform=ax.transAxes, fontsize=7)

        ax.grid(True)

        ax2d = ax3 if i == 0 else ax4
        # 2d heat map

        # Use pcolormesh for more control and correct axis orientation
        T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)
        im = ax2d.pcolormesh(k_mesh, T_mesh, vol_grid, cmap="rainbow", shading="auto", vmin=0.09, vmax=0.66, linewidth=0, rasterized=True)

        #im = ax2d.imshow(vol_grid, extent=[T_grid[0], T_grid[-1], k_grid[0], k_grid[-1]], origin="lower", aspect="auto", cmap="rainbow", vmin=0.09, vmax=0.66)

        ax2d.tick_params(axis="both", which="both", direction="in", labelsize=7 , right=True, top=True)

    ax4.tick_params(labelleft=False)
    ax3.set_xlabel(r"$k$", fontsize=9, labelpad=0)
    ax4.set_xlabel(r"$k$", fontsize=9, labelpad=0)
    ax3.set_ylabel(r"$T$", fontsize=9, labelpad=-6)
    ax3.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax3.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(0.2))


    # add annotation
    ax1.text2D(0.8, 0.8, r"$(a)$", transform=ax1.transAxes, fontsize=9)
    ax2.text2D(0.8, 0.8, r"$(b)$", transform=ax2.transAxes, fontsize=9)
    ax3.text(0.8, 0.15, r"$(c)$", transform=ax3.transAxes, fontsize=9)
    ax4.text(0.8, 0.15, r"$(d)$", transform=ax4.transAxes, fontsize=9)


    # make some space
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("top", size="20%", pad=0.05)
    #cax3.set_visible(False)
    #cax3.set_frame_on(False)
    cax3.axis("off")

    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("top", size="20%", pad=0.05)
    cax4.axis("off")

    # Shrink the colorbar by adjusting its width using set_position
    ax34 = fig.add_subplot(212)
    ax34.axis("off")
    #divider = make_axes_locatable(ax34)
    cax = ax34.inset_axes([0.25, 0.82, 0.5, 0.03])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(which="both", direction="in", labelsize=7, pad=0, right=True, top=True)
    cbar.set_label(r"$\sigma_{BS}$", fontsize=9, labelpad=-8, loc='left')
    # Move the label lower and to the left of the colorbar
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.xaxis.set_label_coords(-0.175, 1)

    plt.tight_layout()

    plt.savefig("./figures/vol_illustration.png", dpi=300)
    plt.savefig("./figures/vol_illustration.pdf", dpi=500, format="pdf")

    plt.show()



def plot_single_vol_grid():

    fig = plt.figure(figsize=(10 / 3 * 0.4, 10 / 3 * 0.4))

    ax = fig.add_subplot(111)


    folder = "../optionsdx_data/2023q4"
    quote_date = "2023-11-29"

    post_process_vol_file = f"{folder}/post_vol_grid_data_{quote_date}.npz"
    post_vol_data = np.load(post_process_vol_file)
    k_grid = post_vol_data["k_grid"]
    T_grid = post_vol_data["T_grid"]
    vol_grid = post_vol_data["vol_grid"]

    print(vol_grid.shape, k_grid.shape, T_grid.shape)

    T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)
    ax.pcolormesh(k_mesh, T_mesh, vol_grid, cmap="rainbow", linewidth=0,rasterized=True) #, vmin=0.09, vmax=0.66)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.1, 0.9, r"$\sigma_{BS}(k,T)$", transform=ax.transAxes, fontsize=10, ha='left', va='top')

    plt.tight_layout(pad=0.1)
    plt.savefig("./figures/single_vol_grid.png", dpi=300)
    plt.savefig("./figures/single_vol_grid.pdf", dpi=500, format="pdf")

    plt.show()

