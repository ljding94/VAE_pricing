import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../pricing")))
from american_put_pricer import read_vol_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_latent_space():
    """
    Plot the latent space of the VAE model.
    """
    print("Plotting latent space...")
    data_path = "../data_process/data_pack/latent_mus_logvars.csv"

    df = pd.read_csv(data_path)
    mus = df[[col for col in df.columns if col.startswith("mu")]].values
    logvars = df[[col for col in df.columns if col.startswith("logvar")]].values

    fig = plt.figure(figsize=(10 / 3 * 1.0, 10 / 3 * 0.7))

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    # Use a colormap to indicate the index of mu and logvars
    num_vars = mus.shape[1]
    cmap = plt.get_cmap("viridis", num_vars)
    colors = [cmap(i) for i in range(num_vars)]

    for i in range(num_vars):
        ax1.hist(mus[:, i], bins=30, density=True, lw=1, histtype="step", color=colors[i], label=rf"$\mu_{{{i+1}}}$")
        ax2.hist(logvars[:, i], bins=30, density=True, lw=1, histtype="step", color=colors[i], label=rf"$2\log s_{{{i+1}}}$")

    ax1.set_xlabel(r"$\mu$", fontsize=9, labelpad=-2)
    ax1.set_ylabel(r"$P(\mu)$", fontsize=9, labelpad=0)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)
    ax1.set_xlim(-3, 4)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    # Create a colorbar for ax1
    cax1 = fig.add_axes([0.43, 0.66, 0.015, 0.25])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_vars - 1))
    sm.set_array([])
    cbar1 = plt.colorbar(sm, cax=cax1, orientation="vertical")
    # cbar1.ax.xaxis.set_ticks_position("top")
    # cbar1.ax.xaxis.set_label_position("top")
    cbar1.ax.set_title("Index", fontsize=7, pad=2)
    cbar1.ax.set_yticks([0, 2, 4, 6, 8])
    cbar1.ax.set_yticklabels([f"${i}$" for i in [0, 2, 4, 6, 8]])
    # cbar1.set_label(r"$\mu$ index", fontsize=9, labelpad=2)
    cbar1.ax.yaxis.set_ticks_position("left")
    cbar1.ax.yaxis.set_label_position("left")
    cbar1.ax.tick_params(axis="both", which="both", direction="in", labelsize=7)

    ax2.set_xlabel(r"$2\log s$", fontsize=9, labelpad=-2)
    ax2.set_ylabel(r"$P(2\log s)$", fontsize=9, labelpad=0)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)
    ax2.set_xlim(-10, -2)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    corr = np.corrcoef(mus, rowvar=False)
    corr = np.abs(corr)
    im = ax3.imshow(corr, cmap="coolwarm", vmin=0, vmax=1)
    # ax3.set_title(r"$|Cov(\mu_i,\mu_j)|$", fontsize=9)
    ax3.set_xticks(np.arange(mus.shape[1]))
    ax3.set_yticks(np.arange(mus.shape[1]))
    ax3.set_xticklabels([f"${i}$" for i in range(mus.shape[1])])
    ax3.set_yticklabels([f"${i}$" for i in range(mus.shape[1])])
    ax3.set_xlabel(r"$\mu$", fontsize=9, labelpad=-2)

    # Create an inset axes for the colorbar at the top of ax3
    cax = fig.add_axes([0.65, 0.87, 0.3, 0.02])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label(r"$|Corr(\mu_i,\mu_j)|$", fontsize=9, labelpad=2)
    cbar.ax.tick_params(axis="both", which="both", direction="in", labelsize=7, pad=0)

    ax3.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)

    # add annotation
    ax1.text(0.84, 0.15, r"$(a)$", transform=ax1.transAxes, fontsize=9)
    ax2.text(0.84, 0.15, r"$(b)$", transform=ax2.transAxes, fontsize=9)
    ax3.text(0.8, -0.17, r"$(c)$", transform=ax3.transAxes, fontsize=9)

    plt.tight_layout(pad=0.1)
    plt.savefig("./figures/latent_variables.png", dpi=300)
    plt.savefig("./figures/latent_variables.pdf", dpi=500, format="pdf")

    plt.show()


def plot_loss_curves():
    # TODO: implement, loss curve, VAE + 3 pricer
    fig = plt.figure(figsize=(10 / 3 * 1.0, 10 / 3 * 1.0))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)


    # 1 vae loss curve
    vae_loss_path_train = "../data_process/data_pack/vae_train_losses.npy"
    vae_loss_path_test = "../data_process/data_pack/vae_test_losses.npy"

    train_losses = np.load(vae_loss_path_train)
    test_losses = np.load(vae_loss_path_test)

    epochs = np.arange(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Train Loss", color="blue", lw=1, alpha=0.7)
    ax1.plot(epochs, test_losses, label="Test Loss", color="orange", lw=1, alpha=0.7)
    ax1.set_xlabel("Epoch", fontsize=9, labelpad=0)
    ax1.set_ylabel("VAE Loss", fontsize=9, labelpad=0)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.legend(title = "VAE", fontsize=9, loc="upper right", ncol=1, columnspacing=0.5, labelspacing=0.1, handlelength=0.5, handletextpad=0.2, frameon=False,title_fontsize=9)
    ax1.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)


    # 2 American put pricer loss curve
    american_put_loss_path_train = "../data_process/data_pack/AmericanPut_pricer_train_losses.npy"
    american_put_loss_path_test = "../data_process/data_pack/AmericanPut_pricer_test_losses.npy"

    # 3 Asian call pricer loss curve
    asian_call_loss_path_train = "../data_process/data_pack/AsianCall_pricer_train_losses.npy"
    asian_call_loss_path_test = "../data_process/data_pack/AsianCall_pricer_test_losses.npy"

    # 4 Asian put pricer loss curve
    asian_put_loss_path_train = "../data_process/data_pack/AsianPut_pricer_train_losses.npy"
    asian_put_loss_path_test = "../data_process/data_pack/AsianPut_pricer_test_losses.npy"


    all_pricer_loss_train = [american_put_loss_path_train, asian_call_loss_path_train, asian_put_loss_path_train]
    all_pricer_loss_test = [american_put_loss_path_test, asian_call_loss_path_test, asian_put_loss_path_test]

    all_pricer_axes = [ax2, ax3, ax4]
    legend_titles = ["American Put", "Asian Call", "Asian Put"]

    for i in range(3):
        train_losses = np.load(all_pricer_loss_train[i])
        test_losses = np.load(all_pricer_loss_test[i])
        epochs = np.arange(1, len(train_losses) + 1)
        all_pricer_axes[i].plot(epochs, train_losses, label="Train Loss", color="red", lw=1, alpha=0.7)
        all_pricer_axes[i].plot(epochs, test_losses, label="Test Loss", color="forestgreen", lw=1, alpha=0.7)
        all_pricer_axes[i].set_xlabel("Epoch", fontsize=9, labelpad=0)
        all_pricer_axes[i].set_ylabel("Pricer Loss", fontsize=9, labelpad=0)
        all_pricer_axes[i].set_yscale("log")
        all_pricer_axes[i].set_xscale("log")
        all_pricer_axes[i].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        all_pricer_axes[i].legend(title=legend_titles[i], fontsize=9, loc="upper right", ncol=1, columnspacing=0.5, labelspacing=0.1, handlelength=0.5, handletextpad=0.2, frameon=False, title_fontsize=9)
        all_pricer_axes[i].tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)


    # add annotation
    ax1.text(0.075, 0.1, r"$(a)$", transform=ax1.transAxes, fontsize=9)
    ax2.text(0.075, 0.1, r"$(b)$", transform=ax2.transAxes, fontsize=9)
    ax3.text(0.075, 0.1, r"$(c)$", transform=ax3.transAxes, fontsize=9)
    ax4.text(0.075, 0.1, r"$(d)$", transform=ax4.transAxes, fontsize=9)

    plt.tight_layout(pad=0.1)
    plt.savefig("./figures/loss_curves.png", dpi=300)
    plt.savefig("./figures/loss_curves.pdf", dpi=500, format="pdf")
    plt.show()






def plot_vol_surface_reconstruction():
    fig = plt.figure(figsize=(10 / 3 * 1.0, 10 / 3 * 1.0))

    axs = fig.subplots(3, 3, sharex=True, sharey=True)

    vae_data_path = "../data_process/data_pack/vae_quote_date_reconstructions.npz"
    if not os.path.exists(vae_data_path):
        print(f"Error: {vae_data_path} not found. Please run the VAE reconstruction first.")
        return 0
    data = np.load(vae_data_path)
    quote_dates = data["quote_dates"]
    k_grid = data["k_grid"]
    T_grid = data["T_grid"]
    vol_surfaces = data["vol_surfaces"]  # shape (3, num_k, num_T)
    recon_vol_surfaces = data["recon_vol_surfaces"]  # shape (3, num_k, num_T)

    T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)
    # axs[0,:]. original vol surface
    for i in range(3):
        ax = axs[0, i]
        ax.pcolormesh(k_mesh, T_mesh, vol_surfaces[i], shading="auto", cmap="rainbow", vmin=0.09, vmax=0.66, linewidth=0, rasterized=True)
        # ax.set_title(f"Original: {quote_dates[i]}", fontsize=9)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=7, right=True, top=True, labelbottom=False, labelleft=False)
        # if i == 0:
        #    ax.set_ylabel(r"$T$", fontsize=9, labelpad=-1)
        #    ax.yaxis.set_major_locator(plt.MultipleLocator(0.4))
        #    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    for i in range(3):
        ax = axs[1, i]
        ax.pcolormesh(k_mesh, T_mesh, recon_vol_surfaces[i], shading="auto", cmap="rainbow", vmin=0.09, vmax=0.66, linewidth=0, rasterized=True)
        # ax.set_title(f"Reconstruction: {quote_dates[i]}", fontsize=9)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=7, right=True, top=True, labelbottom=False, labelleft=False)
        # if i == 0:
        #    ax.set_ylabel(r"$T$", fontsize=9, labelpad=-1)
        #    ax.yaxis.set_major_locator(plt.MultipleLocator(0.4))
        #    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    for i in range(3):
        ax = axs[2, i]
        diff = recon_vol_surfaces[i] - vol_surfaces[i]
        ax.pcolormesh(k_mesh, T_mesh, diff, shading="auto", cmap="bwr", vmin=-0.02, vmax=0.02, linewidth=0, rasterized=True)
        # ax.set_title(f"Difference: {quote_dates[i]}", fontsize=9)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=7, right=True, top=True, labelbottom=True, labelleft=False)

    for i in range(3):
        axs[i, 0].set_ylabel(r"$T$", fontsize=9, labelpad=0)
        axs[i, 0].tick_params(labelleft=True)
        axs[2, i].set_xlabel(r"$k$", fontsize=9, labelpad=0)
        axs[2, i].tick_params(labelbottom=True)

        axs[0, i].set_title(f"{quote_dates[i]}", fontsize=7, pad=0)

    axs[0, 0].yaxis.set_major_locator(plt.MultipleLocator(0.5))
    axs[0, 0].yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    axs[0, 0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
    axs[0, 0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # Add a colorbar for the original/reconstruction plots
    cax1 = fig.add_axes([0.9, 0.5, 0.015, 0.4])  # [left, bottom, width, height]
    cbar1 = plt.colorbar(axs[0, 0].collections[0], cax=cax1, orientation="vertical")
    cbar1.ax.set_title(r"$\sigma_{BS}$", fontsize=9, pad=2)
    cbar1.ax.title.set_position((2, 1.02))  # (x, y) coordinates relative to colorbar axes
    cbar1.ax.tick_params(axis="both", which="both", direction="in", labelsize=7)

    # Add a colorbar for the difference plots
    cax2 = fig.add_axes([0.9, 0.1, 0.015, 0.2])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(axs[2, 0].collections[0], cax=cax2, orientation="vertical")
    cbar2.ax.set_title(r"$\Delta \sigma_{BS}$", fontsize=9)
    cbar2.ax.title.set_position((2.5, 1.02))
    cbar2.ax.tick_params(axis="both", which="both", direction="in", labelsize=7, pad=0, rotation=45)

    # Add annotations
    axs[0, 0].text(-0.4, 0.7, r"orig.", transform=axs[0, 0].transAxes, fontsize=9, ha="left")
    axs[1, 0].text(-0.4, 0.7, r"reco.", transform=axs[1, 0].transAxes, fontsize=9, ha="left")
    axs[2, 0].text(-0.4, 0.7, r"diff.", transform=axs[2, 0].transAxes, fontsize=9, ha="left")

    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$", r"$(g)$", r"$(h)$", r"$(i)$"]
    for row in range(3):
        for col in range(3):
            axs[row, col].text(0.7, 0.15, annos[3 * row + col], transform=axs[row, col].transAxes, fontsize=9)

    plt.tight_layout(pad=0.2, rect=[0.0, 0, 0.9, 1])  # leave space on the right for colorbars
    plt.savefig("./figures/vae_vol_reconstructions.png", dpi=300)
    plt.savefig("./figures/vae_vol_reconstructions.pdf", dpi=500, format="pdf")

    plt.show()


def plot_price_prediction_AmericanPut():

    folder = "../data_process/data_pack"

    # Load the first available price prediction file
    price_data = np.load(f"{folder}/AmericanPut_price_predictions.npz")

    # Print header of price_data
    print("Keys in price_data:")
    for key in price_data.keys():
        print(f"  {key}: shape {price_data[key].shape}")

    predicted_prices = price_data["predicted_test_price"]  # shape (N, )
    target_prices = price_data["test_price"]  # shape (N, )
    pricing_params = price_data["test_pricing_params"]  # shape (N, 2)

    fig = plt.figure(figsize=(10 / 3 * 1.0, 10 / 3 * 0.5))

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    ax1.scatter(target_prices, predicted_prices, s=1, color="royalblue")

    # Plot y=x line
    min_price = min(target_prices.min(), predicted_prices.min())
    max_price = max(target_prices.max(), predicted_prices.max())
    # ax1.plot([min_price, max_price], [min_price, max_price], color="red", linestyle="--", lw=0.5)
    ax1.set_xlim(min_price, max_price)
    ax1.set_ylim(min_price, max_price)

    ax1.set_xlabel("Ground Truth", fontsize=9, labelpad=0)
    ax1.set_ylabel("Predicted Price", fontsize=9, labelpad=0)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)

    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Extract K and T values for the test set
    test_K = pricing_params[:, 0]
    test_T = pricing_params[:, 1]
    test_k = np.log(test_K)

    # Calculate absolute error
    abs_error = np.abs(predicted_prices - target_prices)
    abs_error_max = abs_error.max()
    error = predicted_prices - target_prices

    # Create scatter plot in K-T coordinate with absolute error as color
    scatter = ax2.scatter(test_k, test_T, c=error, s=abs_error * 100, cmap="coolwarm", vmin=-abs_error_max, vmax=abs_error_max)

    ax2.set_xlabel("k", fontsize=9, labelpad=0)
    ax2.set_ylabel("T", fontsize=9, labelpad=-2)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.3))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.15))

    # Add colorbar for absolute error
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.6, orientation="horizontal", pad=0.02, location="top")
    cbar.set_label("Err", fontsize=9, labelpad=-5, loc="left")
    cbar.ax.tick_params(labelsize=7, which="both", direction="in", pad=0)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.xaxis.set_label_coords(-0.4, 1)

    """
    ax2.scatter(target_prices, predicted_prices - target_prices, s=1, alpha=0.5, color="green")
    ax2.axhline(0, color="black", linestyle="--", lw=1)
    ax2.set_xlim(min_price, max_price)
    ax2.set_ylim(-0.15, 0.15)

    ax2.set_xlabel("Ground Truth", fontsize=9, labelpad=0)
    ax2.set_ylabel("Error", fontsize=9, labelpad=-5)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)

    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    """
    ax1.text(0.1,0.8, "American Put", transform=ax1.transAxes, fontsize=9)
    # add annotation
    ax1.text(0.8, 0.15, r"$(a)$", transform=ax1.transAxes, fontsize=9)
    ax2.text(0.8, 0.15, r"$(b)$", transform=ax2.transAxes, fontsize=9)

    plt.tight_layout(pad=0.1)
    plt.savefig("./figures/AmericanPut_price_prediction.png", dpi=300)
    plt.savefig("./figures/AmericanPut_price_prediction.pdf", dpi=500, format="pdf")

    plt.show()


def plot_price_prediction_AsianOpt():
    folder = "../data_process/data_pack"

    fig = plt.figure(figsize=(10 / 3 * 1.0, 10 / 3 * 1))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(224)

    # Load the first available price prediction file
    for product in ["AsianCall", "AsianPut"]:
        price_data = np.load(f"{folder}/{product}_price_predictions.npz")
        axa, axb = (ax1, ax2) if product == "AsianCall" else (ax3, ax4)

        # Print header of price_data
        print("Keys in price_data:")
        for key in price_data.keys():
            print(f"  {key}: shape {price_data[key].shape}")

        predicted_prices = price_data["predicted_test_price"]  # shape (N, )
        target_prices = price_data["test_price"]  # shape (N, )
        pricing_params = price_data["test_pricing_params"]  # shape (N, 2)

        axa.scatter(target_prices, predicted_prices, s=1, color="royalblue")

        # Plot y=x line
        min_price = min(target_prices.min(), predicted_prices.min())
        max_price = max(target_prices.max(), predicted_prices.max())
        # axa.plot([min_price, max_price], [min_price, max_price], color="red", linestyle="--", lw=0.5)
        axa.set_xlim(min_price, max_price)
        axa.set_ylim(min_price, max_price)

        axa.set_xlabel("Ground Truth", fontsize=9, labelpad=0)
        axa.set_ylabel("Predicted Price", fontsize=9, labelpad=0)
        axa.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axa.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)

        axa.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        axa.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        axa.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        axa.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

        # Extract K and T values for the test set
        test_K = pricing_params[:, 0]
        test_T = pricing_params[:, 1]
        test_k = np.log(test_K)

        # Calculate absolute error
        abs_error = np.abs(predicted_prices - target_prices)
        abs_error_max = abs_error.max()
        error = predicted_prices - target_prices

        if product == "AsianCall":
            AC_error = error
            print(f"Asian Call Price Prediction Error Statistics:")
        elif product == "AsianPut":
            AP_error = error
            print(f"Asian Put Price Prediction Error Statistics:")

        # Create scatter plot in K-T coordinate with absolute error as color
        scatter = axb.scatter(test_k, test_T, c=error, s=abs_error * 100, cmap="coolwarm", vmin=-abs_error_max, vmax=abs_error_max)

        axb.set_xlabel("k", fontsize=9, labelpad=0)
        axb.set_ylabel("T", fontsize=9, labelpad=-2)
        axb.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axb.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=7)
        axb.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        axb.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        axb.yaxis.set_major_locator(plt.MultipleLocator(0.3))
        axb.yaxis.set_minor_locator(plt.MultipleLocator(0.15))

        # Add colorbar for absolute error
        cbar = plt.colorbar(scatter, ax=axb, shrink=0.6, orientation="horizontal", pad=0.02, location="top")
        cbar.set_label("Err", fontsize=9, labelpad=-5, loc="left")
        cbar.ax.tick_params(labelsize=7, which="both", direction="in", pad=0)
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.xaxis.set_label_coords(-0.4, 1)
        cbar.ax.xaxis.set_major_locator(plt.MultipleLocator(0.03))
        cbar.ax.xaxis.set_minor_locator(plt.MultipleLocator(0.015))

    ax1.text(0.2,0.8, "Asian Call", transform=ax1.transAxes, fontsize=9)
    ax3.text(0.2,0.8, "Asian Put", transform=ax3.transAxes, fontsize=9)

    # add annotation
    ax1.text(0.8, 0.15, r"$(a)$", transform=ax1.transAxes, fontsize=9)
    ax2.text(0.8, 0.15, r"$(b)$", transform=ax2.transAxes, fontsize=9)
    ax3.text(0.8, 0.15, r"$(c)$", transform=ax3.transAxes, fontsize=9)
    ax4.text(0.8, 0.15, r"$(d)$", transform=ax4.transAxes, fontsize=9)

    plt.tight_layout(pad=0.1)
    plt.savefig("./figures/AsianOpt_price_prediction.png", dpi=300)
    plt.savefig("./figures/AsianOpt_price_prediction.pdf", dpi=500, format="pdf")

    plt.show()
