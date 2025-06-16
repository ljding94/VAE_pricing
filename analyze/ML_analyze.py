import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_vol_surface_data(folder, years, bad_dates=[]):
    all_w_grid = []
    for year in years:
        df = pd.read_csv(f"{folder}/processed_data_{year}.csv", skipinitialspace=True)
        # get all unique quote dates
        all_quote_dates = df["QUOTE_DATE"].unique()

        for quote_date in all_quote_dates:
            # Skip if the date is in the bad_dates list
            if quote_date in bad_dates:
                print(f"Skipping bad date {quote_date}")
                continue
            vol_grid_filename = f"{folder}/{year}/grid_data_{quote_date}.csv"
            if not pd.io.common.file_exists(vol_grid_filename):
                print(f"File {vol_grid_filename} does not exist, skipping.")
                continue
            pd_vol_grid_data = pd.read_csv(vol_grid_filename)
            total_var_grid = pd_vol_grid_data["total_var_grid"].values
            k_grid = pd_vol_grid_data["k_grid"].values
            T_grid = pd_vol_grid_data["T_grid"].values
            all_w_grid.append(total_var_grid)
    return all_w_grid, k_grid, T_grid


def pack_vol_data_to_npz(folder, all_w_grid, k_grid, T_grid):
    all_w_grid = np.array(all_w_grid)
    # unflatten the grid
    all_w_grid, k_grid, T_grid = unflatten_grid(all_w_grid.flatten(), k_grid, T_grid)
    np.savez(f"{folder}/vol_surface_data.npz", all_w_grid=all_w_grid, k_grid=k_grid, T_grid=T_grid)
    print(f"Saved volatility surface data to {folder}/vol_surface_data.npz")


def profile_likelihood_dim(s, eps=1e-12):
    """
    Automatic dimensionality selection via the profile-likelihood method
    of Zhu & Ghodsi (2006).

    Parameters
    ----------
    s : (p,) array_like
        Singular values or eigenvalues in **descending** order.
    eps : float
        Floor placed on the pooled variance to avoid log(0).

    Returns
    -------
    q_hat : int
        Estimated intrinsic dimensionality.
    loglik : ndarray, shape (p-1,)
        Profile log-likelihood values for q = 1 … p-1.
    """
    s = np.asarray(s, dtype=float)
    p = s.size
    loglik = np.empty(p - 1)

    for q in range(1, p):  # candidate cut–off
        S1, S2 = s[:q], s[q:]  # two “groups” of values
        n1, n2 = q, p - q
        mu1, mu2 = S1.mean(), S2.mean()

        # pooled estimate of the common variance  σ²ˆ  (Eq. 8 in the paper)
        var1, var2 = S1.var(ddof=1), S2.var(ddof=1)
        sigma2 = max(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2), eps)

        # log-likelihood under a Gaussian model with common σ²  (Eqs. 1–2)
        ll = -0.5 * (n1 + n2) * np.log(2 * np.pi * sigma2)
        ll -= 0.5 * ((S1 - mu1) ** 2).sum() / sigma2
        ll -= 0.5 * ((S2 - mu2) ** 2).sum() / sigma2
        loglik[q - 1] = ll

    q_hat = np.argmax(loglik) + 1  # +1 because q starts at 1
    return q_hat, loglik


def unflatten_grid(flattened_grid, k_grid, T_grid):
    k_unique = np.unique(k_grid)
    T_unique = np.unique(T_grid)
    grid_2d = np.zeros((len(T_unique), len(k_unique)))

    for i, t in enumerate(T_unique):
        for j, k in enumerate(k_unique):
            idx = np.where((np.isclose(T_grid, t)) & (np.isclose(k_grid, k)))[0]
            if len(idx) > 0:
                grid_2d[i, j] = flattened_grid[idx[0]]

    return grid_2d, k_unique, T_unique


def svd_analysis(folder, all_w_grid, k_grid, T_grid):

    all_w_grid = np.array(all_w_grid)
    print("all_w_grid.shape", all_w_grid.shape)

    # Perform SVD
    U, s, Vt = np.linalg.svd(all_w_grid, full_matrices=False)
    q_hat, loglik = profile_likelihood_dim(s)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot singular values
    idx = np.arange(len(s))
    ax1.plot(idx, s, marker="o", label="Singular value", color="tab:blue")
    # ax1.set_xscale("log")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Singular Value", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # right-hand y-axis: profile log-likelihood
    ax2 = ax1.twinx()
    ax2.plot(idx[1:], loglik, marker="x", label="Profile log-likelihood", color="tab:red")
    ax2.set_ylabel("Profile Log-Likelihood", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # mark optimal dimensionality
    ax2.axvline(q_hat, color="tab:green", linestyle="--", linewidth=1, label=f"$\\hat{{q}}={q_hat}$")

    # combine legends from both axes
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("SVD Scree Plot with Profile Log-Likelihood")
    plt.tight_layout()
    plt.savefig(f"{folder}/svd_singular_values_and_profile.png")
    plt.show()

    # Unflatten the grid for plotting

    # Plot the first six singular vectors as heatmaps
    plt.figure(figsize=(18, 5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        # Vt contains the right singular vectors (modes of variation in volatility surfaces)
        vec = Vt[i]
        # Reshape the vector back to the 2D grid
        vec_2d, k_unique, T_unique = unflatten_grid(vec, k_grid, T_grid)

        img = plt.imshow(vec_2d, aspect="auto", cmap="viridis", origin="lower", extent=[min(k_unique), max(k_unique), min(T_unique), max(T_unique)])
        plt.colorbar(img, label=f"SV {i+1} Value")
        plt.title(f"Singular Vector {i+1}")
        plt.xlabel("Strike (k)")
        plt.ylabel("Time to Maturity (T)")

    plt.tight_layout()
    plt.savefig(f"{folder}/svd_singular_vectors.png")
    plt.show()

    # Randomly select a vol surface and show original vs. reconstruction
    idx = np.random.randint(0, all_w_grid.shape[0])
    sample_surf = all_w_grid[idx]

    # Project onto first 6 singular vectors
    projection = np.zeros_like(sample_surf)
    for i in range(6):
        projection += np.dot(sample_surf, Vt[i]) * Vt[i]

    # Compute the residual
    residual = sample_surf - projection

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Reshape for visualization
    surf_2d, k_unique, T_unique = unflatten_grid(sample_surf, k_grid, T_grid)
    proj_2d, _, _ = unflatten_grid(projection, k_grid, T_grid)
    resid_2d, _, _ = unflatten_grid(residual, k_grid, T_grid)

    # Common color scale for original and reconstruction
    vmin = min(surf_2d.min(), proj_2d.min())
    vmax = max(surf_2d.max(), proj_2d.max())

    # Original surface
    im0 = axes[0].imshow(surf_2d, aspect="auto", cmap="viridis", origin="lower", extent=[min(k_unique), max(k_unique), min(T_unique), max(T_unique)], vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Vol Surface")
    axes[0].set_xlabel("Strike (k)")
    axes[0].set_ylabel("Time to Maturity (T)")
    fig.colorbar(im0, ax=axes[0])

    # Reconstruction using 6 singular vectors
    im1 = axes[1].imshow(proj_2d, aspect="auto", cmap="viridis", origin="lower",
                         extent=[min(k_unique), max(k_unique), min(T_unique), max(T_unique)],
                         vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstruction (6 SVs)")
    axes[1].set_xlabel("Strike (k)")
    fig.colorbar(im1, ax=axes[1])

    # Residual
    im2 = axes[2].imshow(resid_2d, aspect="auto", cmap="coolwarm", origin="lower",
                         extent=[min(k_unique), max(k_unique), min(T_unique), max(T_unique)])
    axes[2].set_title("Residual (Original - Reconstruction)")
    axes[2].set_xlabel("Strike (k)")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"{folder}/vol_surface_reconstruction.png")
    plt.show()

    return U, s, Vt
