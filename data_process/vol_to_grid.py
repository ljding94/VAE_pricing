import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import numpy as np


def grid_total_variance(df, k_grid, T_grid):

    # Extract the input data points
    k_points = df["SPOT_LOG_MONEYNESS"].values
    T_points = df["YTE"].values
    total_var_points = df["TOTAL_VARIANCE"].values
    # Create meshgrid for interpolation
    T, k = np.meshgrid(T_grid, k_grid)  # numpy meshgrid returns (nk, nT) shape, backwards

    # Initialize output array
    total_var_grid = np.zeros_like(k)

    # 1.  Assemble coordinates – scaling helps when k and T have different magnitudes
    xy = np.column_stack((k_points / np.std(k_points), T_points / np.std(T_points)))

    # 2.  Fit the surface (tune `smoothing` & `neighbors` for speed vs. fidelity)
    try:
        rbf = RBFInterpolator(
            xy,
            total_var_points,
            kernel="thin_plate_spline",
            smoothing=1e-4,
            # xy, total_var_points, kernel="linear", degree=1, smoothing=1e-3, neighbors=20
            # infinitely differentiable → smooth  # 0 ⇒ exact interpolation  # optional: speeds up large problems
        )

        # 3.  Evaluate on the grid
        grid_xy = np.column_stack((k.flatten() / np.std(k_points), T.flatten() / np.std(T_points)))
        total_var_grid = rbf(grid_xy).reshape(k.shape)  # → full grid, incl. extrapolated rim
        return total_var_grid
    except:
        return None


def clean_total_variance_calendar_arbitrage(total_var_grid, k_grid, T_grid):
    cleaned_total_var_grid = total_var_grid.copy()
    for i in range(len(k_grid)):
        cum_var = 0.0
        for j in range(len(T_grid)):
            proposed_var = cleaned_total_var_grid[i, j]
            cum_var = max(cum_var, proposed_var)
            cleaned_total_var_grid[i, j] = cum_var
    return cleaned_total_var_grid


def clean_total_variance_butterfly_arbitrage(total_var_grid, k_grid, T_grid):
    S0 = 1.0
    r = 0.02

    def bs_call(S0, K, T, r, sigma):
        if T <= 0 or sigma.any() <= 0:
            return np.maximum(S0 - K, 0.0)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    adjusted_var = total_var_grid.copy()
    K_grid = S0 * np.exp(k_grid)

    for j in range(len(T_grid)):
        T = T_grid[j]
        if T <= 0:
            continue
        vars_slice = adjusted_var[:, j]
        vols = np.sqrt(np.maximum(vars_slice / T, 1e-8))
        calls = bs_call(S0, K_grid, T, r, vols)

        first_diff = np.diff(calls) / np.diff(K_grid)
        second_diff = np.diff(first_diff) / np.diff(K_grid[:-1])

        viol_idx = np.where(second_diff <= 0)[0]
        if len(viol_idx) == 0:
            continue

        for idx in viol_idx:
            #print("enforcing butterfly arbitrage at idx:", idx)
            def objective(delta_vol):
                temp_vols = vols.copy()
                temp_vols[idx : idx + 3] += delta_vol
                temp_vols = np.maximum(temp_vols, 1e-6)
                temp_calls = bs_call(S0, K_grid[idx : idx + 3], T, r, temp_vols[idx : idx + 3])
                temp_fd = np.diff(temp_calls) / np.diff(K_grid[idx : idx + 3])
                temp_sd = np.diff(temp_fd) / np.diff(K_grid[idx : idx + 2])[0]
                return -temp_sd if temp_sd <= 0 else 0

            res = minimize_scalar(objective, bounds=(0, 0.1), method="Bounded")
            if res.success:
                vols[idx : idx + 3] += res.x

        adjusted_var[:, j] = vols**2 * T

    return adjusted_var


import os
import sys

# Add parent directory to path so we can import from there
sys.path.append(os.path.join(os.path.dirname(__file__), "../pricing"))
from american_put_pricer import price_american_put_options_multi_KT


def test_pricing_arbitrage_free_surface(quote_date, vol_grid, k_grid, T_grid, N_sample=5):
    # sample some T and k within the grid range, and test pricing of American put
    # TODO:
    # 1. Find range of k and T
    K_grid = np.exp(k_grid)
    K_min, K_max = min(K_grid), max(K_grid)
    T_min, T_max = min(T_grid), max(T_grid)

    # 2. Sample some k and T within the range
    # Sample N_sample combinations of K and T randomly within the range
    np.random.seed(42)  # For reproducibility
    sampled_K = np.random.uniform(K_min, K_max, N_sample)
    sampled_T = np.random.uniform(T_min, T_max, N_sample)
    sampled_KT = [(K, T) for K, T in zip(sampled_K, sampled_T)]

    # 3. Price American put options for the sampled k and T
    try:
        prices = price_american_put_options_multi_KT(quote_date, vol_grid, K_grid, T_grid, sampled_KT)
        # for (K, T), price in zip(sampled_KT, prices):
        #    print(f"K: {K}, T: {T}, Price: {price}")
    except Exception as e:
        print(f"Error during price calculation: {e}")
        return 0
    #print(f"Test passed for quote date {quote_date} with {N_sample} samples.")
    #print(prices)

    return 1


# TODO: do calendar arbitrage and butterfly arbitrage iteratively for multiple times
# check the arbitrage using quantlib code
# set max cleaning iterations, get rid of that vol surface if can't be fixed


def grid_volatility_surface(df, k_grid, T_grid):
    # Extract the input data points
    k_points = df["SPOT_LOG_MONEYNESS"].values
    T_points = df["YTE"].values
    vol_points = df["BS_VOL"].values

    # Create meshgrid for interpolation
    T, k = np.meshgrid(T_grid, k_grid)

    # Initialize output array
    vol_grid = np.zeros_like(k)

    # 1.  Assemble coordinates – scaling helps when k and T have different magnitudes
    xy = np.column_stack((k_points / np.std(k_points), T_points / np.std(T_points)))

    # 2.  Fit the surface (tune `smoothing` & `neighbors` for speed vs. fidelity)
    try:
        rbf = RBFInterpolator(
            xy,
            vol_points,
            kernel="thin_plate_spline",
            smoothing=1e-4,
        )

        # 3.  Evaluate on the grid
        grid_xy = np.column_stack((k.flatten() / np.std(k_points), T.flatten() / np.std(T_points)))
        vol_grid = rbf(grid_xy).reshape(k.shape)  # → full grid, incl. extrapolated rim

        return vol_grid
    except:
        return None


def plot_vol_and_variance_surface(folder, year, quote_date, df, k_grid, T_grid, total_var_grid, vol_grid):
    plt.figure(figsize=(18, 10))
    ax1 = plt.subplot(2, 3, 1, projection="3d")
    ax2 = plt.subplot(2, 3, 2, projection="3d")
    ax3 = plt.subplot(2, 3, 3, projection="3d")
    ax4 = plt.subplot(2, 3, 4, projection="3d")
    ax5 = plt.subplot(2, 3, 5, projection="3d")
    ax6 = plt.subplot(2, 3, 6, projection="3d")

    ax1.scatter(df["SPOT_LOG_MONEYNESS"], df["YTE"], df["BS_VOL"], s=2)
    ax2.scatter(df["SPOT_LOG_MONEYNESS"], df["YTE"], df["TOTAL_VARIANCE"], s=2)
    ax3.scatter(df["SPOT_LOG_MONEYNESS"], df["YTE"], df["BS_VOL"], s=2)
    ax4.scatter(df["SPOT_LOG_MONEYNESS"], df["YTE"], df["BS_VOL"], s=2)

    k_mesh, T_mesh = np.meshgrid(k_grid, T_grid)

    if total_var_grid is not None:
        ax2.plot_surface(k_mesh, T_mesh, total_var_grid, cmap="rainbow")

        # Convert total variance to BS vol for ax3
        vol_from_total_var = np.sqrt(total_var_grid) / np.sqrt(T_mesh)
        ax3.plot_surface(k_mesh, T_mesh, vol_from_total_var, cmap="rainbow")

    if vol_grid is not None:
        ax4.plot_surface(k_mesh, T_mesh, vol_grid, cmap="rainbow")

    if total_var_grid is not None and vol_grid is not None:
        variance_diff = total_var_grid - (vol_grid**2 * T_mesh)
        ax5.plot_surface(k_mesh, T_mesh, variance_diff, cmap="rainbow")

        # Compare interpolated vol vs vol from total variance
        vol_from_total_var = np.sqrt(total_var_grid) / np.sqrt(T_mesh)
        vol_difference = vol_grid - vol_from_total_var
        ax6.plot_surface(k_mesh, T_mesh, vol_difference, cmap="rainbow")

    ax1.set_xlim(min(k_grid), max(k_grid))
    ax1.set_ylim(min(T_grid), max(T_grid))
    ax2.set_xlim(min(k_grid), max(k_grid))
    ax2.set_ylim(min(T_grid), max(T_grid))
    ax2.set_zlim(0, 0.15)
    ax3.set_xlim(min(k_grid), max(k_grid))
    ax3.set_ylim(min(T_grid), max(T_grid))
    ax4.set_xlim(min(k_grid), max(k_grid))
    ax4.set_ylim(min(T_grid), max(T_grid))
    ax5.set_xlim(min(k_grid), max(k_grid))
    ax5.set_ylim(min(T_grid), max(T_grid))
    ax6.set_xlim(min(k_grid), max(k_grid))
    ax6.set_ylim(min(T_grid), max(T_grid))

    ax1.set_xlabel("Log Moneyness")
    ax1.set_ylabel("Time to Expiry")
    ax1.set_zlabel("Volatility")
    ax2.set_xlabel("Log Moneyness")
    ax2.set_ylabel("Time to Expiry")
    ax2.set_zlabel("Total Variance")
    ax3.set_xlabel("Log Moneyness")
    ax3.set_ylabel("Time to Expiry")
    ax3.set_zlabel("Volatility")
    ax4.set_xlabel("Log Moneyness")
    ax4.set_ylabel("Time to Expiry")
    ax4.set_zlabel("Volatility")
    ax5.set_xlabel("Log Moneyness")
    ax5.set_ylabel("Time to Expiry")
    ax5.set_zlabel("Variance Difference")
    ax6.set_xlabel("Log Moneyness")
    ax6.set_ylabel("Time to Expiry")
    ax6.set_zlabel("Vol Difference")

    ax1.set_title(f"Market BS Vol for {quote_date}")
    ax2.set_title(f"Total Variance Surface for {quote_date}")
    ax3.set_title(f"Vol from Total Variance for {quote_date}")
    ax4.set_title(f"Interpolated Vol Surface for {quote_date}")
    ax5.set_title(f"Variance Difference for {quote_date}")
    ax6.set_title(f"Vol Surface Difference for {quote_date}")

    plt.tight_layout()
    plt.savefig(f"{folder}/{year}/vol_and_variance_surface_{quote_date}.png")
    # plt.show()


def plot_total_var_grid(folder, year, quote_date, df_quote_date, k_grid, T_grid, total_var_grid):
    plt.figure(figsize=(20, 15))

    # Subplot 1: 3D scatter of BS_VOL vs SPOT_LOG_MONEYNESS and YTE
    ax1 = plt.subplot(2, 2, 1, projection="3d")
    ax1.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["BS_VOL"], s=2, alpha=0.7)
    ax1.set_xlabel("Log Moneyness")
    ax1.set_ylabel("Time to Expiry")
    ax1.set_zlabel("BS Volatility")
    ax1.set_title(f"Market BS Vol Scatter for {quote_date}")

    # Subplot 2: 3D scatter of TOTAL_VARIANCE vs SPOT_LOG_MONEYNESS and YTE
    ax2 = plt.subplot(2, 2, 2, projection="3d")
    ax2.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["TOTAL_VARIANCE"], s=2, alpha=0.7)
    ax2.set_xlabel("Log Moneyness")
    ax2.set_ylabel("Time to Expiry")
    ax2.set_zlabel("Total Variance")
    ax2.set_title(f"Market Total Variance Scatter for {quote_date}")

    # Subplot 3: Total variance grid surface with scatter overlay
    ax3 = plt.subplot(2, 2, 3, projection="3d")
    ax3.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["TOTAL_VARIANCE"], s=2, alpha=0.5, color='red', label='Market Data')

    if total_var_grid is not None:
        T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)
        ax3.plot_surface(k_mesh, T_mesh, total_var_grid, cmap="rainbow", alpha=0.8)

    ax3.set_xlabel("Log Moneyness")
    ax3.set_ylabel("Time to Expiry")
    ax3.set_zlabel("Total Variance")
    ax3.set_title(f"Total Variance Surface vs Market Data for {quote_date}")
    ax3.legend()

    # Subplot 4: BS vol grid surface (converted from total variance) with scatter overlay
    ax4 = plt.subplot(2, 2, 4, projection="3d")
    ax4.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["BS_VOL"], s=2, alpha=0.5, color='red', label='Market Data')

    if total_var_grid is not None:
        T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)
        # Convert total variance to BS vol
        bs_vol_grid = np.sqrt(total_var_grid / np.maximum(T_mesh, 1e-6))
        ax4.plot_surface(k_mesh, T_mesh, bs_vol_grid, cmap="rainbow", alpha=0.8)

    ax4.set_xlabel("Log Moneyness")
    ax4.set_ylabel("Time to Expiry")
    ax4.set_zlabel("BS Volatility")
    ax4.set_title(f"BS Vol Surface vs Market Data for {quote_date}")
    ax4.legend()

    # Set consistent axis limits
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(min(k_grid), max(k_grid))
        ax.set_ylim(min(T_grid), max(T_grid))

    ax2.set_zlim(0, 0.15)
    ax3.set_zlim(0, 0.15)

    plt.tight_layout()
    plt.savefig(f"{folder}/{year}/total_variance_surface_{quote_date}.png")
    # plt.show()


def plot_vol_grid(folder, year, quote_date, df_quote_date, k_grid, T_grid, vol_grid):
    plt.figure(figsize=(20, 15))

    # Subplot 1: 3D scatter of BS_VOL vs SPOT_LOG_MONEYNESS and YTE
    ax1 = plt.subplot(2, 2, 1, projection="3d")
    ax1.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["BS_VOL"], s=2, alpha=0.7)
    ax1.set_xlabel("Log Moneyness")
    ax1.set_ylabel("Time to Expiry")
    ax1.set_zlabel("BS Volatility")
    ax1.set_title(f"Market BS Vol Scatter for {quote_date}")

    # Subplot 2: 3D scatter of TOTAL_VARIANCE vs SPOT_LOG_MONEYNESS andYTE
    ax2 = plt.subplot(2, 2, 2, projection="3d")
    ax2.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["TOTAL_VARIANCE"], s=2, alpha=0.7)
    ax2.set_xlabel("Log Moneyness")
    ax2.set_ylabel("Time to Expiry")
    ax2.set_zlabel("Total Variance")
    ax2.set_title(f"Market Total Variance Scatter for {quote_date}")

    # Subplot 3: Vol grid surface with scatter overlay
    ax3 = plt.subplot(2, 2, 3, projection="3d")
    ax3.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["BS_VOL"], s=2, alpha=0.5, color='red',label='Market Data')

    if vol_grid is not None:
        T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)
        ax3.plot_surface(k_mesh, T_mesh, vol_grid, cmap="rainbow", alpha=0.8)

    ax3.set_xlabel("Log Moneyness")
    ax3.set_ylabel("Time to Expiry")
    ax3.set_zlabel("BS Volatility")
    ax3.set_title(f"Vol Surface vs Market Data for {quote_date}")
    ax3.legend()

    # Subplot 4: Total variance grid surface (converted from vol) withscatter overlay
    ax4 = plt.subplot(2, 2, 4, projection="3d")
    ax4.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["TOTAL_VARIANCE"], s=2, alpha=0.5,color='red', label='Market Data')

    if vol_grid is not None:
        T_mesh, k_mesh = np.meshgrid(T_grid, k_grid)
        # Convert vol to total variance
        total_var_from_vol = vol_grid**2 * np.maximum(T_mesh, 1e-6)
        ax4.plot_surface(k_mesh, T_mesh, total_var_from_vol, cmap="rainbow", alpha=0.8)

    ax4.set_xlabel("Log Moneyness")
    ax4.set_ylabel("Time to Expiry")
    ax4.set_zlabel("Total Variance")
    ax4.set_title(f"Total Variance from Vol Surface vs Market Data for{quote_date}")
    ax4.legend()

    # Set consistent axis limits
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(min(k_grid), max(k_grid))
        ax.set_ylim(min(T_grid), max(T_grid))

    ax2.set_zlim(0, 0.15)
    ax4.set_zlim(0, 0.15)

    plt.tight_layout()
    plt.savefig(f"{folder}/{year}/vol_surface_{quote_date}.png")

def process_var_to_grid(folder, year, k_grid, T_grid):
    df = pd.read_csv(f"{folder}/processed_data_{year}.csv", skipinitialspace=True)
    all_quote_dates = df["QUOTE_DATE"].unique()

    var_arb_date_count = 0
    vol_arb_date_count = 0
    for quote_date in all_quote_dates:
        # if quote_date != "2023-10-02":
        # continue # for debugging
        # if quote_date != "2018-04-25":  # this has calendar arbitrage issue
        #    continue  # for debugging
        print(f"Griding vol surface for quote date: {quote_date}")
        df_quote_date = pd.read_csv(f"{folder}/{year}/volatility_surface_{quote_date}.csv")

        # simple grid
        total_var_grid = grid_total_variance(df_quote_date, k_grid, T_grid)
        # remove calendar arbitrage for each strike
        total_var_grid = clean_total_variance_calendar_arbitrage(total_var_grid, k_grid, T_grid)

        #total_var_grid = clean_total_variance_butterfly_arbitrage(total_var_grid, k_grid, T_grid)
        vol_grid_from_totalvar = np.sqrt(total_var_grid / np.maximum(T_grid, 1e-6))  # Convert total variance to vol surface
        # test pricing see if it workds
        testpass = test_pricing_arbitrage_free_surface(quote_date, vol_grid_from_totalvar, k_grid, T_grid)

        if testpass:
            plot_total_var_grid(folder, year, quote_date, df_quote_date, k_grid, T_grid, total_var_grid)
            total_var_grid_data_npz = {"k_grid": k_grid, "T_grid": T_grid, "total_var_grid": total_var_grid}
            np.savez(f"{folder}/{year}/total_var_grid_data_{quote_date}.npz", **total_var_grid_data_npz)
        else:
            var_arb_date_count += 1
            print(f"Pricing test failed for quote date {quote_date}, skipping saving total var grid data")
            # continue

        vol_grid = grid_volatility_surface(df_quote_date, k_grid, T_grid)
        vol_test_pass = test_pricing_arbitrage_free_surface(quote_date, vol_grid, k_grid, T_grid)
        if vol_test_pass:
            plot_vol_grid(folder, year, quote_date, df_quote_date, k_grid, T_grid, vol_grid)
            vol_grid_data_npz = {"k_grid": k_grid, "T_grid": T_grid, "vol_grid": vol_grid}
            np.savez(f"{folder}/{year}/vol_grid_data_{quote_date}.npz", **vol_grid_data_npz)
        else:
            vol_arb_date_count += 1
            print(f"Pricing test failed for vol surface on quote date {quote_date}, skipping saving vol grid data")

        # plot_vol_and_variance_surface(folder, year, quote_date, df_quote_date, k_grid, T_grid, total_var_grid, vol_grid)
        # Create a DataFrame with the grid data
        """
        if vol_grid is not None:
            # Save flatten data to to CSV
            grid_data = pd.DataFrame({"k_grid": np.tile(k_grid, len(T_grid)), "T_grid": np.repeat(T_grid, len(k_grid)), "vol_grid": total_var_grid.flatten()})
            output_filename = f"{folder}/{year}/grid_data_{quote_date}.csv"
            grid_data.to_csv(output_filename, index=False)
            print(f"Grid data saved to {output_filename}")

            # Save raw data to npz
            grid_data_npz = {"k_grid": k_grid, "T_grid": T_grid, "vol_grid": vol_grid}
            np.savez(f"{folder}/{year}/grid_data_{quote_date}.npz", **grid_data_npz)
        """
    print(f"Total {var_arb_date_count}/{len(all_quote_dates)} quote dates failed pricing test due to arbitrage issues")
    print(f"Total {vol_arb_date_count}/{len(all_quote_dates)} quote dates failed pricing test due to arbitrage issues")


def post_process_grid_data(folder, year, k_grid, T_grid):
    # further test the arbitrage free surface using massive pricing
    df = pd.read_csv(f"{folder}/processed_data_{year}.csv", skipinitialspace=True)
    all_quote_dates = df["QUOTE_DATE"].unique()

    var_arb_date_count = 0
    vol_arb_date_count = 0
    for quote_date in all_quote_dates:
        print(f"Post processing vol surface for quote date: {quote_date}")
        vol_file = f"{folder}/{year}/vol_grid_data_{quote_date}.npz"
        if not pd.io.common.file_exists(vol_file):
            print(f"File {vol_file} does not exist, skipping.")
            continue
        vol_grid_data = np.load(vol_file)

        test_pass = test_pricing_arbitrage_free_surface(quote_date, vol_grid_data["vol_grid"], vol_grid_data["k_grid"], vol_grid_data["T_grid"], N_sample=300)
        if test_pass:
            print(f"Post processing vol surface for quote date {quote_date} passed pricing test")
            post_vol_grid_data_npz = {"k_grid": vol_grid_data["k_grid"], "T_grid": vol_grid_data["T_grid"], "vol_grid": vol_grid_data["vol_grid"]}
            np.savez(f"{folder}/{year}/post_vol_grid_data_{quote_date}.npz", **post_vol_grid_data_npz)
        else:
            vol_arb_date_count += 1
            print(f"Pricing test failed for vol surface on quote date {quote_date}, skipping saving post vol grid data")
    '''
    for quote_date in all_quote_dates:
        print(f"Post processing total variance surface for quote date: {quote_date}")
        total_var_file = f"{folder}/{year}/total_var_grid_data_{quote_date}.npz"
        if not pd.io.common.file_exists(total_var_file):
            print(f"File {total_var_file} does not exist, skipping.")
            continue
        total_var_grid_data = np.load(total_var_file)

        vol_grid_from_totalvar = np.sqrt(total_var_grid_data["total_var_grid"] / np.maximum(T_grid, 1e-6))  # Convert total variance to
        test_pass = test_pricing_arbitrage_free_surface(quote_date, vol_grid_from_totalvar, total_var_grid_data["k_grid"], total_var_grid_data["T_grid"], N_sample=30)
        if test_pass:
            post_total_var_grid_data_npz = {"k_grid": total_var_grid_data["k_grid"], "T_grid": total_var_grid_data["T_grid"], "total_var_grid": total_var_grid_data["total_var_grid"]}
            np.savez(f"{folder}/{year}/post_total_var_grid_data_{quote_date}.npz", **post_total_var_grid_data_npz)
        else:
            var_arb_date_count += 1
            print(f"Pricing test failed for total variance surface on quote date {quote_date}, skipping saving post total var grid data")
    '''

    print(f"var arb: {var_arb_date_count}/{len(all_quote_dates)} quote dates failed pricing test due to arbitrage issues")
    print(f"vol arb: {vol_arb_date_count}/{len(all_quote_dates)} quote dates failed pricing test due to arbitrage issues")
    return var_arb_date_count, vol_arb_date_count


### ------------ pack vol surface data to npz for model training ------------- ###


def get_grid_data(folder, years, label, bad_dates=[]):
    all_quote_dates = []
    grid_dict = {}
    for year in years:
        df = pd.read_csv(f"{folder}/processed_data_{year}.csv", skipinitialspace=True)
        # get all unique quote dates
        all_quote_dates = df["QUOTE_DATE"].unique()

        for quote_date in all_quote_dates:
            # Skip if the date is in the bad_dates list
            if quote_date in bad_dates:
                print(f"Skipping bad date {quote_date}")
                continue
            grid_filename = f"{folder}/{year}/{label}grid_data_{quote_date}.npz"
            if not pd.io.common.file_exists(grid_filename):
                print(f"File {grid_filename} does not exist, skipping.")
                continue
            pd_vol_grid_data = np.load(grid_filename)
            if label == "":
                grid = pd_vol_grid_data["vol_grid"]
            elif label == "total_var_" or label == "post_total_var_":
                grid = pd_vol_grid_data["total_var_grid"]
            elif label == "vol_" or label == "post_vol_":
                grid = pd_vol_grid_data["vol_grid"]
            else:
                raise ValueError(f"Unknown label: {label}")

            k_grid = pd_vol_grid_data["k_grid"]
            T_grid = pd_vol_grid_data["T_grid"]

            # all_w_grid.append(total_var_grid)
            # all_quote_dates.append(quote_date)
            grid_dict[quote_date] = np.array(grid)


    # At the end of the function, return both the lists and dictionary
    return grid_dict, k_grid, T_grid


# TODO: package these data into a single npz file as data set

def pack_grid_data_to_npz(folder, label, grid_dict, k_grid, T_grid,  train_ratio=0.8):
    # unflatten the grid
    # Create a list to store dates in order
    dates = sorted(grid_dict.keys())

    # Update k_grid and T_grid to be unique values
    k_grid = np.unique(k_grid)
    T_grid = np.unique(T_grid)

    # randomly shuffle the data to train and test data
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(dates)
    num_train = int(len(dates) * train_ratio)
    train_dates = dates[:num_train]
    test_dates = dates[num_train:]
    # Create train and test dictionaries
    train_grid_surfaces = [grid_dict[date] for date in train_dates]
    test_grid_surfaces = [grid_dict[date] for date in test_dates]

    print(f"Created train set with {len(train_dates)} samples and test set with {len(test_dates)} samples")

    # Save train data
    np.savez(
        f"{folder}/{label}grid_train.npz",
        quote_dates=train_dates,
        surfaces_grid=train_grid_surfaces,
        k_grid=k_grid,
        T_grid=T_grid,
    )
    print(f"Saved train data to {folder}/{label}grid_train.npz")

    # Save test data
    np.savez(
        f"{folder}/{label}grid_test.npz",
        quote_dates=test_dates,
        surfaces_grid=test_grid_surfaces,
        k_grid=k_grid,
        T_grid=T_grid,
    )

    print(f"Saved test data to {folder}/{label}grid_test.npz")


## -------------- [end] pack vol surface data to npz for model training ------------- ###


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


def pack_vol_data_to_npz_old_from_pd(folder, label, vol_surface_dict, k_grid, T_grid, train_ratio=0.8):
    # unflatten the grid
    # Create a list to store dates in order
    dates = sorted(vol_surface_dict.keys())
    # Iterate through each date and unflatten the grid
    for i, date in enumerate(dates):
        flattened_grid = vol_surface_dict[date]
        grid_2d, _, _ = unflatten_grid(flattened_grid, k_grid, T_grid)
        vol_surface_dict[date] = grid_2d

    # Update k_grid and T_grid to be unique values
    k_grid = np.unique(k_grid)
    T_grid = np.unique(T_grid)

    # randomly shuffle the data to train and test data
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(dates)
    num_train = int(len(dates) * train_ratio)
    train_dates = dates[:num_train]
    test_dates = dates[num_train:]
    # Create train and test dictionaries
    train_vol_surfaces = [vol_surface_dict[date] for date in train_dates]
    test_vol_surfaces = [vol_surface_dict[date] for date in test_dates]

    print(f"Created train set with {len(train_dates)} samples and test set with {len(test_dates)} samples")

    # Save train data
    np.savez(
        f"{folder}/{label}_vol_surface_train.npz",
        quote_dates=train_dates,
        vol_surfaces=train_vol_surfaces,
        k_grid=k_grid,
        T_grid=T_grid,
    )
    print(f"Saved train data to {folder}/{label}_vol_surface_train.npz")

    # Save test data
    np.savez(
        f"{folder}/{label}_vol_surface_test.npz",
        quote_dates=test_dates,
        vol_surfaces=test_vol_surfaces,
        k_grid=k_grid,
        T_grid=T_grid,
    )

    print(f"Saved test data to {folder}/{label}_vol_surface_test.npz")
