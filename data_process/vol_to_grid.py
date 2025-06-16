import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def grid_total_variance(df, k_grid, T_grid):
    from scipy.interpolate import RBFInterpolator

    # Extract the input data points
    k_points = df["LOG_MONEYNESS"].values
    T_points = df["YTE"].values
    total_var_points = df["TOTAL_VARIANCE"].values
    # Create meshgrid for interpolation
    K, T = np.meshgrid(k_grid, T_grid)

    # Initialize output array
    total_var_grid = np.zeros_like(K)

    # 1.  Assemble coordinates – scaling helps when k and T have different magnitudes
    xy = np.column_stack((k_points / np.std(k_points), T_points / np.std(T_points)))

    # 2.  Fit the surface (tune `smoothing` & `neighbors` for speed vs. fidelity)
    try:
        rbf = RBFInterpolator(xy, total_var_points, kernel="thin_plate_spline", smoothing=1e-4
            #xy, total_var_points, kernel="linear", degree=1, smoothing=1e-3, neighbors=20
            # infinitely differentiable → smooth  # 0 ⇒ exact interpolation  # optional: speeds up large problems
        )

        # 3.  Evaluate on the grid
        grid_xy = np.column_stack((K.flatten() / np.std(k_points), T.flatten() / np.std(T_points)))
        total_var_grid = rbf(grid_xy).reshape(K.shape)  # → full grid, incl. extrapolated rim

        return total_var_grid
    except:
        return None



def plot_vol_and_variance_surface(folder, year, quote_date, df, k_grid, T_grid, total_var_grid):
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(1, 2, 1, projection="3d")
    ax2 = plt.subplot(1, 2, 2, projection="3d")
    ax1.scatter(df["LOG_MONEYNESS"], df["YTE"], df["BS_VOL"], s=2)
    ax2.scatter(df["LOG_MONEYNESS"], df["YTE"], df["TOTAL_VARIANCE"], s=2)
    k_mesh, T_mesh = np.meshgrid(k_grid, T_grid)

    if total_var_grid is not None:
        ax2.plot_surface(k_mesh, T_mesh, total_var_grid, cmap="rainbow")

    ax1.set_xlim(min(k_grid), max(k_grid))
    ax1.set_ylim(min(T_grid), max(T_grid))
    ax2.set_xlim(min(k_grid), max(k_grid))
    ax2.set_ylim(min(T_grid), max(T_grid))
    ax2.set_zlim(0, 0.15)

    ax1.set_xlabel("Log Moneyness")
    ax1.set_ylabel("Time to Expiry")
    ax1.set_zlabel("Volatility")
    ax2.set_xlabel("Log Moneyness")
    ax2.set_ylabel("Time to Expiry")
    ax2.set_zlabel("Total Variance")
    ax1.set_title(f"Volatility Surface for {quote_date}")
    ax2.set_title(f"Total Variance Surface for {quote_date}")

    plt.savefig(f"{folder}/{year}/vol_and_variance_surface_{quote_date}.png")
    #plt.show()


def process_var_to_grid(folder, year, k_grid, T_grid):
    df = pd.read_csv(f"{folder}/processed_data_{year}.csv", skipinitialspace=True)
    all_quote_dates = df["QUOTE_DATE"].unique()
    for quote_date in all_quote_dates:
        #if quote_date != "2023-10-20":
        #    continue
        print(f"Griding vol surface for quote date: {quote_date}")
        df_quote_date = pd.read_csv(f"{folder}/{year}/volatility_surface_{quote_date}.csv")
        total_var_grid = grid_total_variance(df_quote_date, k_grid, T_grid)
        plot_vol_and_variance_surface(folder, year, quote_date, df_quote_date, k_grid, T_grid, total_var_grid)
        # Create a DataFrame with the grid data
        if total_var_grid is not None:
            grid_data = pd.DataFrame({"k_grid": np.tile(k_grid, len(T_grid)), "T_grid": np.repeat(T_grid, len(k_grid)), "total_var_grid": total_var_grid.flatten()})

            # Save to CSV
            output_filename = f"{folder}/{year}/grid_data_{quote_date}.csv"
            grid_data.to_csv(output_filename, index=False)
            print(f"Grid data saved to {output_filename}")
