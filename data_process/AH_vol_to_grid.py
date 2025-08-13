import QuantLib as ql
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from process_raw_data import find_fwd_per_quote_date


# use Andreasen-Huge method to build a no-arbitrage vol surface on given k and T grid
def AH_vol_grid(folder, year, k_grid, T_grid):
    df = pd.read_csv(f"{folder}/processed_data_{year}.csv", skipinitialspace=True)

    # Create folder for the year if it doesn't exist
    year_folder = f"{folder}/{year}"
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
    # find BS vol for each strike and expiry
    all_quote_dates = df["QUOTE_DATE"].unique()
    print("all_quote_dates", all_quote_dates)
    for quote_date in all_quote_dates:
        if quote_date != "2023-10-02":
            continue

        # read already processed BS_vol
        df_quote_date = pd.read_csv(f"{folder}/{year}/volatility_surface_{quote_date}.csv")
        print(f"Processing quote date: {quote_date}")
        # df_quote_date = df[df["QUOTE_DATE"] == quote_date]
        print(df_quote_date.head())

        # find AH processed arbitrage free vol surface on given k and T grid
        AH_vol_surface = grid_AH_vol_per_quote_date(quote_date, df_quote_date, k_grid, T_grid)

        # spot, r_curve, T_r, quotes = AH_find_OOM_options_per_quote_date(df_quote_date)

        # plot_r_curve_quotes(folder, year, quote_date, r_curve, T_r, quotes)

        # quote_date, spot, r_curve, T_r, quotes, k_grid, T_grid
        # vol_surface = AH_find_vol_grid_per_quote_date(quote_date, spot, r_curve, T_r, k_grid, T_grid)

        plot_AH_vol_surface(folder, year, quote_date, df_quote_date, AH_vol_surface, k_grid, T_grid)


def grid_AH_vol_per_quote_date(quote_date, df_quote_date, k_grid, T_grid):
    # market input
    today = ql.Date(*map(int, quote_date.split("-")[::-1]))
    q_flat = 0.00
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    rTs = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.02, dc))  # flat curve as approximation
    qTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q_flat, dc))
    spot = df_quote_date["UNDERLYING_LAST"].iloc[0]
    spotH = ql.QuoteHandle(ql.SimpleQuote(spot))
    print("spot", spot)
    print("spotH", spotH.value())
    print("df_quote_date", df_quote_date)
    bs_vols = []
    # build calibration set from quotes and spot
    calib = ql.CalibrationSet()

    for _, row in df_quote_date.iterrows():
        K = spot * np.exp(row["SPOT_LOG_MONEYNESS"])
        T = row["YTE"]
        vol = row["BS_VOL"]
        option_type = 1 if K > spot else -1
        if option_type == -1:
            option_type = ql.Option.Put
        elif option_type == 1:
            option_type = ql.Option.Call
        payoff = ql.PlainVanillaPayoff(option_type, K)
        expiry = calendar.advance(today, ql.Period(int(round(T * 365)), ql.Days))
        exercise = ql.EuropeanExercise(expiry)
        option = ql.VanillaOption(payoff, exercise)

        calib.push_back((option, ql.SimpleQuote(vol)))

    # use Andreasen-Huge to interpolate the vol surface
    ah_interp = ql.AndreasenHugeVolatilityInterpl(calib, spotH, rTs, qTS, ql.AndreasenHugeVolatilityInterpl.CubicSpline, ql.AndreasenHugeVolatilityInterpl.CallPut)
    ah_surface = ql.AndreasenHugeVolatilityAdapter(ah_interp)
    ah_surface.enableExtrapolation()
    volTS = ql.BlackVolTermStructureHandle(ah_surface)

    # fill the vol grid
    vol_surface = np.zeros((len(k_grid), len(T_grid)))
    print("vol_surface shape", vol_surface.shape)
    print("T_grid", T_grid)
    print("k_grid", k_grid)
    K_grid = spot * np.exp(k_grid)
    print("K_grid", K_grid)
    for i, K in enumerate(K_grid):
        for j, T in enumerate(T_grid):
            try:
                vol = volTS.blackVol(T, K)
            except RuntimeError as e:
                print(f"Warning: {e}. Setting vol to NaN for T={T}, K={K}")
                vol = np.nan
            vol_surface[i][j] = vol
    print("vol_surface", vol_surface)
    return vol_surface


def AH_find_OOM_options_per_quote_date(df, maxYTE=2.0):
    """
    take in the filtered df of option price etc,
    output
    spot
    forward implied interest rate per expiry
    and the quotes ready for AH vol calibration
    """
    # filter out very long dated options
    # df = df[df["YTE"] <= maxYTE]
    # df = df.reset_index(drop=True)

    # Calculate forward price for each unique expiry

    unique_expiries = df["YTE"].unique()
    forward_prices = {}

    spot = df["UNDERLYING_LAST"].iloc[0]
    r_curve = []
    T_r = []
    quotes = []

    for expiry in unique_expiries:
        expiry_df = df[df["YTE"] == expiry]
        if expiry_df.empty:
            continue
        print(f"Calculating forward price for expiry: {expiry}")
        print(expiry_df)
        forward_prices[expiry] = find_fwd_per_quote_date(expiry_df)
        T_r.append(expiry)
        F = forward_prices[expiry]
        r = np.log(F / spot) / expiry
        r_curve.append(r)

    # Map forward prices back to original dataframe
    df["FORWARD"] = df["YTE"].map(forward_prices)

    # Sort r_curve and T_r together based on T_r
    sorted_pairs = sorted(zip(T_r, r_curve))
    T_r, r_curve = zip(*sorted_pairs)

    for _, row in df.iterrows():
        S = row["UNDERLYING_LAST"]
        K = S * np.exp(row["SPOT_LOG_MONEYNESS"])
        T = row["YTE"]
        F = row["FORWARD"]
        vol = row["BS_VOL"]
        option_type = 1 if K > F else -1  # call=1, put=-1
        quotes.append((K, T, vol, option_type))

    return spot, r_curve, T_r, quotes


def plot_r_curve_quotes(folder, year, quote_date, r_curve, T_r, quotes):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8))

    # Plot r_curve as a 2D plot
    ax1 = fig.add_subplot(121)
    ax1.plot(T_r, r_curve, marker="o", label="Implied Interest Rate Curve")
    ax1.set_xlabel("Time to Expiry (T)")
    ax1.set_ylabel("Implied Interest Rate (r)")
    ax1.set_title(f"Implied Interest Rate Curve on {quote_date} of {year}")
    ax1.grid(True)
    ax1.legend()

    # Plot quotes as a 3D surface
    ax2 = fig.add_subplot(122, projection="3d")
    strikes = [q[0] for q in quotes]
    expiries = [q[1] for q in quotes]
    prices = [q[2] for q in quotes]

    # Create a grid for strikes and expiries
    unique_strikes = np.unique(strikes)
    unique_expiries = np.unique(expiries)
    _strike_grid, _expiry_grid = np.meshgrid(unique_strikes, unique_expiries)

    # Plot scatter points in 3D
    ax2.scatter(strikes, expiries, prices, c=prices, cmap="viridis", marker="o")

    ax2.set_xlabel("Strike Price (K)")
    ax2.set_ylabel("Time to Expiry (T)")
    ax2.set_zlabel("Option Price")
    ax2.set_title(f"Option Prices on {quote_date} of {year}")

    # Save and show the combined figure
    plt.tight_layout()
    plt.savefig(f"{folder}/{year}/combined_plot_{quote_date}.png")
    plt.show()


# ---------------------------------------------------------------
# helper: build a Handle<YieldTermStructure> from (T, r) pairs
# --------------------------------------------------------------


def make_zero_curve(today, T_years, zero_rates, *, calendar=ql.NullCalendar(), day_counter=ql.Actual365Fixed()):

    assert len(T_years) == len(zero_rates), "T and r lengths differ"
    dates = [calendar.advance(today, ql.Period(int(round(t * 365)), ql.Days)) for t in T_years]
    curve = ql.ZeroCurve(dates, list(zero_rates), day_counter, calendar)  # node dates  # per-node rates
    return ql.YieldTermStructureHandle(curve)


def AH_find_vol_grid_per_quote_date(quote_date, spot, r_curve, T_r, quotes, k_grid, T_grid):

    today = ql.Date(*map(int, quote_date.split("-")[::-1]))
    q_flat = 0.00
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    # rTS = make_zero_curve(today=today, T_years=T_r, zero_rates=r_curve, calendar=calendar, day_counter=dc)
    rTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r_curve[0], dc))  # flat curve as approximation
    qTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q_flat, dc))
    spotH = ql.QuoteHandle(ql.SimpleQuote(spot))
    print("spot", spot)
    print("spotH", spotH.value())
    print("quotes", quotes)
    bs_vols = []
    # build calibration set from quotes and spot
    calib = ql.CalibrationSet()
    for K, T, price, option_type in quotes:
        if option_type == -1:
            option_type = ql.Option.Put
        elif option_type == 1:
            option_type = ql.Option.Call
        payoff = ql.PlainVanillaPayoff(option_type, K)
        # expiry = today + int(T * 365 + 0.5)
        expiry = calendar.advance(today, ql.Period(int(round(T * 365)), ql.Days))

        exercise = ql.EuropeanExercise(expiry)
        option = ql.VanillaOption(payoff, exercise)
        calib.push_back((option, ql.SimpleQuote(price)))

        # Calculate BS implied volatility
        process = ql.BlackScholesMertonProcess(spotH, qTS, rTS, ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, 0.02, dc)))

        try:
            implied_vol = option.impliedVolatility(price, process, 1e-6, 100, 0.001, 2.0)
            bs_vols.append((K, T, price, option_type, implied_vol))
            print(f"BS vol  K={K:6.1f}  T={T:.4f}  σ={implied_vol:.3f}")
        except RuntimeError as e:
            print(f"impliedVol failed  K={K:.1f}  T={T:.4f}: {e}")
            bs_vols.append((K, T, price, option_type, np.nan))

    # Andreasen-Huge surface interpolation
    ah_interp = ql.AndreasenHugeVolatilityInterpl(calib, spotH, rTS, qTS, ql.AndreasenHugeVolatilityInterpl.CubicSpline, ql.AndreasenHugeVolatilityInterpl.CallPut)

    ah_surface = ql.AndreasenHugeVolatilityAdapter(ah_interp)
    ah_surface.enableExtrapolation()
    volTS = ql.BlackVolTermStructureHandle(ah_surface)

    # fill the vol grid
    vol_surface = []
    print("k_grid", k_grid)
    print("K_grid", spot * np.exp(k_grid))
    for T in T_grid:
        for k in k_grid:
            K = spot * np.exp(k)
            try:
                vol = volTS.blackVol(T, K)
                print(f"T={T}, K={K}, Vol={vol}")
            except RuntimeError as e:
                print(f"Warning: {e}. Setting vol to NaN for T={T}, K={K}")
                vol = np.nan
            vol_surface.append(vol)

    return vol_surface


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_AH_vol_surface(folder, year, quote_date, df_quote_date, vol_surface, k_grid, T_grid):

    T, K = np.meshgrid(T_grid, k_grid)
    print("K shape", K.shape)
    print("T shape", T.shape)
    print("vol_surface shape", vol_surface.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, T, vol_surface, cmap="rainbow")
    # Scatter plot of BS vol in df_quote_date
    #ax.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["BS_VOL"], color="blue", label="BS Volatility")
    ax.scatter(df_quote_date["SPOT_LOG_MONEYNESS"], df_quote_date["YTE"], df_quote_date["BS_VOL"], color="blue", label="BS Volatility")
    ax.legend()
    ax.set_xlabel("Log-Moneyness (k)")
    ax.set_ylabel("Time to Expiry (T)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"Andreasen-Huge Vol Surface on {quote_date} of {year}")
    plt.savefig(f"{folder}/{year}/AH_vol_surface_{quote_date}.png")
    plt.show()


def sample_pricing_AH_vol_surface():
    pass
    # TODO: do some sample pricing as sanity check
