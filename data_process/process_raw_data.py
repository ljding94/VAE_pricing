import pandas as pd
import time
import os
import glob
import numpy as np
from scipy import stats
from scipy.optimize import fsolve


def get_data_filepath_per_year(folder, year):
    data_folder_per_year = {
        "2010": "spx_eod_2010-hixkhi",
        "2011": "spx_eod_2011-mpv77a",
        "2012": "spx_eod_2012-ftrobg",
        "2013": "spx_eod_2013-3p7m9u",
        "2014": "spx_eod_2014-4ky8n2",
        "2015": "spx_eod_2015-amtjlk",
        "2016": "spx_eod_2016-rctzlw",
        "2017": "spx_eod_2017-xspity",
        "2018": "spx_eod_2018-uqawfl",
        "2019": "spx_eod_2019-chxexe",
        "2020": "spx_eod_2020-wh6jt7",
        "2021h1": "spx_eod_2021-n4uqkm",
        "2021q3": "spx_eod_2021q3-kgzold",
        "2021q4": "spx_eod_2021q4-r3h0dx",
        "2022q1": "spx_eod_2022q1-ff0r18",
        "2022q2": "spx_eod_2022q2-bxgzw0",
        "2022q3": "spx_eod_2022q3-1f2afi",
        "2022q4": "spx_eod_2022q4-dmme3k",
        "2023q1": "spx_eod_2023q1-cfph7w",
        "2023q2": "spx_eod_2023q2-kdxt36",
        "2023q3": "spx_eod_2023q3-w9b0jk",
        "2023q4": "spx_eod_2023q4-ai4uc9",
    }

    if year in data_folder_per_year:
        folder_path = f"{folder}/{data_folder_per_year[year]}"
        filepath_list = glob.glob(os.path.join(folder_path, "*.txt"))
        return filepath_list


def data_filtering(df):
    # filter out rows there P_VOLUME is 0 and C_VOLUME is 0 of NaN
    df_filtered = df[(df["P_VOLUME"] != 0) & (df["C_VOLUME"] != 0)]
    df_filtered = df_filtered.dropna(subset=["P_VOLUME", "C_VOLUME"])

    # filter out rows where C_BID_SIZE is 0 or C_ASK_SIZE is 0
    df_filtered = df_filtered[df_filtered["C_BID_SIZE"] > 0]
    df_filtered = df_filtered[df_filtered["C_ASK_SIZE"] > 0]
    df_filtered = df_filtered[df_filtered["P_BID_SIZE"] > 0]
    df_filtered = df_filtered[df_filtered["P_ASK_SIZE"] > 0]

    # filter out DTE less than 7
    df_filtered = df_filtered[df_filtered["DTE"] >= 7]
    return df_filtered


def data_augmentation(df):
    # add spot log moneyness k = log(K/S)
    # df["SPOT_LOG_MONEYNES"] = np.log(df["STRIKE"] / df["UNDERLYING_LAST"], dtype=np.float32)
    df["YTE"] = df["DTE"] / 365.0  # Year to expiration
    df["C_MID"] = (df["C_BID"] + df["C_ASK"]) / 2.0
    df["P_MID"] = (df["P_BID"] + df["P_ASK"]) / 2.0
    df["C_MID-P_MID"] = df["C_MID"] - df["P_MID"]
    return df


def data_column_cleaning(df):
    # remove not needed columns
    columns_to_keep = ["QUOTE_DATE", "UNDERLYING_LAST", "STRIKE", "DTE", "YTE", "C_MID", "P_MID", "C_MID-P_MID"]
    df = df[columns_to_keep]
    return df


def load_and_process_data(folder, year):
    filepaths = get_data_filepath_per_year(folder, year)
    if not filepaths:
        print(f"No data files found for year {year}.")
        return pd.DataFrame()

    useful_columns = ["[QUOTE_DATE]", "[EXPIRE_DATE]", "[UNDERLYING_LAST]", "[DTE]", "[STRIKE]", "[C_VOLUME]", "[C_SIZE]", "[C_BID]", "[C_ASK]", "[P_VOLUME]", "[P_SIZE]", "[P_BID]", "[P_ASK]"]

    dataframes = []
    for filepath in filepaths:
        # First read without usecols to get all columns and strip whitespace
        df_temp = pd.read_csv(filepath, skipinitialspace=True)
        df_temp.columns = df_temp.columns.str.strip()

        # Now select only the useful columns
        df = df_temp[useful_columns]
        print(df.head())  # Display the first few rows of the dataframe
        # Rename columns to remove brackets
        df.columns = [col.strip().replace("[", "").replace("]", "") for col in df.columns]
        # seperate bid and ask size
        df["C_BID_SIZE"] = df["C_SIZE"].str.split("x").str[0].astype(int)
        df["C_ASK_SIZE"] = df["C_SIZE"].str.split("x").str[1].astype(int)
        df["P_BID_SIZE"] = df["P_SIZE"].str.split("x").str[0].astype(int)
        df["P_ASK_SIZE"] = df["P_SIZE"].str.split("x").str[1].astype(int)

        # filter the dataframe to keep only the useful columns
        df = data_filtering(df)

        # data augmentation
        df = data_augmentation(df)

        # data column cleaning
        df = data_column_cleaning(df)

        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.sort_values(by="QUOTE_DATE", inplace=True)

    # Save the combined dataframe
    output_filename = f"{folder}/processed_data_{year}.csv"
    combined_df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")

    return combined_df


def process_volatility_surface(folder, year):
    df = pd.read_csv(f"{folder}/processed_data_{year}.csv", skipinitialspace=True)

    # Create folder for the year if it doesn't exist
    year_folder = f"{folder}/{year}"
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
    # find BS vol for each strike and expiry
    all_quote_dates = df["QUOTE_DATE"].unique()
    print("all_quote_dates", all_quote_dates)
    for quote_date in all_quote_dates:
        # if quote_date != "2023-10-18":
        #    continue
        print(f"Processing quote date: {quote_date}")
        df_quote_date = df[df["QUOTE_DATE"] == quote_date]
        print(df_quote_date)
        df_vol_surface = find_volatility_surface_per_quote_date(df_quote_date)
        # df_vol_surface = df_vol_surface[["LOG_MONEYNESS", "YTE", "BS_VOL"]]
        df_vol_surface = df_vol_surface[["FORWARD", "UNDERLYING_LAST", "SPOT_LOG_MONEYNESS", "FWD_LOG_MONEYNESS", "YTE", "BS_VOL", "TOTAL_VARIANCE"]]
        df_vol_surface = df_vol_surface[df_vol_surface["BS_VOL"] > 0]
        df_vol_surface = df_vol_surface[df_vol_surface["YTE"] < 1.5]
        df_vol_surface.to_csv(f"{year_folder}/volatility_surface_{quote_date}.csv", index=False)


def find_fwd_per_quote_date(df):
    df = df.sort_values(by="STRIKE")
    df = df.reset_index(drop=True)

    for i in range(len(df)):
        if df["C_MID-P_MID"].iloc[i] < 0:
            break
    i0 = i

    if i0 == 0:
        end_idx = min(10, len(df))
        ys = df["STRIKE"].iloc[:end_idx]
        xs = df["C_MID-P_MID"].iloc[:end_idx]
    elif i0 >= len(df) - 1:
        start_idx = max(0, len(df) - 10)
        ys = df["STRIKE"].iloc[start_idx:]
        xs = df["C_MID-P_MID"].iloc[start_idx:]
    else:
        start_idx = max(0, i0 - 5)
        end_idx = min(len(df), i0 + 5)
        ys = df["STRIKE"].iloc[start_idx:end_idx]
        xs = df["C_MID-P_MID"].iloc[start_idx:end_idx]
    print(xs, ys)
    fwd = stats.linregress(xs, ys).intercept

    return fwd


def black76_implied_volatility(option_price, forward_price, strike_price, time_to_expiry, interest_rate, option_type="call"):
    V = option_price
    F = forward_price
    K = strike_price
    T = time_to_expiry
    r = interest_rate
    option_type = option_type

    def black76_price(vol):
        d1 = (np.log(F / K) + (0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        if option_type == "call":
            return np.exp(-r * T) * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
        elif option_type == "put":
            return np.exp(-r * T) * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def vega(vol):
        d1 = (np.log(F / K) + (0.5 * vol**2) * T) / (vol * np.sqrt(T))
        return np.exp(-r * T) * F * stats.norm.pdf(d1) * np.sqrt(T)

    vol_guess = 0.2
    tolerance = 1e-6
    max_iterations = 100

    for i in range(max_iterations):
        price_guess = black76_price(vol_guess)
        diff = price_guess - V
        if abs(diff) < tolerance:
            return vol_guess
        vol_guess = vol_guess - diff / vega(vol_guess)

    return -1


def find_volatility_surface_per_quote_date(df):
    # Calculate forward price for each unique expiry
    unique_expiries = df["YTE"].unique()
    forward_prices = {}

    for expiry in unique_expiries:
        expiry_df = df[df["YTE"] == expiry]
        if expiry_df.empty:
            continue
        print(f"Calculating forward price for expiry: {expiry}")
        print(expiry_df)
        forward_prices[expiry] = find_fwd_per_quote_date(expiry_df)

    # Map forward prices back to original dataframe
    df["FORWARD"] = df["YTE"].map(forward_prices)

    df["BS_VOL"] = df.apply(
        lambda r: black76_implied_volatility(
            option_price=r["C_MID"] if r["STRIKE"] > r["FORWARD"] else r["P_MID"],
            forward_price=r["FORWARD"],
            strike_price=r["STRIKE"],
            time_to_expiry=r["YTE"],
            interest_rate=0.02,
            option_type="call" if r["STRIKE"] > r["FORWARD"] else "put",
        ),
        axis=1,
    )
    df["SPOT_LOG_MONEYNESS"] = np.log(df["STRIKE"] / df["UNDERLYING_LAST"])
    df["FWD_LOG_MONEYNESS"] = np.log(df["STRIKE"] / df["FORWARD"])
    df["TOTAL_VARIANCE"] = df["BS_VOL"] ** 2 * df["YTE"]
    print(df.head())
    return df
