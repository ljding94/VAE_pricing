import QuantLib as ql
import numpy as np


def read_vol_data(path, label):
    """
    will return all vol data from the npz file
    """
    data = np.load(path)
    print("Keys in the loaded data:", data.keys())

    quote_dates = data["quote_dates"]
    k_grid = data["k_grid"]
    K_grid = np.exp(k_grid)
    T_grid = data["T_grid"]

    if label == "vol_" or label == "post_vol_":
        vol_surfaces = data["surfaces_grid"]
    elif label == "total_var_" or label == "post_total_var_":
        vol_surfaces = []
        for i in range(len(data["surfaces_grid"])):
            # convert vol surface to total variance surface
            total_var_grid = data["surfaces_grid"][i]
            vol_surface = np.sqrt(total_var_grid / np.maximum(T_grid, 1e-6))  # Convert total variance to vol surface
            vol_surfaces.append(vol_surface)
    else:
        raise ValueError("Unsupported label. Use 'vol_' or 'total_var_'.")

    return data, quote_dates, vol_surfaces, K_grid, T_grid

    # data, quote_dates, vol_surfaces, K_grid, T_grid = read_vol_data("../data_process/data_pack/SPX_vol_surface_train.npz")


def price_american_put_options_multi_KT(quote_date, vol_surface, K_grid, T_grid, eval_KTs):
    """
    input: 1. (quote date, vol_surface), n (K,T)
    output: n NPV of american put options
    """
    # some constants
    S0 = 1.0
    r = 0.02
    q = 0.0
    # K_grid is 41
    # T_grid is 20
    # K_mesh: 20*41
    # vol_surface  is 41*20
    print("np.shape(vol_surface), np.shape(K_grid), np.shape(T_grid)")
    print(np.shape(vol_surface), np.shape(K_grid), np.shape(T_grid))

    # environment setup
    today = ql.Date(*map(int, quote_date.split("-")[::-1]))
    ql.Settings.instance().evaluationDate = today
    calendar = ql.NullCalendar()
    dayCounter = ql.Actual365Fixed()

    T_grid_expiry_dates = [today + int(T * 365 + 0.5) for T in T_grid]
    exact_ts = [dayCounter.yearFraction(today, d) for d in T_grid_expiry_dates]

    # build vol surface object
    volMatrix = ql.Matrix(len(K_grid), len(T_grid))
    for i in range(len(K_grid)):
        for j in range(len(T_grid)):
            # volMatrix[i][j] = vol_surface[j, i] # previous
            volMatrix[i][j] = vol_surface[i, j]  # after fix meshgrid order

    # Adjust volMatrix to make internal variances match cleaned total_var exactly
    for i in range(len(K_grid)):
        for j in range(len(T_grid)):
            if exact_ts[j] > 0:
                adjustment_factor = np.sqrt(T_grid[j] / exact_ts[j])
                volMatrix[i][j] *= adjustment_factor
            else:
                volMatrix[i][j] = 0.0  # Rare edge case for T=0

    BlackSurf = ql.BlackVarianceSurface(today, calendar, T_grid_expiry_dates, K_grid, volMatrix, dayCounter)

    volTS = ql.BlackVolTermStructureHandle(BlackSurf)
    volTS.enableExtrapolation()

    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    ratesTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dayCounter))
    divTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dayCounter))

    process = ql.BlackScholesMertonProcess(spot, divTS, ratesTS, volTS)
    engine = ql.FdBlackScholesVanillaEngine(process, 400, 400)  # numerical PDE engine

    AmericanP_NPVs = []
    for i in range(len(eval_KTs)):
        K = eval_KTs[i][0]
        T = eval_KTs[i][1]
        # print(f"Evaluating option {i+1}/{len(eval_KTs)}: K={K}, T={T}")
        maturity = today + int(T * 365 + 0.5)  # no holidays, just add days
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
        exercise = ql.AmericanExercise(today, maturity)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)
        NPV = option.NPV()
        # print(f"Quote date: {quote_date}, K: {K:.4f}, T: {T:.4f}, NPV: {NPV:.6f}")
        AmericanP_NPVs.append(NPV)

    return np.array(AmericanP_NPVs)


def generate_AmericanPut_data_set(folder, N_data, vol_data_path, label, dataset_type="test"):

    # 1. read all vol data
    data, quote_dates, vol_surfaces, K_grid, T_grid = read_vol_data(vol_data_path, label)
    """
    data prepared
    k_grid = np.linspace(-0.3, 0.3, 41)
    T_grid = np.linspace(0.05, 1.0, 20)
    """
    # 2. n (K,T) to eval for each quote date
    n = int(N_data / len(quote_dates))
    n_per_date = [n] * len(quote_dates)
    remainder = N_data % len(quote_dates)
    for i in range(remainder):
        n_per_date[i] += 1
    print("n_per_date", n_per_date)

    all_AmericanP_NPVS_data = {"quote_date": [], "vol_surface": [], "K": [], "T": [], "NPV": []}
    arb_date = []

    # Set random seed for reproducible sampling (same as in test_pricing_arbitrage_free_surface)
    np.random.seed(42)

    for i in range(len(quote_dates)):
        # Use same sampling strategy as test_pricing_arbitrage_free_surface
        K_min, K_max = min(K_grid), max(K_grid)
        T_min, T_max = min(T_grid), max(T_grid)

        eval_Ks = np.random.uniform(K_min, K_max, size=n_per_date[i])
        eval_Ts = np.random.uniform(T_min, T_max, size=n_per_date[i])
        """
        if dataset_type == "train":


        elif dataset_type == "test":
            K_mid = (K_min + K_max) / 2
            K_half_range = (K_max - K_min) * 0.45
            eval_Ks = np.random.uniform(K_mid - K_half_range, K_mid + K_half_range, size=n_per_date[i])

            T_mid = (T_min + T_max) / 2
            T_half_range = (T_max - T_min) * 0.45
            eval_Ts = np.random.uniform(T_mid - T_half_range, T_mid + T_half_range, size=n_per_date[i])
        """

        eval_KTs = [[K, T] for K, T in zip(eval_Ks, eval_Ts)]
        if n_per_date[i] == 0:
            continue
        print(f"Evaluating {n_per_date[i]} (K,T) for quote date {quote_dates[i]}")
        # print("eval_Ks", eval_Ks)
        # print("eval_Ts", eval_Ts)
        # print("eval_KTs", eval_KTs)
        try:
            AmericanP_NPVS = price_american_put_options_multi_KT(quote_dates[i], vol_surfaces[i], K_grid, T_grid, eval_KTs)
        except Exception as e:
            print(f"Error processing quote date {quote_dates[i]}: {e}")
            arb_date.append(quote_dates[i])
            continue
        print("AmericanP_NPVS", AmericanP_NPVS)
        for j in range(len(AmericanP_NPVS)):
            all_AmericanP_NPVS_data["quote_date"].append(quote_dates[i])
            all_AmericanP_NPVS_data["vol_surface"].append(vol_surfaces[i])
            all_AmericanP_NPVS_data["K"].append(eval_KTs[j][0])
            all_AmericanP_NPVS_data["T"].append(eval_KTs[j][1])
            all_AmericanP_NPVS_data["NPV"].append(AmericanP_NPVS[j])
    print(f"Processed {len(all_AmericanP_NPVS_data['quote_date'])} American put options.")
    print("error dates:", len(arb_date), arb_date)

    # 3. save data
    np.savez(
        f"{folder}/AmericanPut_pricing_data_{dataset_type}.npz",
        quote_dates=all_AmericanP_NPVS_data["quote_date"],
        vol_surfaces=all_AmericanP_NPVS_data["vol_surface"],
        K=all_AmericanP_NPVS_data["K"],
        T=all_AmericanP_NPVS_data["T"],
        NPV=all_AmericanP_NPVS_data["NPV"],
    )
    print(f"American put data with {N_data} samples saved to {folder}/AmericanPut_pricing_data_{dataset_type}.npz")
    return 0
