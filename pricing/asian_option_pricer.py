import QuantLib as ql
import numpy as np
from american_put_pricer import read_vol_data


# TODO: implement the Asian option pricer


def price_asian_option_multi_KT(quote_date, vol_surface, K_grid, T_grid, eval_KTs):
    """
    input: 1. (quote date, vol_surface), n (K,T)
    output: n NPV of Asian options
    """
    # some constants
    S0 = 1.0
    r = 0.02
    q = 0.0

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
            volMatrix[i][j] = vol_surface[i][j]

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
    engine = ql.FdBlackScholesAsianEngine(process, tGrid=400, xGrid=400, aGrid=200)  # numerical PDE with Grid=200: Number of steps for the running average of the underlying asset’s price.

    AsianC_NPVs = []
    AsianP_NPVs = []
    for i in range(len(eval_KTs)):
        K = eval_KTs[i][0]
        T = eval_KTs[i][1]
        # Create the Asian option
        maturity = today + int(T * 365 + 0.5)  # no holidays, just add days
        average_type = ql.Average.Arithmetic
        exercise = ql.EuropeanExercise(maturity)
        pastFixings = 0
        asianFixingDates = [today + x for x in range(1, int(T * 365 + 1))]  # daily fixings
        # Call
        call_payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        call_option = ql.DiscreteAveragingAsianOption(average_type, 0.0, pastFixings, asianFixingDates, call_payoff, exercise)

        call_option.setPricingEngine(engine)
        call_npv = call_option.NPV()
        # Put
        put_payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
        put_option = ql.DiscreteAveragingAsianOption(average_type, 0.0, pastFixings, asianFixingDates, put_payoff, exercise)
        put_option.setPricingEngine(engine)
        put_npv = put_option.NPV()

        AsianC_NPVs.append(call_npv)
        AsianP_NPVs.append(put_npv)

    return np.array(AsianC_NPVs), np.array(AsianP_NPVs)


def generate_AsianOption_data_set(folder, N_data, vol_data_path, label, dataset_type="train"):

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

    all_AsianC_NPVS_data = {"quote_date": [], "vol_surface": [], "K": [], "T": [], "NPV": []}
    all_AsianP_NPVS_data = {"quote_date": [], "vol_surface": [], "K": [], "T": [], "NPV": []}
    arb_date = []

    # Set random seed for reproducible sampling
    np.random.seed(42)

    for i in range(len(quote_dates)):
        # Uniform sampling in (K,T)
        K_min, K_max = min(K_grid), max(K_grid)
        T_min, T_max = min(T_grid), max(T_grid)

        eval_Ks = np.random.uniform(K_min, K_max, size=n_per_date[i])
        eval_Ts = np.random.uniform(T_min, T_max, size=n_per_date[i])

        eval_KTs = [[K, T] for K, T in zip(eval_Ks, eval_Ts)]
        if n_per_date[i] == 0:
            continue
        print(f"Evaluating {n_per_date[i]} (K,T) for quote date {quote_dates[i]}")

        try:
            AsianC_NPVs, AsianP_NPVs = price_asian_option_multi_KT(quote_dates[i], vol_surfaces[i], K_grid, T_grid, eval_KTs)
        except Exception as e:
            print(f"Error pricing Asian options for quote date {quote_dates[i]}: {e}")
            arb_date.append(quote_dates[i])
            continue
        print("AsianC_NPVs", AsianC_NPVs)
        print("AsianP_NPVs", AsianP_NPVs)

        for j in range(len(eval_KTs)):
            all_AsianC_NPVS_data["quote_date"].append(quote_dates[i])
            all_AsianC_NPVS_data["vol_surface"].append(vol_surfaces[i])
            all_AsianC_NPVS_data["K"].append(eval_KTs[j][0])
            all_AsianC_NPVS_data["T"].append(eval_KTs[j][1])
            all_AsianC_NPVS_data["NPV"].append(AsianC_NPVs[j])

            all_AsianP_NPVS_data["quote_date"].append(quote_dates[i])
            all_AsianP_NPVS_data["vol_surface"].append(vol_surfaces[i])
            all_AsianP_NPVS_data["K"].append(eval_KTs[j][0])
            all_AsianP_NPVS_data["T"].append(eval_KTs[j][1])
            all_AsianP_NPVS_data["NPV"].append(AsianP_NPVs[j])
    print(f"Processed {len(all_AsianC_NPVS_data['NPV'])} Asian Call options and {len(all_AsianP_NPVS_data['NPV'])} Asian Put options.")
    print("error quote dates:", arb_date)

    # 3. save data
    np.savez(
        f"{folder}/AsianCall_pricing_data_{dataset_type}.npz",
        quote_dates=all_AsianC_NPVS_data["quote_date"],
        vol_surfaces=all_AsianC_NPVS_data["vol_surface"],
        K=all_AsianC_NPVS_data["K"],
        T=all_AsianC_NPVS_data["T"],
        NPV=all_AsianC_NPVS_data["NPV"],
    )
    print(f"Asian Call data with {N_data} samples saved to {folder}/AsianCall_pricing_data_{dataset_type}.npz")

    np.savez(
        f"{folder}/AsianPut_pricing_data_{dataset_type}.npz",
        quote_dates=all_AsianP_NPVS_data["quote_date"],
        vol_surfaces=all_AsianP_NPVS_data["vol_surface"],
        K=all_AsianP_NPVS_data["K"],
        T=all_AsianP_NPVS_data["T"],
        NPV=all_AsianP_NPVS_data["NPV"],
    )
    print(f"Asian Put data with {N_data} samples saved to {folder}/AsianPut_pricing_data_{dataset_type}.npz")
