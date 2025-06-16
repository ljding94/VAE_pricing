from process_raw_data import *
from vol_to_grid import *

def main():

    folder = "../optionsdx_data"
    for year in [
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",  # market explode: 2020-03-09 , 12, 13, 16, 17, 18, 19, 20, 23
        #"2021h1",
        #"2021q3", # bad date 2021-08-18, 2021-08-19, 20, 23
        #"2021q4",  # bad date 2021-10-25
        #"2022q1", # bad date: 2022-01- 12, 20, 24,25
        #"2022q2", # done
        #"2022q3", # done
        #"2022q4", # done
        #"2023q1", # done
        #"2023q2", # done
        #"2023q3", # done
        #"2023q4", # done
    ][-1:]:
        print(f"Processing data for year: {year}")
        load_and_process_data(folder, year)
        process_volatility_surface(folder, year)

        k_grid = np.linspace(-0.3, 0.3, 21)
        T_grid = np.linspace(0.1, 1.0, 10)
        process_var_to_grid(folder, year, k_grid, T_grid)


    # load_and_process_data(folder, "2010")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    running_time = end_time - start_time
    print(f"\nRunning time: {running_time:.2f} seconds")
