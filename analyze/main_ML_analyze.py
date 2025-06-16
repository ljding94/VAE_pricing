from ML_analyze import *
import time


def main():
    folder = "../optionsdx_data"
    years = [
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
        "2020",  # done
        "2021h1",  # done
        "2021q3",  # done
        "2021q4",  # done
        "2022q1",  # done
        "2022q2",  # done
        "2022q3",  # done
        "2022q4",  # done
        "2023q1",  # done
        "2023q2",  # done
        "2023q3",  # done
        "2023q4",  # done
    ]
    years = years[-12:]
    bad_dates = [
        # vcovid crash
        "2020-03-09",
        "2020-03-12",
        "2020-03-13",
        "2020-03-16",
        "2020-03-17",
        "2020-03-18",
        "2020-03-19",
        "2020-03-20",
        "2020-03-23",
        # some bad fitting due to lack of data
        "2021-08-18",
        "2021-08-19",
        "2021-08-20",
        "2021-08-23",
        "2021-10-25",
        "2022-01-12",
        "2022-01-20",
        "2022-01-24",
        "2022-01-25",
    ]
    print(f"Analyzing data for years: {years}")
    all_w_grid, k_grid, T_grid = get_vol_surface_data(folder, years, bad_dates)
    svd_analysis(folder, all_w_grid, k_grid, T_grid)


if __name__ == "__main__":
    # Time the execution of the main function
    start_time = time.time()

    # Run the main function from ML_analyze.py
    main()

    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
