from process_raw_data import *
from vol_to_grid import *
from AH_vol_to_grid import *


def main():

    folder = "../optionsdx_data"
    total_post_var_date_count = 0
    total_post_vol_date_count = 0
    for year in [
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",  # low # of data point at high k / weird shape: 2018-01-10 01-29 02-05 05-01 05-23 12-24
        "2019",  # dad data (not many point) 2019-02-11 06-05 06-10 06-11 06-14 06-17 06-18 06-27 07-03 07-05 07-11 07-15 07-16 07-30 09-09 09-25 09-30 10-01 10-30
        "2020",  # market explode: 2020-03-09 , 12, 13, 16, 17, 18, 19, 20, 23
        "2021h1",
        "2021q3",  # bad date 2021-08-18, 2021-08-19, 20, 23
        "2021q4",  # bad date 2021-10-25
        "2022q1",  # bad date: 2022-01- 12, 20, 24,25
        "2022q2",  # done
        "2022q3",  # done
        "2022q4",  # done
        "2023q1",  # done
        "2023q2",  # done
        "2023q3",  # done
        "2023q4",  # done
    ][-14:-1]:
        if 0:
            k_grid = np.linspace(-0.3, 0.3, 41)
            T_grid = np.linspace(0.05, 1.0, 20)

            print(f"Processing data for year: {year}")
            # some filtering of the raw data, and add columns
            # 1. preprocessing raw data
            # load_and_process_data(folder, year)

            # 2. process vol to grid
            #process_var_to_grid(folder, year, k_grid, T_grid)

            # post processing of the vol grid data, more intense pricing test
            # Remove existing post_*.npz files before post processing
            year_folder = f"{folder}/{year}"
            if os.path.exists(year_folder):
                for filename in os.listdir(year_folder):
                    if filename.startswith("post_") and filename.endswith(".npz"):
                        file_path = os.path.join(year_folder, filename)
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
            var_arb_date_count, vol_arb_date_count = post_process_grid_data(folder, year, k_grid, T_grid)
            total_post_var_date_count += var_arb_date_count
            total_post_vol_date_count += vol_arb_date_count
    print(f"Total post processing var arb date count: {total_post_var_date_count}")
    print(f"Total post processing vol arb date count: {total_post_vol_date_count}")
    # Total post processing var arb date count: 36
    # Total post processing vol arb date count: 66

    # TODO: done the above for all year 2018-2023
    # next: package them into a data set! while counting how many we have
    # load processed data and save pack to test and train data
    years_to_read = [
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",  # done
        "2019",  # done
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
    ][-14:]

    """
    bad_dates_per_year = {
        "2018": ["2018-01-10", "2018-01-29", "2018-02-05", "2018-05-01", "2018-05-23", "2018-12-24"],
        "2019": ["2019-02-11", "2019-06-05", "2019-06-10", "2019-06-11", "2019-06-14",
                 "2019-06-17", "2019-06-18", "2019-06-27", "2019-07-03", "2019-07-05",
                 "2019-07-11", "2019-07-15", "2019-07-16", "2019-07-30", "2019-09-09",
                 "2019-09-25", "2019-09-30", "2019-10-01", "2019-10-30"],
        "2020": ["2020-03-09", "2020-03-12", "2020-03-13", "2020-03-16", "2020-03-17",
                 "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-23"],
        "2021q3": ["2021-08-18", "2021-08-19", "2021-08-20", "2021-08-23"],
        "2022q1": ["2022-01-12", "2022-01-20", "2022-01-24", "2022-01-25"],
    }
    """
    if 1:
        bad_dates = []
        # for year_dates in bad_dates_per_year.values():
        #    bad_dates.extend(year_dates)
        #label = "total_var_"
        label = "post_vol_"
        vol_surface_dict, k_grid, T_grid = get_grid_data(folder, years_to_read, label, bad_dates)
        print(f"Number of dates in vol_surface_dict: {len(vol_surface_dict)}")

        pack_grid_data_to_npz("data_pack", label, vol_surface_dict, k_grid, T_grid, train_ratio=0.8)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    running_time = end_time - start_time
    print(f"\nRunning time: {running_time:.2f} seconds")
