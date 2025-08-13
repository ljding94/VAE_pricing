import time
from american_put_pricer import *
from asian_option_pricer import *
import os
import sys


# def main(product="american_put", data_type="",label=0, N_data=10, folder="../data/data_pool"):
def main():
    folder = "../data_process/data_pack"
    label = "post_vol_"


    if 0:
        print("Generating American Put training data...")
        N_data = 20000
        #N_data = 20000

        vol_data_path = f"../data_process/data_pack/{label}grid_train.npz"
        generate_AmericanPut_data_set(folder, N_data, vol_data_path, label, dataset_type="train")

        print("Generating American Put training data...")
        N_data = 4000
        #N_data = 4000
        type = "test"
        vol_data_path = f"../data_process/data_pack/{label}grid_{type}.npz"
        generate_AmericanPut_data_set(folder, N_data, vol_data_path, label, dataset_type=type)

    if 1:
        print("Generating Asian Option training data...")
        N_data = 100
        vol_data_path = f"../data_process/data_pack/{label}grid_train.npz"
        generate_AsianOption_data_set(folder, N_data, vol_data_path, label, dataset_type="train")

        print("Generating Asian Option test data...")
        N_data = 20
        vol_data_path = f"../data_process/data_pack/{label}grid_test.npz"
        generate_AsianOption_data_set(folder, N_data, vol_data_path, label, dataset_type="test")



if __name__ == "__main__":


    '''
    if len(sys.argv) == 6:
        try:
            product = sys.argv[1]
            label = int(sys.argv[2])
            N_data = int(sys.argv[3])
            data_type = sys.argv[4]
            folder = sys.argv[5]
            print("using arguments:")
            print(f"product: {product}, label: {label}, N_data: {N_data}, folder: {folder}")
        except ValueError:
            print("Invalid arguments, please use: <product> <label> <N_data> <folder>")
            sys.exit(1)
    '''

    start_time = time.time()
    #main(product, data_type, label, N_data, folder)
    main()
    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
