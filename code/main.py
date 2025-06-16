import time
from American_put import *
from variance_swap import *
import os
import sys


def main(product="american_put", data_type="",label=0, N_data=10, folder="../data/data_pool"):
    # Generate the dataset

    if product.lower() == "american_put":
        generate_american_put_data_set(folder, label=label, N_data=N_data, data_type=data_type)
    elif product.lower() == "variance_swap":
        generate_variance_swap_data(folder, N_data=N_data, data_type=data_type)


if __name__ == "__main__":

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

    start_time = time.time()
    main(product, data_type, label, N_data, folder)
    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
