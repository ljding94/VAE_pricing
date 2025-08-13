from vol_illustration import *
from svd_analysis import *
from NN_plot import *
def main():
    print("plotting figures for VAE pricing paper")

    # 1. illustrative market vol vs interpolated/arbitrage free + 2d heat map illustration
    #plot_vol_illustration()

    # 2. SVD analysis of all of the vol data set
    #plot_svd_analysis()

    # 3. NN architecture plot (ppt)
    #plot_single_vol_grid()

    # 4 latent space correlation?

    #plot_latent_space()

    # 5, do I need loss curve?

    #6 , reconstruction of vol surface?
    #plot_vol_surface_reconstruction()

    # 7, American put pricing prediction?
    plot_price_prediction()





if __name__ == "__main__":
    main()