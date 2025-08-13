# analyze 2D vol surface using VAE
from VAE_model import *
from torch.utils.data import Subset


def main():
    folder = "../data_process/data_pack"
    ld = 8
    if 0:
        train_and_save_VAE_alone(folder, latent_dim=ld, num_epochs=500)

    if 0:
        train_and_save_pricer(folder, product_type="AmericanPut", vae_model_path=f"{folder}/vae_state_dict.pt", latent_dim=ld, pricing_param_dim=2, num_epochs=300, num_epochs_fine_tune=200)

    if 1:
        train_and_save_pricer(folder, product_type="AsianCall", vae_model_path=f"{folder}/vae_state_dict.pt", latent_dim=ld, pricing_param_dim=2, num_epochs=300, num_epochs_fine_tune=200)


    #visualize_latent_distribution(f"{folder}/vae_state_dict.pt", folder, latent_dim=ld, save_path=f"{folder}/latent_distribution.png")

    #show_random_reconstructions(folder, f"{folder}/vae_state_dict.pt", latent_dim=ld)

    if 0:

        quote_dates = ["2020-03-10", "2021-06-15", "2023-11-29"]

        show_quote_date_reconstructions(
            folder=folder,
            quote_dates=quote_dates,
            model_path=f"{folder}/vae_state_dict.pt",
            latent_dim=ld
        )
    if 0:
        plot_loss_curves(folder)
        plot_predict_prices_from_vol_surface_and_params(
            folder=folder,
            product_type="AmericanPut",
            pricer_model_path=f"{folder}/pricer_state_dict.pt",
            include_train=True,  # This will evaluate both train and test data
            latent_dim=ld,
            pricing_param_dim=2
        )


if __name__ == "__main__":
    main()
