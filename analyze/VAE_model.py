# all NN function for VAE
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ML_analyze import *
from sklearn.decomposition import PCA
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR  # ← NEW import

# --------------------------
# Data Loading Functions
# --------------------------


class VolsurfaceDataset(Dataset):
    def __init__(self, folder, label, data_type, compute_stats=False):
        # start with simple parsing
        # get raw vol data
        vol_data_path = f"{folder}/{label}grid_{data_type}.npz"
        vol_data = np.load(vol_data_path)

        self.k_grid = vol_data["k_grid"]
        self.T_grid = vol_data["T_grid"]
        self.quote_dates = vol_data["quote_dates"]
        self.surfaces_grid = vol_data["surfaces_grid"]

        # Normalization statistics file path
        self.stats_path = f"{folder}/vol_normalization_stats.npz"

        if compute_stats:
            # Compute normalization statistics from the data
            self.norm_mean = np.mean(self.surfaces_grid)
            self.norm_std = np.std(self.surfaces_grid)
            # Save to file
            np.savez(self.stats_path, mean=self.norm_mean, std=self.norm_std)
            print(f"Computed normalization stats: mean={self.norm_mean:.6f}, std={self.norm_std:.6f}")
            print(f"Saved normalization stats to {self.stats_path}")
        else:
            # Load existing normalization statistics
            if os.path.exists(self.stats_path):
                stats = np.load(self.stats_path)
                self.norm_mean = stats["mean"].item()
                self.norm_std = stats["std"].item()
                print(f"Loaded normalization stats: mean={self.norm_mean:.6f}, std={self.norm_std:.6f}")
            else:
                # If no stats file exists, compute them (fallback)
                print(f"Warning: No normalization stats found at {self.stats_path}, computing from current data")
                self.norm_mean = np.mean(self.surfaces_grid)
                self.norm_std = np.std(self.surfaces_grid)
                np.savez(self.stats_path, mean=self.norm_mean, std=self.norm_std)

        # Normalize the data
        self.surfaces_grid = (self.surfaces_grid - self.norm_mean) / self.norm_std

    def __len__(self):
        return len(self.quote_dates)

    def __getitem__(self, idx):
        surface_grid = self.surfaces_grid[idx]
        surface_grid = torch.from_numpy(surface_grid).float().unsqueeze(0)  # add channel
        k_grid = torch.tensor(self.k_grid, dtype=torch.float32)
        T_grid = torch.tensor(self.T_grid, dtype=torch.float32)
        return surface_grid, k_grid, T_grid


def create_dataloader(folder, label, data_type, batch_size=32, shuffle=True, compute_stats=False):
    dataset = VolsurfaceDataset(folder, label, data_type, compute_stats=compute_stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset


class PricingDataset(Dataset):
    def __init__(self, folder, product_type, data_type, compute_param_stats=False):
        """
        Dataset for pricing data that contains vol surfaces and corresponding option prices.

        Args:
            folder: Path to folder containing pricing data
            label: Label prefix for the data files (e.g., "AmericanP_")
            data_type: Type of data ("train" or "test")
        """
        # Load pricing data
        pricing_data_path = f"{folder}/{product_type}_pricing_data_{data_type}.npz"
        print(f"Loading pricing data from {pricing_data_path}")
        pricing_data = np.load(pricing_data_path, allow_pickle=True)

        # Extract data arrays
        if product_type == "AmericanPut" or product_type == "AsianCall" or product_type == "AsianPut":
            # Print header information for debugging
            print(f"Loading {product_type} pricing data from {pricing_data_path}")
            print("Available keys in pricing data:", list(pricing_data.keys()))
            print("Data shapes:")
            for key in pricing_data.keys():
                if hasattr(pricing_data[key], "shape"):
                    print(f"  {key}: {pricing_data[key].shape}")
                else:
                    print(f"  {key}: {type(pricing_data[key])}")
            self.quote_dates = pricing_data["quote_dates"]
            self.vol_surfaces = pricing_data["vol_surfaces"]
            self.price_params = np.column_stack((pricing_data["K"], pricing_data["T"]))  # Combine K and T into a single array
            self.prices = pricing_data["NPV"]

        else:
            raise ValueError(f"Unsupported product type: {product_type}")

        # Load normalization statistics for vol surfaces
        vol_stats_path = f"{folder}/vol_normalization_stats.npz"
        if os.path.exists(vol_stats_path):
            stats = np.load(vol_stats_path)
            self.norm_mean = stats["mean"].item()
            self.norm_std = stats["std"].item()
            print(f"Loaded normalization stats: mean={self.norm_mean:.6f}, std={self.norm_std:.6f}")
        else:
            raise FileNotFoundError(f"Normalization stats not found at {vol_stats_path}")

        # Normalize vol surfaces
        self.vol_surfaces = (self.vol_surfaces - self.norm_mean) / self.norm_std

        # Normalize pricing parameters and prices
        param_stats_path = f"{folder}/{product_type}_pricing_param_stats.npz"
        if compute_param_stats:
            self.price_params_mean = np.mean(self.price_params, axis=0)
            self.price_params_std = np.std(self.price_params, axis=0)
            self.price_mean = np.mean(self.prices)
            self.price_std = np.std(self.prices)

            price_param_stats = {
                "params_mean": self.price_params_mean,
                "params_std": self.price_params_std,
                "price_mean": self.price_mean,
                "price_std": self.price_std,
            }
            # save to file
            np.savez(param_stats_path, **price_param_stats)
            print(f"Computed pricing parameter stats: mean={self.price_params_mean}, std={self.price_params_std}")
            print(f"Computed price stats: mean={self.price_mean:.6f}, std={self.price_std:.6f}")
            print(f"Saved pricing parameter stats to {param_stats_path}")

        else:
            # load existing parameter statistics
            if os.path.exists(param_stats_path):
                stats = np.load(param_stats_path)

                self.price_params_mean = stats["params_mean"]
                self.price_params_std = stats["params_std"]
                self.price_mean = stats["price_mean"]
                self.price_std = stats["price_std"]
                print(f"Loaded pricing parameter stats: mean={self.price_params_mean}, std={self.price_params_std}")
                print(f"Loaded price stats: mean={self.price_mean:.6f}, std={self.price_std:.6f}")
            else:
                warnings.warn(f"No pricing parameter stats found at {param_stats_path}, computing from current data")

        # Apply normalization to pricing parameters and prices
        self.price_params = (self.price_params - self.price_params_mean) / self.price_params_std
        self.prices = (self.prices - self.price_mean) / self.price_std

        print(f"Loaded {len(self.quote_dates)} pricing data samples")
        print(f"Price params range: min={np.min(self.price_params, axis=0)}, max={np.max(self.price_params, axis=0)}")
        print(f"Price range: [{np.min(self.prices):.3f}, {np.max(self.prices):.3f}]")

    def __len__(self):
        return len(self.quote_dates)

    def __getitem__(self, idx):
        vol_surface = torch.from_numpy(self.vol_surfaces[idx]).float().unsqueeze(0)  # add channel dimension
        pricing_param = torch.tensor(self.price_params[idx], dtype=torch.float32)
        price = torch.tensor(self.prices[idx], dtype=torch.float32).unsqueeze(0)  # make it (1,) for consistency
        return vol_surface, pricing_param, price


def create_pricing_dataloader(folder, label, data_type, batch_size=32, shuffle=True, compute_stats=False):
    """Create a DataLoader for pricing data."""
    dataset = PricingDataset(folder, label, data_type, compute_param_stats=compute_stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset

# --------------------------
# Model Definitions
# --------------------------
"""
Input (10×21) ──► Encoder ──► z  ──► Decoder ──► x̂
                           │
                           └──► Regressor g(z) ──► ŷ
Loss =  MSE(x̂,x)  +  λ·MSE(ŷ,y)  +  β·KL
"""


# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self, latent_dim=8, input_shape=(41, 20)):
        super().__init__()
        self.input_shape = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # (41,20)->(21,10)
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # (21,10)->(11,5)
        )

        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            conv_output = self.conv(dummy_input)
            self.flat = conv_output.numel()

        self.fc_mu = nn.Linear(self.flat, latent_dim)
        self.fc_logvar = nn.Linear(self.flat, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


# ---------- Decoder ----------
class Decoder(nn.Module):
    def __init__(self, latent_dim=8, output_shape=(41, 20)):
        super().__init__()
        self.output_shape = output_shape

        # Calculate intermediate dimensions after convolutions
        # We need to work backwards from output_shape to find conv dimensions
        # For (41, 20) output, we need conv output of approximately (21, 10)
        # which means we need (11, 5) before the first transpose conv
        self.conv_h, self.conv_w = 11, 5
        self.flat = 32 * self.conv_h * self.conv_w

        self.fc = nn.Sequential(nn.Linear(latent_dim, self.flat), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # (11,5) -> (21,10)
            nn.ConvTranspose2d(16, 1, 4, 2, 1),  # (21,10) -> (41,20)
            # No sigmoid - output normalized data that will be denormalized later
        )

    def forward(self, z):
        # Handle both single samples and batched samples from VAE forward
        original_shape = z.shape
        if len(original_shape) == 3:  # [num_samples, batch_size, latent_dim]
            num_samples, batch_size, latent_dim = original_shape
            z = z.view(-1, latent_dim)  # Flatten to [num_samples * batch_size, latent_dim]
            h = self.fc(z).view(-1, 32, self.conv_h, self.conv_w)
            deconv_out = self.deconv(h)
            # Crop or pad to exact output shape if needed
            deconv_out = deconv_out[..., : self.output_shape[0], : self.output_shape[1]]
            # Reshape back to [num_samples, batch_size, 1, H, W]
            return deconv_out.view(num_samples, batch_size, 1, self.output_shape[0], self.output_shape[1])
        else:  # [batch_size, latent_dim] - normal case
            h = self.fc(z).view(z.size(0), 32, self.conv_h, self.conv_w)
            deconv_out = self.deconv(h)
            # Crop or pad to exact output shape if needed
            return deconv_out[..., : self.output_shape[0], : self.output_shape[1]]


class VAE(nn.Module):
    def __init__(self, latent_dim=8, extra_dim=0, out_dim=None, input_shape=(41, 20)):
        super().__init__()
        self.encoder = Encoder(latent_dim, input_shape)
        self.decoder = Decoder(latent_dim, input_shape)

    @staticmethod
    def reparameterise(mu, logvar):
        std, eps = (0.5 * logvar).exp(), torch.randn_like(logvar)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        epsilons = torch.randn(10, *mu.shape, device=mu.device)
        z_samples = mu.unsqueeze(0) + epsilons * (0.5 * logvar).exp().unsqueeze(0)
        recons = self.decoder(z_samples)
        recon_avg = recons.mean(dim=0)
        return recon_avg, mu, logvar


class Pricer(nn.Module):
    """
    The pricer model takes in the vol surface + pricing parameters
    The vol surface goes to the encoder to find mu and logvar, which gives z
    z and pricing parameters together go through a MLP to get the price
    Resample multiple z to get the ensemble average price
    """

    def __init__(self, latent_dim=8, pricing_param_dim=2, vol_input_shape=(41, 20)):
        super().__init__()
        self.latent_dim = latent_dim
        self.pricing_param_dim = pricing_param_dim

        # Use the same encoder as VAE, plain encoder, need to load the weights from trained VAE
        self.encoder = Encoder(latent_dim, vol_input_shape)

        # MLP for pricing: takes latent + pricing params -> price
        '''
        self.pricing_mlp = nn.Sequential(
            nn.Linear(latent_dim + pricing_param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )  # single price output
        '''
        self.pricing_mlp = nn.Sequential(
            nn.Linear(latent_dim + pricing_param_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )  # single price output

    def forward(self, vol_surface, pricing_params):
        """
        Args:
            vol_surface: (batch_size, 1, H, W) - normalized vol surface
            pricing_params: (batch_size, pricing_param_dim) - pricing parameters

        Returns:
            price_mean: (batch_size, 1) - ensemble average price
            price_std: (batch_size, 1) - standard deviation of price samples
            mu: (batch_size, latent_dim) - latent mean
            logvar: (batch_size, latent_dim) - latent log variance
        """
        batch_size = vol_surface.size(0)

        # Encode vol surface to get latent distribution
        mu, logvar = self.encoder(vol_surface)
        n_sample = 10
        epsilons = torch.randn(n_sample, *mu.shape, device=mu.device)
        z_samples = mu.unsqueeze(0) + epsilons * (0.5 * logvar).exp().unsqueeze(0)

        # Repeat pricing params for each sample
        pricing_params_repeated = pricing_params.unsqueeze(0).repeat(n_sample, 1, 1)  # (num_samples, batch_size, pricing_param_dim)

        """
        print("===========checking pricer=========")
        print("z_samples shape:", z_samples.shape)
        print("pricing_params_repeated shape:", pricing_params_repeated.shape)
        print("===========done checking pricer=========")
        """

        # Concatenate latent samples with pricing params along the feature dimension
        mlp_input = torch.cat([z_samples, pricing_params_repeated], dim=2)  # (num_samples, batch_size, latent_dim + pricing_param_dim)

        # Reshape for MLP: (num_samples * batch_size, latent_dim + pricing_param_dim)
        mlp_input = mlp_input.view(-1, self.latent_dim + self.pricing_param_dim)

        # Get price predictions
        prices = self.pricing_mlp(mlp_input)  # (num_samples * batch_size, 1)

        # Reshape back to (num_samples, batch_size, 1) and average over samples
        prices = prices.view(n_sample, batch_size, 1)
        price_avg = prices.mean(dim=0)  # Average over samples -> (batch_size, 1)

        return price_avg

    def load_vae_encoder(self, vae_model_path, device=None):
        """Load encoder weights from a pre-trained VAE model"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load VAE state dict
        vae_state = torch.load(vae_model_path, map_location=device)

        # Extract encoder weights and remove "encoder." prefix
        encoder_state = {}
        for key, value in vae_state.items():
            if key.startswith("encoder."):
                # Remove "encoder." prefix to match the encoder's state dict structure
                new_key = key[8:]  # Remove "encoder." (8 characters)
                encoder_state[new_key] = value

        # Load into our encoder
        self.encoder.load_state_dict(encoder_state, strict=True)
        print(f"Loaded encoder weights from {vae_model_path}")

    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning only the pricing MLP"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for end-to-end training"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder parameters unfrozen")


def denormalize_surface(normalized_surface, mean, std):
    """Denormalize a surface using provided mean and std."""
    return normalized_surface * std + mean


def normalize_surface(surface, mean, std):
    """Normalize a surface using provided mean and std."""
    return (surface - mean) / std


def vae_loss(x, recon):
    # TODO: add weight on losses such that k->0 will have more weight on recon_loss
    # may not need the above for now
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    # kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    loss = recon_loss
    return loss, recon_loss


# Training function for standalone β-VAE with train/test loss recording


def train_and_save_VAE_alone(
    folder: str,
    latent_dim: int = 6,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 20,
) -> tuple:
    """
    Train a standalone β-VAE on SPX vol surface data,
    record train/test losses, and save model & losses to disk.

    Args:
        folder: Path to folder containing SPX_vol_surface_train.npz and SPX_vol_surface_test.npz
        folder: Directory where the model and loss files will be saved
        latent_dim: Dimensionality of the latent code
        batch_size: Training batch size
        num_epochs: Number of epochs to train
        lr: Learning rate for Adam optimizer
        weight_decay: Weight decay for Adam
        warmup_epochs: Number of epochs to warm up β

    Returns:
        model: Trained VAE instance
        train_losses: List of average training losses per epoch
        test_losses: List of average test losses per epoch
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure save directory exists
    os.makedirs(folder, exist_ok=True)

    # Data loaders - compute stats for training data, use existing stats for test data
    train_loader, train_dataset = create_dataloader(folder, "post_vol_", "train", batch_size=batch_size, shuffle=True, compute_stats=True)
    test_loader, test_dataset = create_dataloader(folder, "post_vol_", "test", batch_size=batch_size, shuffle=False, compute_stats=False)

    # Determine input shape from first batch
    first_batch, _, _ = next(iter(train_loader))
    input_shape = first_batch.shape[2:]  # Remove batch and channel dimensions
    print(f"Detected input shape: {input_shape}")
    print(f"Normalized data range: min={first_batch.min():.6f}, max={first_batch.max():.6f}")
    print(f"Normalized data mean: {first_batch.mean():.6f}, std: {first_batch.std():.6f}")

    # Model, optimizer, scheduler
    model = VAE(latent_dim=latent_dim, input_shape=input_shape).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler will anneal LR from `lr` → `lr*lr_min_mult` over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)  # full period is the training run  # final learning rate

    train_losses, test_losses = [], []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = total_train_recon = total_train_kl = 0.0

        for x, _, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, recon_l = vae_loss(x, recon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_train_loss += loss.item() * bs
            total_train_recon += recon_l.item() * bs

        # Step the scheduler after all optimizer steps for this epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        total_test_loss = total_test_recon = total_test_kl = 0.0
        with torch.no_grad():
            for x_test, _, _ in test_loader:
                x_test = x_test.to(device)
                recon_t, mu_t, logvar_t = model(x_test)
                loss_t, recon_t_l = vae_loss(x_test, recon_t)

                bs = x_test.size(0)
                total_test_loss += loss_t.item() * bs
                total_test_recon += recon_t_l.item() * bs

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}"
            f"| LR={current_lr:.2e} "
            f"| train_loss={avg_train_loss:.8f} "
            f"| test_loss={avg_test_loss:.8f} "
            f"| train_recon={total_train_recon/len(train_loader.dataset):.8f} "
            f"| test_recon={total_test_recon/len(test_loader.dataset):.8f} "
            f"| train_kl={total_train_kl/len(train_loader.dataset):.8f} "
            f"| test_kl={total_test_kl/len(test_loader.dataset):.8f}"
        )

    # Save model state dict and loss histories
    state_path = os.path.join(folder, "vae_state_dict.pt")
    torch.save(model.state_dict(), state_path)
    np.save(os.path.join(folder, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(folder, "test_losses.npy"), np.array(test_losses))

    print(f"Saved model state to {state_path}")
    print(f"Saved train/test losses to {folder}")

    return model, train_losses, test_losses


def train_and_save_pricer(
    folder: str,
    product_type: str,
    vae_model_path: str,
    latent_dim: int = 6,
    pricing_param_dim: int = 2,
    batch_size: int = 32,
    num_epochs: int = 100,
    num_epochs_fine_tune: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple:
    """
    Train a Pricer model that uses a pre-trained VAE encoder.

    Args:
        folder: Path to folder containing vol surface data and pricing data
        vae_model_path: Path to pre-trained VAE model state dict
        latent_dim: Dimensionality of the latent space
        pricing_param_dim: Dimensionality of pricing parameters
        batch_size: Training batch size
        num_epochs: Number of epochs to train
        lr: Learning rate for Adam optimizer
        weight_decay: Weight decay for Adam
        freeze_encoder: Whether to freeze the encoder weights during training

    Returns:
        model: Trained Pricer instance
        train_losses: List of average training losses per epoch
        test_losses: List of average test losses per epoch
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure save directory exists
    os.makedirs(folder, exist_ok=True)

    # Load pricing data
    train_loader, train_dataset = create_pricing_dataloader(folder, product_type, "train", batch_size=batch_size, shuffle=True, compute_stats=True)
    test_loader, test_dataset = create_pricing_dataloader(folder, product_type, "test", batch_size=batch_size, shuffle=False, compute_stats=False)

    # Determine input shape from first batch
    vol_first_batch, pricing_param_first_batch, price_first_batch = next(iter(train_loader))
    vol_input_shape = vol_first_batch.shape[2:]  # Remove batch and channel dimensions
    print(f"Detected vol surface input shape: {vol_input_shape}")
    price_param_input_shape = pricing_param_first_batch.shape[1]  # Should match pricing_param_dim
    print(f"Detected pricing parameter input shape: {price_param_input_shape}")

    # Create pricer model
    model = Pricer(latent_dim=latent_dim, pricing_param_dim=pricing_param_dim, vol_input_shape=vol_input_shape).to(device)

    # Load pre-trained VAE encoder weights
    model.load_vae_encoder(vae_model_path, device)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # step1, train pricer with VAE encoder frozen
    model.freeze_encoder()
    print("Encoder parameters frozen for initial training")

    # Prepare pricing parameters and target prices
    train_pricing_params = torch.tensor(train_dataset.price_params, dtype=torch.float32)
    train_prices = torch.tensor(train_dataset.prices, dtype=torch.float32).unsqueeze(1)  # Make it (batch_size, 1)
    test_pricing_params = torch.tensor(test_dataset.price_params, dtype=torch.float32)
    test_prices = torch.tensor(test_dataset.prices, dtype=torch.float32).unsqueeze(1)

    train_losses, test_losses = [], []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (vol_surface, _, _) in enumerate(train_loader):
            vol_surface = vol_surface.to(device)
            batch_size_actual = vol_surface.size(0)

            # Get corresponding pricing parameters and target prices for this batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size_actual
            pricing_params = train_pricing_params[start_idx:end_idx].to(device)
            target_prices = train_prices[start_idx:end_idx].to(device)

            # Forward pass
            predicted_prices = model(vol_surface, pricing_params)

            # Compute loss (MSE between predicted and target prices)
            loss = F.mse_loss(predicted_prices, target_prices)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_size_actual

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (vol_surface_test, _, _) in enumerate(test_loader):
                vol_surface_test = vol_surface_test.to(device)
                batch_size_actual = vol_surface_test.size(0)

                # Get corresponding test pricing parameters and target prices
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size_actual
                pricing_params_test = test_pricing_params[start_idx:end_idx].to(device)
                target_prices_test = test_prices[start_idx:end_idx].to(device)

                # Forward pass
                predicted_prices_test = model(vol_surface_test, pricing_params_test)

                # Compute loss
                loss_test = F.mse_loss(predicted_prices_test, target_prices_test)
                total_test_loss += loss_test.item() * batch_size_actual

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.6f} " f"| test_loss={avg_test_loss:.6f}")

    # step 2, unfreeze encoder and continue training as fine tuning
    model.unfreeze_encoder()
    train_losses_fine_tune, test_losses_fine_tune = [], []
    print("Encoder parameters unfrozen for fine-tuning")

    # Fine-tuning loop
    for eposh in range(num_epochs_fine_tune):
        model.train()
        total_train_loss = 0.0
        for batch_idx, (vol_surface, _, _) in enumerate(train_loader):
            vol_surface = vol_surface.to(device)
            batch_size_actual = vol_surface.size(0)
            # Get corresponding pricing parameters and target prices for this batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size_actual
            pricing_params = train_pricing_params[start_idx:end_idx].to(device)
            target_prices = train_prices[start_idx:end_idx].to(device)

            # Forward pass
            predicted_prices = model(vol_surface, pricing_params)

            # Compute loss (MSE between predicted and target prices)
            loss = F.mse_loss(predicted_prices, target_prices)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_size_actual

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses_fine_tune.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (vol_surface_test, _, _) in enumerate(test_loader):
                vol_surface_test = vol_surface_test.to(device)
                batch_size_actual = vol_surface_test.size(0)

                # Get corresponding test pricing parameters and target prices
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size_actual
                pricing_params_test = test_pricing_params[start_idx:end_idx].to(device)
                target_prices_test = test_prices[start_idx:end_idx].to(device)

                # Forward pass
                predicted_prices_test = model(vol_surface_test, pricing_params_test)

                # Compute loss
                loss_test = F.mse_loss(predicted_prices_test, target_prices_test)
                total_test_loss += loss_test.item() * batch_size_actual

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses_fine_tune.append(avg_test_loss)

        print(f"Fine-tuning Epoch {eposh+1}/{num_epochs_fine_tune} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.6f} " f"| test_loss={avg_test_loss:.6f}")

    # Combine losses
    train_losses.extend(train_losses_fine_tune)
    test_losses.extend(test_losses_fine_tune)

    # Save model state dict and loss histories
    state_path = os.path.join(folder, "pricer_state_dict.pt")
    torch.save(model.state_dict(), state_path)
    np.save(os.path.join(folder, "pricer_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(folder, "pricer_test_losses.npy"), np.array(test_losses))

    print(f"Saved pricer model state to {state_path}")
    print(f"Saved pricer train/test losses to {folder}")

    return model, train_losses, test_losses


def plot_loss_curves(folder: str):
    """
    -----------------------------------------------------------
    Plot train / test loss versus *epoch* for both VAE and Pricer models.
    Automatically detects and includes pricer loss curves if available.
    Parameters
    ----------
    folder     : directory that contains the saved .npy files
    -----------------------------------------------------------
    """
    # Load VAE losses
    train_file = os.path.join(folder, "train_losses.npy")
    test_file = os.path.join(folder, "test_losses.npy")
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        raise FileNotFoundError("Could not find one or both VAE loss files " f"({train_file}, {test_file})")

    train_losses = np.load(train_file)
    test_losses = np.load(test_file)

    # Try to load pricer loss files
    pricer_train_file = os.path.join(folder, "pricer_train_losses.npy")
    pricer_test_file = os.path.join(folder, "pricer_test_losses.npy")

    pricer_exists = os.path.exists(pricer_train_file) and os.path.exists(pricer_test_file)
    if pricer_exists:
        pricer_train_losses = np.load(pricer_train_file)
        pricer_test_losses = np.load(pricer_test_file)
        print("Found pricer loss files - including in plot")
    else:
        print("Pricer loss files not found - showing only VAE losses")

    # Create subplots - always try to show pricer if available
    if pricer_exists:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # Plot VAE losses
    epochs = np.arange(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="train", linewidth=2, color="blue")
    ax1.plot(epochs, test_losses, label="test", linewidth=2, linestyle="--", color="lightblue")
    ax1.set_yscale("log")  # log scale for better visibility
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("VAE Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot pricer losses if available
    if pricer_exists:
        pricer_epochs = np.arange(1, len(pricer_train_losses) + 1)
        ax2.plot(pricer_epochs, pricer_train_losses, label="train", linewidth=2, color="red")
        ax2.plot(pricer_epochs, pricer_test_losses, label="test", linewidth=2, linestyle="--", color="lightcoral")
        ax2.set_yscale("log")  # log scale for better visibility
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Pricer Loss Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{folder}/loss_curve.png", dpi=300)
    plt.show()
    plt.close()


def visualize_latent_distribution(model_path: str, folder: str, latent_dim: int = 6, save_path: str = None):
    """
    Visualize the distribution of latent variables from training data using a trained encoder.
    Args:
        model_path: Path to saved model state dict
        folder: Path to folder containing training data
        latent_dim: Dimensionality of the latent space
        save_path: Optional path to save the visualization
        max_samples: Maximum number of samples to use for visualization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training data to determine input shape
    train_loader, _ = create_dataloader(folder, "post_vol_", "train", batch_size=32, shuffle=False, compute_stats=False)
    first_batch, _, _ = next(iter(train_loader))
    input_shape = first_batch.shape[2:]  # Remove batch and channel dimensions

    # Load trained model
    model = VAE(latent_dim=latent_dim, input_shape=input_shape)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    # Extract latent representations
    latent_mus = []
    latent_logvars = []
    with torch.no_grad():
        sample_count = 0
        for x, _, _ in train_loader:
            x = x.to(device)
            mu, logvar = model.encoder(x)
            latent_mus.append(mu.cpu().numpy())
            latent_logvars.append(logvar.cpu().numpy())
            sample_count += x.size(0)
    latent_mus = np.concatenate(latent_mus, axis=0)
    latent_logvars = np.concatenate(latent_logvars, axis=0)
    # Create visualization
    if latent_dim <= 2:
        # Direct 2D visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # Scatter plot for mu
        axes[0, 0].scatter(latent_mus[:, 0], latent_mus[:, 1], alpha=0.6)
        axes[0, 0].set_xlabel("Mu Dim 1")
        axes[0, 0].set_ylabel("Mu Dim 2")
        axes[0, 0].set_title("Latent Mu Distribution")
        # Scatter plot for logvar
        axes[0, 1].scatter(latent_logvars[:, 0], latent_logvars[:, 1], alpha=0.6, color="orange")
        axes[0, 1].set_xlabel("LogVar Dim 1")
        axes[0, 1].set_ylabel("LogVar Dim 2")
        axes[0, 1].set_title("Latent LogVar Distribution")
        # Marginal distributions for mu
        axes[1, 0].hist(latent_mus[:, 0], bins=50, alpha=0.7, label="Mu Dim 1")
        if latent_dim == 2:
            axes[1, 0].hist(latent_mus[:, 1], bins=50, alpha=0.7, label="Mu Dim 2")
        axes[1, 0].set_xlabel("Mu Value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Mu Marginal Distributions")
        axes[1, 0].legend()
        # Marginal distributions for logvar
        axes[1, 1].hist(latent_logvars[:, 0], bins=50, alpha=0.7, label="LogVar Dim 1", color="orange")
        if latent_dim == 2:
            axes[1, 1].hist(latent_logvars[:, 1], bins=50, alpha=0.7, label="LogVar Dim 2", color="red")
        axes[1, 1].set_xlabel("LogVar Value")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("LogVar Marginal Distributions")
        axes[1, 1].legend()
    else:
        # For higher dimensions, use PCA and show multiple visualizations
        pca_mu = PCA(n_components=min(2, latent_dim))
        pca_logvar = PCA(n_components=min(2, latent_dim))
        mu_2d = pca_mu.fit_transform(latent_mus)
        logvar_2d = pca_logvar.fit_transform(latent_logvars)
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        # PCA 2D projection for mu
        axes[0, 0].scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.6)
        axes[0, 0].set_xlabel(f"PC1 ({pca_mu.explained_variance_ratio_[0]:.2%} var)")
        axes[0, 0].set_ylabel(f"PC2 ({pca_mu.explained_variance_ratio_[1]:.2%} var)")
        axes[0, 0].set_title("PCA Projection of Latent Mu")
        # PCA 2D projection for logvar
        axes[0, 1].scatter(logvar_2d[:, 0], logvar_2d[:, 1], alpha=0.6, color="orange")
        axes[0, 1].set_xlabel(f"PC1 ({pca_logvar.explained_variance_ratio_[0]:.2%} var)")
        axes[0, 1].set_ylabel(f"PC2 ({pca_logvar.explained_variance_ratio_[1]:.2%} var)")
        axes[0, 1].set_title("PCA Projection of Latent LogVar")
        # Correlation matrix for mu
        corr_matrix_mu = np.corrcoef(latent_mus.T)
        im1 = axes[1, 0].imshow(corr_matrix_mu, cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, 0].set_title("Mu Dimension Correlations")
        axes[1, 0].set_xlabel("Latent Dimension")
        axes[1, 0].set_ylabel("Latent Dimension")
        plt.colorbar(im1, ax=axes[1, 0])
        # Correlation matrix for logvar
        corr_matrix_logvar = np.corrcoef(latent_logvars.T)
        im2 = axes[1, 1].imshow(corr_matrix_logvar, cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, 1].set_title("LogVar Dimension Correlations")
        axes[1, 1].set_xlabel("Latent Dimension")
        axes[1, 1].set_ylabel("Latent Dimension")
        plt.colorbar(im2, ax=axes[1, 1])
        # Distribution of each dimension for mu
        for i in range(min(6, latent_dim)):
            axes[2, 0].hist(latent_mus[:, i], histtype="step", bins=30, alpha=0.7, label=f"Dim {i+1}")
        axes[2, 0].set_xlabel("Mu Value")
        axes[2, 0].set_ylabel("Frequency")
        axes[2, 0].set_title("Mu Marginal Distributions (First 6 dims)")
        axes[2, 0].legend()
        # Distribution of each dimension for logvar
        for i in range(min(6, latent_dim)):
            axes[2, 1].hist(latent_logvars[:, i], histtype="step", bins=30, alpha=0.7, label=f"Dim {i+1}")
        axes[2, 1].set_xlabel("LogVar Value")
        axes[2, 1].set_ylabel("Frequency")
        axes[2, 1].set_title("LogVar Marginal Distributions (First 6 dims)")
        axes[2, 1].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    plt.show()
    # Print summary statistics
    print(f"\nLatent Mu Statistics (using {latent_mus.shape[0]} samples):")
    print(f"Mean: {np.mean(latent_mus, axis=0)}")
    print(f"Std:  {np.std(latent_mus, axis=0)}")
    print(f"Min:  {np.min(latent_mus, axis=0)}")
    print(f"Max:  {np.max(latent_mus, axis=0)}")
    print(f"\nLatent LogVar Statistics (using {latent_logvars.shape[0]} samples):")
    print(f"Mean: {np.mean(latent_logvars, axis=0)}")
    print(f"Std:  {np.std(latent_logvars, axis=0)}")
    print(f"Min:  {np.min(latent_logvars, axis=0)}")
    print(f"Max:  {np.max(latent_logvars, axis=0)}")

    # Save latent_mus and latent_logvars to the same CSV file
    latent_df = pd.DataFrame(np.hstack([latent_mus, latent_logvars]), columns=[f"mu_{i}" for i in range(latent_mus.shape[1])] + [f"logvar_{i}" for i in range(latent_logvars.shape[1])])
    latent_csv_path = os.path.join(folder, "latent_mus_logvars.csv")
    latent_df.to_csv(latent_csv_path, index=False)
    print(f"Saved latent_mus and latent_logvars to {latent_csv_path}")

    return latent_mus, latent_logvars


def show_random_reconstructions(
    folder: str,
    model_path: str | None = None,
    model: VAE | None = None,
    latent_dim: int = 6,
    num_samples: int = 4,
    device: str | torch.device | None = None,
    cmap: str = "rainbow",
):
    # -------------------------------------------------------------
    # 1. Resolve device
    # -------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load normalization statistics
    vol_stats_path = f"{folder}/vol_normalization_stats.npz"
    if os.path.exists(vol_stats_path):
        vol_stats = np.load(vol_stats_path)
        vol_mean = vol_stats["mean"]
        vol_std = vol_stats["std"]
    else:
        print("Reconstructions will be shown in normalized space")
        vol_mean = None
        vol_std = None

    # -------------------------------------------------------------
    # 2. Load / validate the model
    # -------------------------------------------------------------
    if model is None:
        assert model_path is not None, "Provide `model` or `model_path`."
        # Determine input shape
        loader, _ = create_dataloader(folder, "post_vol_", "train", batch_size=1, shuffle=True, compute_stats=False)
        first_batch, _, _ = next(iter(loader))
        input_shape = first_batch.shape[2:]

        model = VAE(latent_dim=latent_dim, input_shape=input_shape).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = model.to(device)
        # Create a loader for getting samples
        loader, _ = create_dataloader(folder, "post_vol_", "train", batch_size=1, shuffle=True, compute_stats=False)

    model.eval()

    # -------------------------------------------------------------
    # 3. Generate reconstructions
    # -------------------------------------------------------------
    figs = []
    with torch.no_grad():
        for _ in range(num_samples):
            # grab one random surface
            x, k_grid, T_grid = next(iter(loader))
            x = x.to(device)
            # Use the VAE's Monte Carlo sampling (100 samples averaged)
            recon, mu, _ = model(x)

            # Convert to numpy and denormalize if possible
            x_np = x.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
            recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
            mu_np = mu.squeeze(0).cpu().numpy()  # (latent_dim,)

            # Denormalize for visualization if stats are available
            if vol_mean is not None and vol_std is not None:
                x_denorm = x_np * vol_std + vol_mean
                recon_denorm = recon_np * vol_std + vol_mean
                figs.append((x_denorm, recon_denorm, mu_np, x_np, recon_np))
            else:
                figs.append((x_np, recon_np, mu_np, x_np, recon_np))

    # ─────────────────────────────── plotting ────────────────────────────────
    n_rows = num_samples
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    for idx, fig_data in enumerate(figs):
        if len(fig_data) == 5:  # denormalized data available
            x_denorm, recon_denorm, mu_np, x_norm, recon_norm = fig_data
            x_display, recon_display = x_denorm, recon_denorm
            title_suffix = " (denormalized)"
        else:  # only normalized data
            x_display, recon_display, mu_np, _, _ = fig_data
            title_suffix = " (normalized)"

        # Prepare meshgrid for pcolormesh
        T_mesh, k_mesh = np.meshgrid(T_grid.numpy(), k_grid.numpy())
        print("K_mesh.shape, T_mesh.shape, x_display.shape")
        print(k_mesh.shape, T_mesh.shape, x_display.shape)

        # input
        ax_in = axes[idx, 0]
        pcm0 = ax_in.pcolormesh(k_mesh, T_mesh, x_display, shading="auto", cmap=cmap)
        ax_in.set_title(f"Input #{idx}{title_suffix}")
        ax_in.set_ylabel("k")
        ax_in.set_xlabel("T")
        fig.colorbar(pcm0, ax=ax_in, fraction=0.046, pad=0.04)

        # reconstruction
        ax_out = axes[idx, 1]
        pcm1 = ax_out.pcolormesh(k_mesh, T_mesh, recon_display, shading="auto", cmap=cmap)
        mu_str = ", ".join([f"{v:+.3f}" for v in mu_np])
        ax_out.set_title(f"Recon #{idx}{title_suffix}\nμ = [{mu_str}]")
        ax_out.set_ylabel("k")
        ax_out.set_xlabel("T")
        fig.colorbar(pcm1, ax=ax_out, fraction=0.046, pad=0.04)

        # difference
        ax_diff = axes[idx, 2]
        diff_display = x_display - recon_display
        pcm2 = ax_diff.pcolormesh(k_mesh, T_mesh, diff_display, shading="auto", cmap="RdBu_r")
        mse = np.mean(diff_display**2)
        ax_diff.set_title(f"Difference #{idx}{title_suffix}\nMSE = {mse:.6f}")
        ax_diff.set_ylabel("k")
        ax_diff.set_xlabel("T")
        fig.colorbar(pcm2, ax=ax_diff, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{folder}/vae_reconstructions.png", dpi=300)
    plt.show()



def show_quote_date_reconstructions(
    folder: str,
    quote_dates: list,
    model_path: str | None = None,
    model: VAE | None = None,
    latent_dim: int = 6,
    device: str | torch.device | None = None,
    cmap: str = "rainbow",
    data_type: str = "train",
):
    """
    Show reconstructions for specific quote dates.

    Args:
        folder: Path to folder containing data
        quote_dates: List of quote dates to visualize (should be 4 dates)
        model_path: Path to saved model state dict
        model: Pre-loaded VAE model (alternative to model_path)
        latent_dim: Dimensionality of the latent space
        device: Device to run on
        cmap: Colormap for visualization
        data_type: Type of data to load ("train" or "test")
    """
    # Validate input
    if len(quote_dates) != 3:
        raise ValueError(f"Expected exactly 3 quote dates, got {len(quote_dates)}")

    # -------------------------------------------------------------
    # 1. Resolve device
    # -------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load normalization statistics
    vol_stats_path = f"{folder}/vol_normalization_stats.npz"
    if os.path.exists(vol_stats_path):
        vol_stats = np.load(vol_stats_path)
        vol_mean = vol_stats["mean"]
        vol_std = vol_stats["std"]
        print(f"Loaded normalization stats: mean={vol_mean:.6f}, std={vol_std:.6f}")
    else:
        print("Warning: No normalization stats found. Reconstructions will be shown in normalized space")
        vol_mean = None
        vol_std = None

    # -------------------------------------------------------------
    # 2. Load dataset to find quote dates
    # -------------------------------------------------------------
    _, dataset = create_dataloader(folder, "post_vol_", data_type, batch_size=1, shuffle=False, compute_stats=False)

    # Find indices for the requested quote dates
    quote_date_indices = []
    available_dates = dataset.quote_dates

    print(f"Looking for quote dates: {quote_dates}")
    print(f"Available date range: {available_dates[0]} to {available_dates[-1]}")

    for target_date in quote_dates:
        # Find the closest date or exact match
        if target_date in available_dates:
            idx = np.where(available_dates == target_date)[0][0]
            quote_date_indices.append(idx)
            print(f"Found exact match for {target_date} at index {idx}")
        else:
            # Find closest date
            date_diffs = np.abs(available_dates.astype('datetime64[D]') - np.datetime64(target_date))
            closest_idx = np.argmin(date_diffs)
            closest_date = available_dates[closest_idx]
            quote_date_indices.append(closest_idx)
            print(f"No exact match for {target_date}, using closest date {closest_date} at index {closest_idx}")

    # -------------------------------------------------------------
    # 3. Load / validate the model
    # -------------------------------------------------------------
    if model is None:
        assert model_path is not None, "Provide `model` or `model_path`."
        # Determine input shape from dataset
        sample_surface, _, _ = dataset[0]
        input_shape = sample_surface.shape[1:]  # Remove channel dimension

        model = VAE(latent_dim=latent_dim, input_shape=input_shape).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        model = model.to(device)

    model.eval()

    # -------------------------------------------------------------
    # 4. Generate reconstructions for specific dates
    # -------------------------------------------------------------
    figs = []
    actual_dates_used = []

    with torch.no_grad():
        for i, date_idx in enumerate(quote_date_indices):
            # Get specific surface by index
            surface, k_grid, T_grid = dataset[date_idx]
            actual_date = available_dates[date_idx]
            actual_dates_used.append(actual_date)

            # Prepare for model input
            x = surface.unsqueeze(0).to(device)  # Add batch dimension

            # Get reconstruction
            recon, mu, _ = model(x)

            # Convert to numpy
            x_np = x.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
            recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
            mu_np = mu.squeeze(0).cpu().numpy()  # (latent_dim,)

            # Denormalize for visualization if stats are available
            if vol_mean is not None and vol_std is not None:
                x_denorm = x_np * vol_std + vol_mean
                recon_denorm = recon_np * vol_std + vol_mean
                figs.append((x_denorm, recon_denorm, mu_np, actual_date, k_grid, T_grid))
            else:
                figs.append((x_np, recon_np, mu_np, actual_date, k_grid, T_grid))

    # ─────────────────────────────── plotting ────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    title_suffix = " (denormalized)" if vol_mean is not None else " (normalized)"

    for idx, fig_data in enumerate(figs):
        x_display, recon_display, mu_np, actual_date, k_grid, T_grid = fig_data

        # Prepare meshgrid for pcolormesh
        T_mesh, k_mesh = np.meshgrid(T_grid.numpy(), k_grid.numpy())

        # Input surface
        ax_in = axes[idx, 0]
        pcm0 = ax_in.pcolormesh(k_mesh, T_mesh, x_display, shading="auto", cmap=cmap, vmin=0.09, vmax=0.66)
        ax_in.set_title(f"Input: {actual_date}{title_suffix}")
        ax_in.set_ylabel("k (log moneyness)")
        ax_in.set_xlabel("T (time to expiry)")
        fig.colorbar(pcm0, ax=ax_in, fraction=0.046, pad=0.04)

        # Reconstruction
        ax_out = axes[idx, 1]
        pcm1 = ax_out.pcolormesh(k_mesh, T_mesh, recon_display, shading="auto", cmap=cmap, vmin=0.09, vmax=0.66)
        mu_str = ", ".join([f"{v:+.3f}" for v in mu_np[:3]])  # Show first 3 dimensions
        if len(mu_np) > 3:
            mu_str += "..."
        ax_out.set_title(f"Reconstruction: {actual_date}{title_suffix}\nμ ≈ [{mu_str}]")
        ax_out.set_ylabel("k (log moneyness)")
        ax_out.set_xlabel("T (time to expiry)")
        fig.colorbar(pcm1, ax=ax_out, fraction=0.046, pad=0.04)

        # Difference
        ax_diff = axes[idx, 2]
        diff_display = x_display - recon_display
        pcm2 = ax_diff.pcolormesh(k_mesh, T_mesh, diff_display, shading="auto", cmap="RdBu_r")
        mse = np.mean(diff_display**2)
        mae = np.mean(np.abs(diff_display))
        ax_diff.set_title(f"Difference: {actual_date}\nMSE = {mse:.6f}, MAE = {mae:.6f}")
        ax_diff.set_ylabel("k (log moneyness)")
        ax_diff.set_xlabel("T (time to expiry)")
        fig.colorbar(pcm2, ax=ax_diff, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{folder}/vae_quote_date_reconstructions.png", dpi=300, bbox_inches="tight")
    print(f"Saved reconstruction plot to {folder}/vae_quote_date_reconstructions.png")
    plt.show()

    # Print summary
    print(f"\nReconstruction Summary for {data_type} data:")
    for i, (requested, actual) in enumerate(zip(quote_dates, actual_dates_used)):
        if str(requested) == str(actual):
            print(f"  {i+1}. {requested} ✓ (exact match)")
        else:
            print(f"  {i+1}. {requested} → {actual} (closest available)")

    # Save quote_dates, vol surfaces, and reconstructed vol surfaces to npz
    save_dict = {
        "quote_dates": np.array([str(d) for d in actual_dates_used]),
        "k_grid": k_grid.numpy(),
        "T_grid": T_grid.numpy(),
        "vol_surfaces": np.stack([fig_data[0] for fig_data in figs]),  # input surfaces (denormalized or normalized)
        "recon_vol_surfaces": np.stack([fig_data[1] for fig_data in figs]),  # reconstructed surfaces
    }
    np.savez(os.path.join(folder, "vae_quote_date_reconstructions.npz"), **save_dict)
    print(f"Saved quote_dates, vol_surfaces, and recon_vol_surfaces to {folder}/vae_quote_date_reconstructions.npz")

    return figs


def plot_predict_prices_from_vol_surface_and_params(
    folder: str,
    product_type: str,
    pricer_model_path: str,
    include_train: bool = True,
    latent_dim: int = 8,
    pricing_param_dim: int = 2,
    vol_input_shape: tuple = (41, 20),
    batch_size: int = 32,
    device: str | torch.device | None = None,
) -> dict:
    """
    Predict prices using a trained Pricer model and plot predicted vs ground truth prices for both train and test data.

    Args:
        folder: Path to folder containing pricing data and normalization stats
        product_type: Type of product (e.g., "AmericanPut")
        pricer_model_path: Path to trained pricer model state dict
        include_train: Whether to include training data comparison (default: True)
        latent_dim: Dimensionality of the latent space
        pricing_param_dim: Dimensionality of pricing parameters
        vol_input_shape: Shape of volatility surface input
        batch_size: Batch size for prediction
        device: Device to run predictions on

    Returns:
        results: Dictionary containing results for both train and test sets
    """
    # Resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load trained pricer model
    model = Pricer(latent_dim=latent_dim, pricing_param_dim=pricing_param_dim, vol_input_shape=vol_input_shape)
    model.load_state_dict(torch.load(pricer_model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded trained pricer model from {pricer_model_path}")

    # Load pricing parameter normalization statistics for denormalization
    param_stats_path = f"{folder}/{product_type}_pricing_param_stats.npz"
    if os.path.exists(param_stats_path):
        stats = np.load(param_stats_path)
        price_params_mean = stats["params_mean"]
        price_params_std = stats["params_std"]
        price_mean = stats["price_mean"]
        price_std = stats["price_std"]
        print(f"Loaded pricing parameter stats: mean={price_params_mean}, std={price_params_std}")
        print(f"Loaded price stats: mean={price_mean:.6f}, std={price_std:.6f}")
    else:
        raise FileNotFoundError(f"Pricing parameter stats not found at {param_stats_path}")

    def evaluate_dataset(data_type: str):
        """Helper function to evaluate a single dataset"""
        print(f"\nLoading {product_type} {data_type} data for price prediction...")

        # Load data
        loader, dataset = create_pricing_dataloader(folder, product_type, data_type, batch_size=batch_size, shuffle=False, compute_stats=False)

        # Collect predictions
        predicted_prices_list = []
        target_prices_list = []
        vol_surfaces_list = []
        pricing_params_norm_list = []

        print(f"Running predictions on {len(dataset)} {data_type} samples...")

        with torch.no_grad():
            for batch_idx, (vol_surface, pricing_param, target_price) in enumerate(loader):
                vol_surface = vol_surface.to(device)
                pricing_param = pricing_param.to(device)
                target_price = target_price.to(device)

                # Forward pass through pricer
                predicted_price = model(vol_surface, pricing_param)

                # Collect results
                predicted_prices_list.append(predicted_price.cpu().numpy())
                target_prices_list.append(target_price.cpu().numpy())
                vol_surfaces_list.append(vol_surface.cpu().numpy())
                pricing_params_norm_list.append(pricing_param.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(loader)} batches")

        # Concatenate all results
        predicted_prices = np.concatenate(predicted_prices_list, axis=0)
        target_prices = np.concatenate(target_prices_list, axis=0)
        vol_surfaces = np.concatenate(vol_surfaces_list, axis=0)
        pricing_params_norm = np.concatenate(pricing_params_norm_list, axis=0)

        # Denormalize pricing parameters for interpretation
        pricing_params_denorm = pricing_params_norm * price_params_std + price_params_mean

        # Denormalize prices for meaningful interpretation
        predicted_prices_denorm = predicted_prices * price_std + price_mean
        target_prices_denorm = target_prices * price_std + price_mean

        # Calculate prediction metrics on denormalized prices
        mse = np.mean((predicted_prices_denorm - target_prices_denorm) ** 2)
        mae = np.mean(np.abs(predicted_prices_denorm - target_prices_denorm))
        rmse = np.sqrt(mse)

        # Calculate R² score on denormalized prices
        ss_res = np.sum((target_prices_denorm - predicted_prices_denorm) ** 2)
        ss_tot = np.sum((target_prices_denorm - np.mean(target_prices_denorm)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        print(f"\n{data_type.title()} Set Prediction Results (denormalized prices):")
        print(f"Number of predictions: {len(predicted_prices_denorm)}")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R² Score: {r2_score:.6f}")
        print(f"Target price range: [{np.min(target_prices_denorm):.3f}, {np.max(target_prices_denorm):.3f}]")
        print(f"Predicted price range: [{np.min(predicted_prices_denorm):.3f}, {np.max(predicted_prices_denorm):.3f}]")

        return {
            'predicted_prices': predicted_prices_denorm,
            'target_prices': target_prices_denorm,
            'pricing_params': pricing_params_denorm,
            'vol_surfaces': vol_surfaces,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2_score
        }

    # Evaluate datasets
    results = {}
    data_types = ['test']
    if include_train:
        data_types = ['train', 'test']

    for data_type in data_types:
        results[data_type] = evaluate_dataset(data_type)

    # Create comprehensive plots
    if include_train:
        # Both train and test comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for i, data_type in enumerate(['train', 'test']):
            result = results[data_type]
            predicted_flat = result['predicted_prices'].flatten()
            target_flat = result['target_prices'].flatten()

            # Scatter plot
            ax_scatter = axes[i, 0]
            ax_scatter.scatter(target_flat, predicted_flat, alpha=0.6, s=20,
                               color="blue" if data_type == "train" else "orange")

            # Perfect prediction line (y=x)
            min_price = min(np.min(target_flat), np.min(predicted_flat))
            max_price = max(np.max(target_flat), np.max(predicted_flat))
            ax_scatter.plot([min_price, max_price], [min_price, max_price], "r--", linewidth=2, label="Perfect Prediction")

            ax_scatter.set_xlabel("Ground Truth Price")
            ax_scatter.set_ylabel("Predicted Price")
            ax_scatter.set_title(f"{product_type} Price Prediction\n{data_type.title()} Set (R² = {result['r2_score']:.4f})")
            ax_scatter.legend()
            ax_scatter.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f"MSE: {result['mse']:.4f}\nMAE: {result['mae']:.4f}\nRMSE: {result['rmse']:.4f}\nSamples: {len(predicted_flat)}"
            ax_scatter.text(0.05, 0.95, stats_text, transform=ax_scatter.transAxes, fontsize=10,
                            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

            # Residual plot
            ax_residual = axes[i, 1]
            residuals = predicted_flat - target_flat
            ax_residual.scatter(target_flat, residuals, alpha=0.6, s=20,
                                color="green" if data_type == "train" else "red")
            ax_residual.axhline(y=0, color="r", linestyle="--", linewidth=2, label="Zero Error")
            ax_residual.set_xlabel("Ground Truth Price")
            ax_residual.set_ylabel("Prediction Error (Predicted - Truth)")
            ax_residual.set_title(f"{data_type.title()} Set Residuals")
            ax_residual.legend()
            ax_residual.grid(True, alpha=0.3)

            # Add residual statistics
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            residual_text = f"Mean Error: {residual_mean:.4f}\nStd Error: {residual_std:.4f}"
            ax_residual.text(0.05, 0.95, residual_text, transform=ax_residual.transAxes, fontsize=10,
                             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{folder}/pricer_prediction_comparison_train_test.png", dpi=300)

    else:
        # Test only comparison (original format)
        result = results['test']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        predicted_flat = result['predicted_prices'].flatten()
        target_flat = result['target_prices'].flatten()

        # Main scatter plot
        ax1.scatter(target_flat, predicted_flat, alpha=0.6, s=20, color="blue")

        # Perfect prediction line (y=x)
        min_price = min(np.min(target_flat), np.min(predicted_flat))
        max_price = max(np.max(target_flat), np.max(predicted_flat))
        ax1.plot([min_price, max_price], [min_price, max_price], "r--", linewidth=2, label="Perfect Prediction")

        ax1.set_xlabel("Ground Truth Price")
        ax1.set_ylabel("Predicted Price")
        ax1.set_title(f"{product_type} Price Prediction\nTest Set (R² = {result['r2_score']:.4f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"MSE: {result['mse']:.4f}\nMAE: {result['mae']:.4f}\nRMSE: {result['rmse']:.4f}\nSamples: {len(predicted_flat)}"
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # Residual plot
        residuals = predicted_flat - target_flat
        ax2.scatter(target_flat, residuals, alpha=0.6, s=20, color="green")
        ax2.axhline(y=0, color="r", linestyle="--", linewidth=2, label="Zero Error")
        ax2.set_xlabel("Ground Truth Price")
        ax2.set_ylabel("Prediction Error (Predicted - Truth)")
        ax2.set_title("Residual Plot")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add residual statistics
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        residual_text = f"Mean Error: {residual_mean:.4f}\nStd Error: {residual_std:.4f}"
        ax2.text(0.05, 0.95, residual_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{folder}/pricer_prediction_comparison.png", dpi=300)

    plt.show()
    plt.close()

    # Print comparison summary if both datasets evaluated
    if include_train and len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY: Train vs Test Performance")
        print(f"{'='*60}")
        print(f"{'Metric':<15} {'Train':<15} {'Test':<15} {'Difference':<15}")
        print(f"{'-'*60}")

        train_result = results['train']
        test_result = results['test']

        metrics = ['mse', 'mae', 'rmse', 'r2_score']
        metric_names = ['MSE', 'MAE', 'RMSE', 'R² Score']

        for metric, name in zip(metrics, metric_names):
            train_val = train_result[metric]
            test_val = test_result[metric]
            diff = test_val - train_val
            print(f"{name:<15} {train_val:<15.6f} {test_val:<15.6f} {diff:<+15.6f}")

        print(f"{'-'*60}")

        # Check for potential overfitting
        if test_result['r2_score'] < train_result['r2_score'] - 0.1:
            print("⚠️  Potential overfitting detected: Test R² significantly lower than Train R²")
        elif abs(test_result['r2_score'] - train_result['r2_score']) < 0.05:
            print("✅ Good generalization: Train and Test R² scores are similar")
        else:
            print("ℹ️  Model shows reasonable generalization performance")

    # Save price prediction results to npz file
    save_dict = {}
    for data_type in results.keys():
        result = results[data_type]
        save_dict[f'{data_type}_price'] = result['target_prices'].flatten()
        save_dict[f'predicted_{data_type}_price'] = result['predicted_prices'].flatten()
        save_dict[f'{data_type}_pricing_params'] = result['pricing_params']
        save_dict[f'{data_type}_metrics'] = np.array([
            result['mse'], result['mae'], result['rmse'], result['r2_score']
        ])

    # Save to npz file
    save_path = os.path.join(folder, f"{product_type}_price_predictions.npz")
    np.savez(save_path, **save_dict)
    print(f"\n💾 Saved price prediction results to {save_path}")
    print(f"   Contains: {list(save_dict.keys())}")

    # Print summary of saved data
    for data_type in results.keys():
        result = results[data_type]
        n_samples = len(result['target_prices'])
        price_range = [np.min(result['target_prices']), np.max(result['target_prices'])]
        print(f"   📊 {data_type.title()} set: {n_samples} samples, price range [{price_range[0]:.3f}, {price_range[1]:.3f}]")

    return results
