# VAE-Based Option Pricing Framework

A deep learning framework that combines Variational Autoencoders (VAE) with option pricing models to predict option prices from volatility surfaces. The system uses a two-stage approach: first learning compressed representations of volatility surfaces via VAE, then training a pricer model to predict option prices from these latent representations and pricing parameters.

## 🎯 Project Overview

This project implements a novel approach to option pricing by:
1. **Volatility Surface Compression**: Using VAE to learn low-dimensional representations of SPX volatility surfaces
2. **Price Prediction**: Training a neural network pricer that maps from compressed volatility surfaces and option parameters (Strike, Time-to-expiry) to option prices
3. **Multi-Asset Support**: Framework supports American Put options, Asian options, and can be extended to other exotic options

## 🏗️ Architecture

```
Volatility Surface (41×20) → VAE Encoder → Latent Code (z) → Pricer → Option Price
                                                    ↑
                                          Pricing Parameters (K, T)
```

- **VAE**: Compresses volatility surfaces from (41×20) to latent dimension (typically 6-8)
- **Pricer**: Neural network that takes latent codes + pricing parameters and outputs option prices
- **Training Strategy**: Two-stage training with encoder freezing followed by fine-tuning

## 📁 Project Structure

```
VAE_Pricing/
├── analyze/                    # Main ML models and training scripts
│   ├── VAE_model.py           # Core VAE and Pricer model definitions
│   ├── main_VAE.py            # VAE training pipeline
│   ├── ML_analyze.py          # Analysis utilities
│   └── main_ML_analyze.py     # Main analysis script
├── data_process/              # Data preprocessing pipeline
│   ├── main_process.py        # Main data processing script
│   ├── process_raw_data.py    # Raw data processing utilities
│   ├── vol_to_grid.py         # Volatility surface grid conversion
│   ├── AH_vol_to_grid.py      # Advanced volatility processing
│   └── data_pack/             # Processed data storage
├── pricing/                   # Option pricing implementations
│   ├── american_put_pricer.py # American Put option pricing
│   ├── asian_option_pricer.py # Asian option pricing
│   └── main_price.py          # Main pricing script
├── optionsdx_data/           # Raw SPX options data (2010-2023)
├── option_data.ipynb         # Data exploration notebook
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision numpy pandas matplotlib seaborn
pip install scikit-learn scipy
pip install QuantLib  # For option pricing
pip install jupyter  # For notebooks
```

### 1. Data Processing

Process raw SPX options data into volatility surfaces:

```bash
cd data_process
python main_process.py
```

This will:
- Read raw options data from `optionsdx_data/`
- Convert to volatility surfaces on standardized grids
- Split into train/test sets
- Save processed data to `data_pack/`

### 2. Train VAE

Train the volatility surface VAE:

```bash
cd analyze
python main_VAE.py
```

Configuration options:
- `latent_dim`: Latent space dimensionality (default: 6)
- `num_epochs`: Training epochs (default: 100)
- `batch_size`: Batch size (default: 32)

### 3. Generate Pricing Data

Generate option price datasets using QuantLib:

```bash
cd pricing
python main_price.py
```

This creates training data for the pricer by:
- Sampling (K, T) parameter combinations
- Computing option prices using QuantLib
- Pairing with corresponding volatility surfaces

### 4. Train Pricer

Train the option pricer model:

```bash
cd analyze
# Edit main_ML_analyze.py to specify:
# - VAE model path
# - Product type (AmericanPut, AsianCall, etc.)
# - Training parameters
python main_ML_analyze.py
```

### 5. Evaluate Results

The system automatically generates:
- Loss curves for both VAE and Pricer
- Prediction vs ground truth scatter plots
- Latent space visualizations
- Reconstruction quality assessments

## 🔧 Key Components

### VAE Model (`analyze/VAE_model.py`)

**Architecture:**
- **Encoder**: Conv2D layers (41,20) → latent space (typically 6-8 dims)
- **Decoder**: Deconv layers latent → (41,20) reconstruction
- **Loss**: MSE reconstruction loss

**Key Features:**
- Batch normalization for stable training
- Cosine annealing learning rate scheduling
- Comprehensive normalization pipeline
- Monte Carlo sampling for reconstruction

### Pricer Model (`analyze/VAE_model.py`)

**Architecture:**
- **VAE Encoder**: Pre-trained, initially frozen
- **Parameter Network**: Processes (K, T) pricing parameters
- **Fusion Network**: Combines latent codes with parameters
- **Output**: Single option price prediction

**Training Strategy:**
1. **Stage 1**: Freeze VAE encoder, train only pricer layers
2. **Stage 2**: Unfreeze encoder for end-to-end fine-tuning

### Data Pipeline

**Normalization Strategy:**
- All normalization statistics computed from **training data only**
- Same statistics applied to both train/test sets (no data leakage)
- Z-score normalization: `(x - μ) / σ`
- Proper denormalization for final predictions

## 📊 Model Performance

### VAE Results
- **Reconstruction Quality**: Low MSE on volatility surfaces
- **Latent Space**: Meaningful compression of volatility dynamics
- **Generalization**: Stable performance on unseen dates

### Pricer Results
- **Accuracy**: R² > 0.95 on test data for American Put options
- **Speed**: 1000x faster than QuantLib after training
- **Robustness**: Handles out-of-sample volatility regimes

## 🎛️ Configuration

### VAE Training
```python
# In main_VAE.py
latent_dim = 6          # Latent space dimension
batch_size = 32         # Training batch size
num_epochs = 100        # Training epochs
lr = 1e-3              # Learning rate
weight_decay = 1e-4    # L2 regularization
```

### Pricer Training
```python
# In main_ML_analyze.py
num_epochs = 100           # Initial training epochs
num_epochs_fine_tune = 50  # Fine-tuning epochs
lr = 1e-3                 # Learning rate
freeze_encoder = True     # Two-stage training
```

## 📈 Data Requirements

### Input Data Format
- **Volatility Surfaces**: (41, 20) grids
  - K-dimension: 41 log-moneyness points [-0.3, 0.3]
  - T-dimension: 20 time points [0.05, 1.0] years
- **Options Data**: OptionsDX format with fields:
  - Quote date, Strike, Expiry, IV, Underlying price
- **Pricing Parameters**: (K, T) combinations for training

### Data Splits
- **Training**: ~80% of available dates
- **Testing**: ~20% of available dates
- **Temporal Split**: Ensures no data leakage across time

## 🔬 Research Features

### Diagnostic Tools
- **Normalization Diagnostics**: Verify data preprocessing integrity
- **Latent Space Analysis**: PCA, correlation analysis of learned representations
- **Prediction Analysis**: Residual plots, bias detection
- **Reconstruction Visualization**: Input vs reconstructed surfaces

### Extensibility
- **New Option Types**: Add pricing functions in `pricing/`
- **Model Architectures**: Modify VAE/Pricer in `analyze/VAE_model.py`
- **Data Sources**: Extend processing pipeline in `data_process/`

## 🐛 Troubleshooting

### Common Issues

**"wrong argument type" in QuantLib:**
```python
# Ensure all inputs are native Python floats
K = float(K_value)
T = float(T_value)
vol_matrix[i][j] = float(vol_surface[i][j])
```

**Systematic bias in predictions:**
- Check normalization statistics consistency
- Verify train/test use same normalization
- Run diagnostic: `diagnose_normalization_issues()`

**VAE training instability:**
- Reduce learning rate
- Increase batch size
- Check data normalization quality

## 📚 References

### Key Papers
- Kingma & Welling (2014): Auto-Encoding Variational Bayes
- Hernandez (2017): Model Calibration with Neural Networks

### QuantLib Documentation
- [QuantLib Python Cookbook](https://quantlib-python-docs.readthedocs.io/)
- [Asian Option Pricing](https://www.quantlib.org/reference/group__asianOption.html)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **QuantLib**: Comprehensive quantitative finance library
- **PyTorch**: Deep learning framework
- **OptionsDX**: SPX options data provider

---

**Project Status**: Active Development
**Last Updated**: August 2025
**Python Version**: 3.8+

For questions or issues, please open a GitHub issue or contact the maintainers.
