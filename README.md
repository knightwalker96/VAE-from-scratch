# Variational AutoEncoder (VAE) on MNIST

This repository implements a Variational AutoEncoder (VAE) from scratch using PyTorch, trained on the MNIST dataset. The VAE learns to reconstruct handwritten digits and generate new digit-like images.

## Features
- Simple VAE with fully connected encoder and decoder.
- Trained with Binary Cross-Entropy (BCE) loss and KL divergence.
- Saves original vs. reconstructed comparison images during training.
- Generates individual and grid images for each digit (0–9) during inference.
- Configurable hyperparameters via command-line arguments.
- Saves trained model for reuse.

## Prerequisites
- Python 3.8 or higher
- PyTorch
- torchvision
- tqdm

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/knightwalker96/vae-from-scratch.git
   cd vae-from-scratch
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Usage: Train the VAE and generate samples
   ```bash
   python train.py --epochs 20 --batch-size 32 --lr 3e-4 --output-dir outputs
   ```
   or
   
   ```bash
   python train.py
