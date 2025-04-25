"""Train and evaluate a Variational AutoEncoder (VAE) on the MNIST dataset."""

import argparse
import os
import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from model import VariationalAutoEncoder

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a VAE on MNIST dataset.")
    parser.add_argument('--input-dim', type=int, default=784, help='Input dimension')
    parser.add_argument('--hidden-dim', type=int, default=200, help='Hidden layer dimension')
    parser.add_argument('--output-dim', type=int, default=20, help='Latent space dimension')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--dataset-dir', type=str, default='dataset', help='Directory for MNIST dataset')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory for outputs')
    parser.add_argument('--save-model', type=str, default='vae_model.pth', help='Path to save trained model')
    parser.add_argument('--num-examples', type=int, default=3, help='Number of generated examples per digit')
    return parser.parse_args()

def get_digit_examples(dataset, num_classes=10):
    """Collect one example per digit from the dataset."""
    images = []
    found = set()
    for x, y in DataLoader(dataset, batch_size=1, shuffle=False):
        if y.item() not in found:
            images.append(x)
            found.add(y.item())
        if len(found) == num_classes:
            break
    if len(images) != num_classes:
        raise ValueError(f"Found only {len(images)} digits instead of {num_classes}.")
    return images

def inference(digit, num_examples, model, test_dataset, device, output_dir):
    """Generate samples for a specific digit and save individual and grid images."""
    images = get_digit_examples(test_dataset)
    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            input_img = images[d].view(1, 784).to(device)
            mu, sigma = model.encode(input_img)
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    outs = []
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma).to(device)
        z = mu + sigma * epsilon
        with torch.no_grad():
            out = model.decode(z)
            out = out.view(1, 1, 28, 28).squeeze(0)  # Shape: [1, 28, 28]
        outs.append(out)

    if outs:
        grid = make_grid(outs, nrow=num_examples)
        save_image(grid, os.path.join(output_dir, f"Generated_{digit}_grid.png"))

def main():
    args = parse_args()

    # Create directories
    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = datasets.MNIST(
        root=args.dataset_dir,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = datasets.MNIST(
        root=args.dataset_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    # Initialize model, optimizer, and loss
    model = VariationalAutoEncoder(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss(reduction="sum")

    # Training loop
    for epoch in range(args.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for i, (x, _) in loop:
            x = x.to(device).view(x.shape[0], args.input_dim)
            x_reconstructed, mu, sigma = model(x)

            # Compute loss
            recon_loss = loss_fn(x_reconstructed, x)
            kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = recon_loss + kl_div

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

            # Save comparison images for first batch
            if i == 0:
                batch_size = x.shape[0]
                x_reshaped = x.view(batch_size, 1, 28, 28)
                x_recon_reshaped = x_reconstructed.view(batch_size, 1, 28, 28)
                comparison = torch.cat([x_reshaped, x_recon_reshaped])
                save_image(comparison, os.path.join(args.output_dir, f"epoch_{epoch}_comparison.png"), nrow=batch_size)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, args.save_model))

    # Inference
    model.eval()
    for digit in range(10):
        inference(digit, args.num_examples, model, test_dataset, device, args.output_dir)

if __name__ == "__main__":
    main()
