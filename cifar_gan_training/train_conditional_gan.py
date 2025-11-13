#!/usr/bin/env python3
"""
Conditional GAN Training on CIFAR-10
Optimized for 4x RTX 4090 GPUs
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Available GPUs: {torch.cuda.device_count()}")

# Hyperparameters
batch_size = 128  # Large batch for 4x RTX 4090s
learning_rate = 0.0002
beta1 = 0.5
num_epochs = 50
latent_dim = 100
num_classes = 10  # CIFAR-10 classes

# CIFAR-10 classes for reference
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

class Generator(nn.Module):
    """Conditional Generator for CIFAR-10"""
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, 50)

        # Generator layers
        self.main = nn.Sequential(
            # Input: latent_dim + 50 (embedded label)
            nn.ConvTranspose2d(latent_dim + 50, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 4x4x512

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8x8x256

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16x128

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32x64

            nn.ConvTranspose2d(64, 3, 1, 1, 0, bias=False),
            nn.Tanh()
            # 32x32x3 (CIFAR-10 resolution)
        )

    def forward(self, noise, labels):
        # Embed labels and reshape
        label_embed = self.label_embedding(labels).view(labels.size(0), 50, 1, 1)

        # Concatenate noise and embedded labels
        gen_input = torch.cat([noise, label_embed], 1)

        return self.main(gen_input)

class Discriminator(nn.Module):
    """Conditional Discriminator for CIFAR-10"""
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, 50)

        # Discriminator layers
        self.main = nn.Sequential(
            # Input: 3 + 50 channels (image + embedded label)
            nn.Conv2d(3 + 50, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16x64

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8x128

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4x256

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 2x2x512

            nn.Conv2d(512, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
            # 1x1x1
        )

    def forward(self, img, labels):
        # Embed labels and reshape to match image dimensions
        label_embed = self.label_embedding(labels).view(labels.size(0), 50, 1, 1)
        label_embed = label_embed.expand(labels.size(0), 50, img.size(2), img.size(3))

        # Concatenate image and embedded labels
        disc_input = torch.cat([img, label_embed], 1)

        return self.main(disc_input).view(-1, 1).squeeze(1)

def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    print("Setting up CIFAR-10 Conditional GAN Training...")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Download and prepare CIFAR-10
    print("Downloading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    print(f"Dataset loaded: {len(train_dataset)} training samples")

    # Initialize models
    generator = Generator(latent_dim, num_classes).to(device)
    discriminator = Discriminator(num_classes).to(device)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Fixed noise for consistent visualization
    fixed_noise = torch.randn(num_classes, latent_dim, 1, 1, device=device)
    fixed_labels = torch.arange(0, num_classes, dtype=torch.long, device=device)

    print(f"Starting training for {num_epochs} epochs...")

    # Training loop
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size_current = imgs.size(0)

            # --- Train Discriminator ---
            discriminator.zero_grad()

            # 1. Train with real images (with label smoothing)
            real_labels_d = (torch.rand(batch_size_current, device=device) * 0.2 + 0.8)
            output = discriminator(imgs, labels)
            errD_real = criterion(output, real_labels_d)
            errD_real.backward()

            # 2. Train with fake images
            noise = torch.randn(batch_size_current, latent_dim, 1, 1, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size_current,), device=device)
            fake_imgs = generator(noise, fake_labels)

            fake_labels_d = torch.zeros(batch_size_current, device=device)
            output = discriminator(fake_imgs.detach(), fake_labels)
            errD_fake = criterion(output, fake_labels_d)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizer_d.step()

            # --- Train Generator (twice for every discriminator update) ---
            num_generator_updates = 2
            for _ in range(num_generator_updates):
                generator.zero_grad()

                noise = torch.randn(batch_size_current, latent_dim, 1, 1, device=device)
                fake_labels = torch.randint(0, num_classes, (batch_size_current,), device=device)
                fake_imgs = generator(noise, fake_labels)

                output = discriminator(fake_imgs, fake_labels)
                errG = criterion(output, real_labels_d)
                errG.backward()
                optimizer_g.step()

            # --- Print Progress ---
            if i % 100 == 0 or i == len(trainloader) - 1:
                tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(trainloader)}] "
                          f"Loss D: {errD.item():.4f} Loss G: {errG.item():.4f}")

        # --- Save generated images for visualization ---
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            generator.eval()
            with torch.no_grad():
                generated_samples = generator(fixed_noise, fixed_labels).detach().cpu()

                # Unnormalize for visualization
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
                std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)
                generated_samples = generated_samples * std + mean
                generated_samples = torch.clamp(generated_samples, 0, 1)

                # Save images
                os.makedirs("generated_images", exist_ok=True)
                vutils.save_image(generated_samples,
                                f"generated_images/epoch_{epoch+1}.png",
                                nrow=num_classes, normalize=False)

                print(f"Generated images saved for epoch {epoch+1}")
            generator.train()

        # Save model checkpoints
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
            }, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")

    print("Training complete!")

    # Save final model
    torch.save(generator.state_dict(), "final_generator.pth")
    torch.save(discriminator.state_dict(), "final_discriminator.pth")

    # Export generator to ONNX
    print("Exporting generator to ONNX format...")
    generator.eval()

    # Create dummy inputs for ONNX export
    dummy_noise = torch.randn(1, latent_dim, 1, 1, device=device)
    dummy_label = torch.randint(0, num_classes, (1,), device=device)

    # Handle DataParallel wrapper
    model_to_export = generator.module if hasattr(generator, "module") else generator

    torch.onnx.export(
        model_to_export,
        (dummy_noise, dummy_label),
        "conditional_gan_cifar10.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["noise", "class_label"],
        output_names=["generated_image"],
        dynamic_axes={
            "noise": {0: "batch_size"},
            "class_label": {0: "batch_size"},
            "generated_image": {0: "batch_size"}
        }
    )

    print("ONNX export complete: conditional_gan_cifar10.onnx")
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
