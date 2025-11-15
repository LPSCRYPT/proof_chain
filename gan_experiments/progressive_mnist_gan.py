#!/usr/bin/env python3
"""
Progressive Resolution GAN training with MNIST
Tests if simpler datasets and progressive training improve ZK-optimized GAN performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import sys

# Add path for imports
sys.path.insert(0, '/root/proof_chain')
sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')

class ProgressiveMNISTGenerator(nn.Module):
    """Progressive resolution generator for MNIST (simpler than CIFAR)"""

    def __init__(self, latent_dim=50, num_classes=10, embed_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.current_resolution = 7  # Start with 7x7 (MNIST natural size after pooling)

        # Class embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Progressive layers (7x7 -> 14x14 -> 28x28)
        total_input = latent_dim + embed_dim

        # Initial MLP to 7x7
        self.initial = nn.Sequential(
            nn.Linear(total_input, 128),
            nn.ReLU(),
            nn.Linear(128, 7 * 7 * 16),
            nn.ReLU()
        )

        # 7x7 -> 14x14 upsampling layer
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        # 14x14 -> 28x28 upsampling layer
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        # Output layers for each resolution
        self.output_7x7 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh()
        )

        self.output_14x14 = nn.Sequential(
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Tanh()
        )

        self.output_28x28 = nn.Sequential(
            nn.Conv2d(4, 1, 3, 1, 1),
            nn.Tanh()
        )

    def set_resolution(self, resolution):
        """Set the target resolution for progressive training"""
        assert resolution in [7, 14, 28], f"Invalid resolution: {resolution}"
        self.current_resolution = resolution

    def forward(self, noise, labels):
        # Embed labels
        label_embed = self.label_embedding(labels)

        # Concatenate noise and labels
        x = torch.cat([noise.view(noise.size(0), -1), label_embed], dim=1)

        # Initial projection to 7x7
        x = self.initial(x)
        x = x.view(x.size(0), 16, 7, 7)

        if self.current_resolution == 7:
            return self.output_7x7(x)

        # Upsample to 14x14
        x = self.upsample1(x)
        if self.current_resolution == 14:
            return self.output_14x14(x)

        # Upsample to 28x28
        x = self.upsample2(x)
        return self.output_28x28(x)

    def count_ops_at_resolution(self, resolution):
        """Estimate einsum operations at each resolution"""
        ops = 0

        # Base layers always used
        ops += 1  # embedding
        ops += 2  # initial MLP layers

        if resolution == 7:
            ops += 1  # output conv
        elif resolution == 14:
            ops += 2  # upsample1 conv
            ops += 1  # batchnorm
            ops += 1  # output conv
        elif resolution == 28:
            ops += 2  # upsample1 conv
            ops += 1  # batchnorm
            ops += 2  # upsample2 conv
            ops += 1  # batchnorm
            ops += 1  # output conv

        return ops


class SimpleMNISTDiscriminator(nn.Module):
    """Simple discriminator for MNIST"""

    def __init__(self, num_classes=10, embed_dim=20):
        super().__init__()
        self.embed_dim = embed_dim

        # Adaptive input based on resolution
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)

        # Adaptive pooling to handle different resolutions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Class embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Linear(embed_dim, 128)

        # Output
        self.output = nn.Linear(128, 1)

    def forward(self, x, labels):
        # Convolutional features
        x = F.leaky_relu(self.conv1(x), 0.2)

        # Only use deeper layers for higher resolutions
        if x.size(2) > 4:
            x = F.leaky_relu(self.conv2(x), 0.2)
        if x.size(2) > 4:
            x = F.leaky_relu(self.conv3(x), 0.2)

        # Adaptive pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Class conditioning
        label_embed = self.label_embedding(labels)
        label_feat = self.label_proj(label_embed)

        # Combine and output
        x = x * label_feat
        return self.output(x)


def train_progressive_mnist(device='cuda'):
    """Train progressive MNIST GAN"""
    print("="*80)
    print("PROGRESSIVE MNIST GAN TRAINING")
    print("Testing if simpler dataset improves ZK optimization")
    print("="*80)

    # Create models
    generator = ProgressiveMNISTGenerator(latent_dim=50, num_classes=10, embed_dim=20).to(device)
    discriminator = SimpleMNISTDiscriminator(num_classes=10, embed_dim=20).to(device)

    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())

    print(f"\nModel Statistics:")
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")

    # Progressive training schedule
    resolutions = [7, 14, 28]
    epochs_per_resolution = [10, 10, 20]

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Results tracking
    results = {
        'dataset': 'MNIST',
        'gen_params': gen_params,
        'disc_params': disc_params,
        'losses': [],
        'resolution_metrics': {}
    }

    # Training loop for each resolution
    for res_idx, (resolution, epochs) in enumerate(zip(resolutions, epochs_per_resolution)):
        print(f"\n{'='*60}")
        print(f"Training at {resolution}x{resolution} resolution")
        print(f"Estimated ops: {generator.count_ops_at_resolution(resolution)}")
        print("="*60)

        generator.set_resolution(resolution)

        # Adjust transforms for current resolution
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load MNIST dataset
        dataset = torchvision.datasets.MNIST(
            root='/root/proof_chain/data',
            train=True,
            download=True,
            transform=transform
        )

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

        # Train at this resolution
        res_losses_g = []
        res_losses_d = []

        for epoch in range(epochs):
            epoch_loss_g = []
            epoch_loss_d = []

            for i, (real_images, labels) in enumerate(dataloader):
                if i > 100:  # Limit iterations for quick testing
                    break

                batch_size = real_images.size(0)
                real_images = real_images.to(device)
                labels = labels.to(device)

                # Labels for loss
                real_labels = torch.ones(batch_size, 1).to(device) * 0.9
                fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1

                # Train Discriminator
                optimizer_d.zero_grad()

                # Real images
                outputs_real = discriminator(real_images, labels)
                loss_real = criterion(outputs_real, real_labels)

                # Fake images
                noise = torch.randn(batch_size, 50, 1, 1).to(device)
                fake_images = generator(noise, labels)
                outputs_fake = discriminator(fake_images.detach(), labels)
                loss_fake = criterion(outputs_fake, fake_labels)

                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()

                outputs_fake = discriminator(fake_images, labels)
                loss_g = criterion(outputs_fake, real_labels)
                loss_g.backward()
                optimizer_g.step()

                epoch_loss_g.append(loss_g.item())
                epoch_loss_d.append(loss_d.item())

            avg_loss_g = np.mean(epoch_loss_g)
            avg_loss_d = np.mean(epoch_loss_d)
            res_losses_g.append(avg_loss_g)
            res_losses_d.append(avg_loss_d)

            if epoch % 5 == 0:
                print(f"Epoch [{epoch}/{epochs}] Loss_G: {avg_loss_g:.4f}, Loss_D: {avg_loss_d:.4f}")

        # Test quality at this resolution
        print(f"\nTesting quality at {resolution}x{resolution}...")
        generator.eval()

        with torch.no_grad():
            # Generate samples for each class
            samples = []
            for class_idx in range(10):
                noise = torch.randn(10, 50, 1, 1).to(device)
                labels = torch.full((10,), class_idx, dtype=torch.long).to(device)
                fake_images = generator(noise, labels)
                samples.append(fake_images.cpu())

            samples = torch.cat(samples, dim=0)

            # Calculate diversity
            diversity_scores = []
            for i in range(10):
                class_samples = samples[i*10:(i+1)*10]
                if len(class_samples) > 1:
                    pairwise_dists = []
                    for j in range(len(class_samples)):
                        for k in range(j+1, len(class_samples)):
                            dist = torch.mean((class_samples[j] - class_samples[k]) ** 2).item()
                            pairwise_dists.append(dist)
                    diversity_scores.append(np.mean(pairwise_dists))

            mean_diversity = np.mean(diversity_scores)

            # Calculate class separation
            class_means = []
            for i in range(10):
                class_samples = samples[i*10:(i+1)*10]
                class_mean = torch.mean(class_samples, dim=0)
                class_means.append(class_mean)

            separations = []
            for i in range(10):
                for j in range(i+1, 10):
                    sep = torch.mean((class_means[i] - class_means[j]) ** 2).item()
                    separations.append(sep)

            mean_separation = np.mean(separations)

        generator.train()

        # Store metrics
        results['resolution_metrics'][f"{resolution}x{resolution}"] = {
            'ops': generator.count_ops_at_resolution(resolution),
            'final_loss_g': res_losses_g[-1],
            'final_loss_d': res_losses_d[-1],
            'diversity': mean_diversity,
            'class_separation': mean_separation
        }

        print(f"Results at {resolution}x{resolution}:")
        print(f"  Diversity: {mean_diversity:.4f}")
        print(f"  Class separation: {mean_separation:.4f}")

    # Final assessment
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING RESULTS")
    print("="*80)

    print("\nComparison across resolutions:")
    for res, metrics in results['resolution_metrics'].items():
        print(f"\n{res}:")
        print(f"  Ops: {metrics['ops']}")
        print(f"  Diversity: {metrics['diversity']:.4f}")
        print(f"  Separation: {metrics['class_separation']:.4f}")
        print(f"  Final Loss G: {metrics['final_loss_g']:.4f}")

    # Compare to CIFAR results
    print("\n" + "="*60)
    print("MNIST vs CIFAR-10 COMPARISON")
    print("="*60)

    mnist_28_metrics = results['resolution_metrics']['28x28']
    print(f"\nMNIST (28x28):")
    print(f"  Ops: {mnist_28_metrics['ops']}")
    print(f"  Diversity: {mnist_28_metrics['diversity']:.4f}")
    print(f"  Parameters: {gen_params:,}")

    print(f"\nCIFAR-10 MLP (32x32) - Previous Best:")
    print(f"  Ops: 20")
    print(f"  Diversity: 0.1168")
    print(f"  Accuracy: 19.4%")

    improvement = (mnist_28_metrics['diversity'] / 0.1168 - 1) * 100 if mnist_28_metrics['diversity'] > 0 else 0
    print(f"\nDiversity improvement: {improvement:+.1f}%")

    if mnist_28_metrics['ops'] < 20:
        print(f"✅ BETTER: Lower ops ({mnist_28_metrics['ops']} vs 20)")
    elif mnist_28_metrics['diversity'] > 0.15:
        print(f"✅ BETTER: Higher diversity ({mnist_28_metrics['diversity']:.3f} vs 0.117)")
    else:
        print(f"⚠️  Similar performance, simpler dataset didn't help significantly")

    # Save model
    model_path = "/root/proof_chain/gan_experiments/progressive_mnist_generator.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(generator.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")

    # Save results
    results_path = "/root/proof_chain/gan_experiments/progressive_mnist_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_path}")

    return results


def test_depthwise_separable():
    """Quick test of depthwise separable convolutions for efficiency"""
    print("\n" + "="*80)
    print("TESTING DEPTHWISE SEPARABLE CONVOLUTIONS")
    print("="*80)

    class DepthwiseSeparableConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            super().__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                       stride, padding, groups=in_channels)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x

    # Compare ops
    regular_conv = nn.Conv2d(64, 128, 3, 1, 1)
    depthwise_sep = DepthwiseSeparableConv(64, 128, 3, 1, 1)

    regular_params = sum(p.numel() for p in regular_conv.parameters())
    depthwise_params = sum(p.numel() for p in depthwise_sep.parameters())

    print(f"\nParameter comparison (3x3, 64->128 channels):")
    print(f"Regular Conv2d: {regular_params:,} params (2 einsum ops)")
    print(f"Depthwise Separable: {depthwise_params:,} params (2 einsum ops)")
    print(f"Reduction: {(1 - depthwise_params/regular_params)*100:.1f}%")

    # For ZK circuits, depthwise doesn't reduce ops but reduces parameters
    print(f"\n⚠️ Note: For ZK circuits, both still count as 2 ops")
    print(f"   But reduced parameters mean smaller proof size")

    return {
        'regular_params': regular_params,
        'depthwise_params': depthwise_params,
        'param_reduction': (1 - depthwise_params/regular_params)
    }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Run progressive MNIST training
    mnist_results = train_progressive_mnist(device)

    # Test depthwise separable convolutions
    depthwise_results = test_depthwise_separable()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)