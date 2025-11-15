#!/usr/bin/env python3
"""
Binary and Ternary Weight Networks for ZK-optimized GANs
Tests extreme quantization for circuit size reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import sys
from datetime import datetime

# Add path for imports
sys.path.insert(0, '/root/proof_chain')
sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')


class BinaryQuantize(torch.autograd.Function):
    """Binary quantization: weights in {-1, +1}"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Straight-through estimator
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input


class TernaryQuantize(torch.autograd.Function):
    """Ternary quantization: weights in {-1, 0, +1}"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Threshold at 0.7 * std
        threshold = 0.7 * input.std()
        output = input.clone()
        output[input.abs() < threshold] = 0
        output[input >= threshold] = 1
        output[input <= -threshold] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Straight-through estimator
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input


class BinaryLinear(nn.Module):
    """Linear layer with binary weights"""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialize
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        # Quantize weights to binary
        binary_weight = BinaryQuantize.apply(self.weight)

        # Scale factor for binary weights (important for training stability)
        scale = self.weight.abs().mean()

        return F.linear(x, binary_weight * scale, self.bias)


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights"""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialize
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        # Quantize weights to ternary
        ternary_weight = TernaryQuantize.apply(self.weight)

        # Scale factor
        scale = self.weight[self.weight != 0].abs().mean() if (self.weight != 0).any() else 1.0

        return F.linear(x, ternary_weight * scale, self.bias)


class BinaryConv2d(nn.Module):
    """Conv2d with binary weights"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Initialize
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        # Quantize weights
        binary_weight = BinaryQuantize.apply(self.weight)

        # Scale factor
        scale = self.weight.abs().mean()

        return F.conv2d(x, binary_weight * scale, self.bias, self.stride, self.padding)


class QuantizedGenerator(nn.Module):
    """Generator with quantized weights for minimal ZK circuits"""

    def __init__(self, latent_dim=100, num_classes=10, embed_dim=50, quantization='binary'):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.quantization = quantization

        # Choose quantization type
        if quantization == 'binary':
            LinearLayer = BinaryLinear
        elif quantization == 'ternary':
            LinearLayer = TernaryLinear
        else:  # 'none' for baseline
            LinearLayer = lambda i, o, bias=False: nn.Linear(i, o, bias=bias)

        # Class embedding (keep full precision for better conditioning)
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Main network with quantized weights
        self.fc1 = LinearLayer(latent_dim + embed_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = LinearLayer(256, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = LinearLayer(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc4 = LinearLayer(1024, 3 * 32 * 32)

    def forward(self, noise, labels):
        # Embed labels
        label_embed = self.label_embedding(labels)

        # Concatenate
        x = torch.cat([noise.view(noise.size(0), -1), label_embed], dim=1)

        # Forward through quantized layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.tanh(self.fc4(x))

        return x.view(x.size(0), 3, 32, 32)

    def count_ops(self):
        """Estimate ops for ZK circuit"""
        ops = 0
        ops += 1  # embedding
        ops += 4  # 4 linear layers (quantized still count as matmul)
        ops += 3  # 3 batchnorm
        return ops

    def get_sparsity(self):
        """Calculate weight sparsity (for ternary)"""
        if self.quantization != 'ternary':
            return 0.0

        total_params = 0
        zero_params = 0

        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                quantized = TernaryQuantize.apply(param)
                total_params += quantized.numel()
                zero_params += (quantized == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0


class SimpleDiscriminator(nn.Module):
    """Standard discriminator for training"""
    def __init__(self, num_classes=10, embed_dim=50):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        # Class embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Linear(embed_dim, 256)

        # Output
        self.fc = nn.Linear(256 * 4 * 4, 1)

    def forward(self, x, labels):
        # Convolutional features
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Class conditioning
        label_embed = self.label_embedding(labels)
        label_feat = self.label_proj(label_embed)
        label_feat = label_feat.view(label_feat.size(0), 256, 1).expand(-1, -1, 16)
        label_feat = label_feat.view(label_feat.size(0), -1)

        # Combine
        x = x * torch.sigmoid(label_feat)

        return self.fc(x)


def train_quantized_gan(quantization_type='binary', device='cuda'):
    """Train GAN with quantized weights"""
    print("="*80)
    print(f"TRAINING {quantization_type.upper()} QUANTIZED GAN")
    print("="*80)

    # Create models
    generator = QuantizedGenerator(
        latent_dim=100,
        num_classes=10,
        embed_dim=50,
        quantization=quantization_type
    ).to(device)

    discriminator = SimpleDiscriminator(
        num_classes=10,
        embed_dim=50
    ).to(device)

    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())

    print(f"\nModel Statistics:")
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Estimated ops: {generator.count_ops()}")

    if quantization_type == 'ternary':
        print(f"Initial sparsity: {generator.get_sparsity():.1%}")

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='/root/proof_chain/data',
        train=True,
        download=False,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # Training
    epochs = 20
    results = {
        'quantization': quantization_type,
        'gen_params': gen_params,
        'ops': generator.count_ops(),
        'losses_g': [],
        'losses_d': [],
        'sparsity': []
    }

    print("\nStarting training...")
    for epoch in range(epochs):
        epoch_loss_g = []
        epoch_loss_d = []

        for i, (real_images, labels) in enumerate(dataloader):
            if i > 100:  # Quick training
                break

            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Labels for loss
            real_labels = torch.ones(batch_size, 1).to(device) * 0.9
            fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1

            # Train Discriminator
            optimizer_d.zero_grad()

            outputs_real = discriminator(real_images, labels)
            loss_real = criterion(outputs_real, real_labels)

            noise = torch.randn(batch_size, 100, 1, 1).to(device)
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
        results['losses_g'].append(avg_loss_g)
        results['losses_d'].append(avg_loss_d)

        if quantization_type == 'ternary':
            sparsity = generator.get_sparsity()
            results['sparsity'].append(sparsity)

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{epochs}] Loss_G: {avg_loss_g:.4f}, Loss_D: {avg_loss_d:.4f}", end="")
            if quantization_type == 'ternary':
                print(f", Sparsity: {sparsity:.1%}")
            else:
                print()

    # Test generation quality
    print("\nTesting generation quality...")
    generator.eval()

    with torch.no_grad():
        samples = []
        for class_idx in range(10):
            noise = torch.randn(10, 100, 1, 1).to(device)
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

        results['diversity'] = np.mean(diversity_scores)

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

        results['class_separation'] = np.mean(separations)

    print(f"\nResults:")
    print(f"  Diversity: {results['diversity']:.4f}")
    print(f"  Class separation: {results['class_separation']:.4f}")
    print(f"  Final Loss G: {results['losses_g'][-1]:.4f}")
    print(f"  Ops count: {results['ops']}")

    if quantization_type == 'ternary' and results['sparsity']:
        print(f"  Final sparsity: {results['sparsity'][-1]:.1%}")

    # Save model
    model_path = f"/root/proof_chain/gan_experiments/{quantization_type}_quantized_generator.pth"
    torch.save(generator.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")

    return results


def compare_quantization_methods(device='cuda'):
    """Compare different quantization approaches"""
    print("="*80)
    print("COMPARING QUANTIZATION METHODS FOR ZK-OPTIMIZED GANS")
    print("="*80)

    all_results = {}

    # Test each quantization type
    for quant_type in ['none', 'binary', 'ternary']:
        print(f"\n{'='*60}")
        print(f"Testing {quant_type} quantization...")
        print("="*60)

        results = train_quantized_gan(quant_type, device)
        all_results[quant_type] = results

    # Summary comparison
    print("\n" + "="*80)
    print("QUANTIZATION COMPARISON SUMMARY")
    print("="*80)

    print("\n{:<12} {:>8} {:>12} {:>12} {:>12} {:>12}".format(
        "Type", "Ops", "Parameters", "Diversity", "Separation", "Final Loss"
    ))
    print("-"*80)

    for quant_type in ['none', 'binary', 'ternary']:
        r = all_results[quant_type]
        print("{:<12} {:>8} {:>12,} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            quant_type.capitalize(),
            r['ops'],
            r['gen_params'],
            r['diversity'],
            r['class_separation'],
            r['losses_g'][-1]
        ))

        if quant_type == 'ternary' and r['sparsity']:
            print(f"{'':12} {'':8} Sparsity: {r['sparsity'][-1]:>6.1%}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS FOR ZK DEPLOYMENT")
    print("="*60)

    baseline = all_results['none']
    binary = all_results['binary']
    ternary = all_results['ternary']

    print("\nQuantization Impact on Quality:")
    print(f"  Binary diversity loss: {(1 - binary['diversity']/baseline['diversity'])*100:.1f}%")
    print(f"  Ternary diversity loss: {(1 - ternary['diversity']/baseline['diversity'])*100:.1f}%")

    print("\nQuantization Benefits for ZK:")
    print(f"  Binary: Weights in {{-1, +1}} - simpler constraints")
    print(f"  Ternary: {ternary['sparsity'][-1]:.1%} sparsity - fewer active weights")

    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)

    if binary['diversity'] > 0.08 and binary['class_separation'] > 0.15:
        print("✅ Binary quantization recommended:")
        print("   - Acceptable quality retention")
        print("   - Simpler ZK constraints")
        print("   - Weights only need 1 bit")
    elif ternary['diversity'] > 0.09 and ternary['sparsity'][-1] > 0.3:
        print("✅ Ternary quantization recommended:")
        print(f"   - {ternary['sparsity'][-1]:.1%} weight sparsity")
        print("   - Better quality than binary")
        print("   - Sparse matrix optimizations possible")
    else:
        print("⚠️  Quantization degrades quality too much")
        print("   - Consider other optimization approaches")

    # Save comparison
    comparison_path = "/root/proof_chain/gan_experiments/quantization_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Comparison saved to {comparison_path}")

    return all_results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Run comparison
    results = compare_quantization_methods(device)