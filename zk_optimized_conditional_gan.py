#!/usr/bin/env python3
"""
ZK-Optimized Conditional GAN Architectures
Designed to minimize einsum operations while maintaining visual quality
Target: <100 einsum operations for deployable verifiers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ZKOptimizedGeneratorV1(nn.Module):
    """
    Minimal architecture: ~40-50 einsum operations
    Trade-off: Lower quality for maximum circuit efficiency
    """
    def __init__(self, latent_dim=80, num_classes=10, embed_dim=20, ngf=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Minimal embedding for class conditioning
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Reduced layers for minimal ops
        self.main = nn.Sequential(
            # Input: 100 (80+20) → 256×4×4
            nn.ConvTranspose2d(latent_dim + embed_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 256×4×4 → 128×8×8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 128×8×8 → 64×16×16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 64×16×16 → 3×32×32
            nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels).view(-1, self.embed_dim, 1, 1)
        gen_input = torch.cat([noise, label_embed], dim=1)
        return self.main(gen_input)

    def get_estimated_einsum_ops(self):
        # Rough estimate: 4 ConvTranspose + 3 BatchNorm + embedding
        return 4 + 3 * 0.5 + 1  # ≈ 7 base ops (actual will be higher with dimensions)

class ZKOptimizedGeneratorV2(nn.Module):
    """
    Balanced architecture: ~60-70 einsum operations
    Trade-off: Good quality with acceptable circuit size
    """
    def __init__(self, latent_dim=100, num_classes=10, embed_dim=50, ngf=48):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Standard embedding for class conditioning
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Balanced architecture
        self.main = nn.Sequential(
            # Input: 150 (100+50) → 384×4×4
            nn.ConvTranspose2d(latent_dim + embed_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 384×4×4 → 192×8×8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 192×8×8 → 96×16×16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 96×16×16 → 48×32×32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 48×32×32 → 3×32×32
            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels).view(-1, self.embed_dim, 1, 1)
        gen_input = torch.cat([noise, label_embed], dim=1)
        return self.main(gen_input)

    def get_estimated_einsum_ops(self):
        return 5 + 4 * 0.5 + 1  # ≈ 8 base ops

class ZKOptimizedGeneratorV3(nn.Module):
    """
    Quality-focused architecture: ~80-90 einsum operations
    Trade-off: Maximum quality while staying under 100 ops
    """
    def __init__(self, latent_dim=100, num_classes=10, embed_dim=50, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Full embedding for best conditioning
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Add skip connection for better gradient flow
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + embed_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Two conv layers for refinement
        self.refine = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels).view(-1, self.embed_dim, 1, 1)
        gen_input = torch.cat([noise, label_embed], dim=1)

        x = self.initial(gen_input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.refine(x)

        return x

    def get_estimated_einsum_ops(self):
        return 6 + 5 * 0.5 + 1  # ≈ 10 base ops

class ZKLiteGenerator(nn.Module):
    """
    Ultra-light architecture: ~30-40 einsum operations
    For extreme circuit constraints
    Uses strided convolutions instead of separate upsampling
    """
    def __init__(self, latent_dim=64, num_classes=10, embed_dim=16, ngf=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Minimal embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Direct path with fewer layers
        self.main = nn.Sequential(
            # 80 → 256×8×8
            nn.ConvTranspose2d(latent_dim + embed_dim, ngf * 8, 8, 1, 0, bias=True),
            nn.ReLU(True),
            # 256×8×8 → 128×16×16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # 128×16×16 → 3×32×32
            nn.ConvTranspose2d(ngf * 4, 3, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels).view(-1, self.embed_dim, 1, 1)
        gen_input = torch.cat([noise, label_embed], dim=1)
        return self.main(gen_input)

    def get_estimated_einsum_ops(self):
        return 3 + 1  # ≈ 4 base ops (no BatchNorm)

class ZKOptimizedDiscriminator(nn.Module):
    """
    Discriminator optimized for ZK circuits
    Uses AvgPool instead of MaxPool, minimal layers
    """
    def __init__(self, num_classes=10, ndf=64):
        super().__init__()
        self.num_classes = num_classes

        # Label embedding for conditional discriminator
        self.label_embedding = nn.Embedding(num_classes, 32*32)

        # Main discriminator path
        self.main = nn.Sequential(
            # Input: 4×32×32 (3 + 1 label channel)
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64×16×16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128×8×8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256×4×4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512×2×2
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        # Embed labels and reshape to image dimensions
        label_embed = self.label_embedding(labels).view(-1, 1, 32, 32)

        # Concatenate image and label
        x = torch.cat([images, label_embed], dim=1)

        return self.main(x).view(-1, 1)

def create_generator_config(target_einsum_ops=75, quality_priority=0.5):
    """
    Create generator configuration based on constraints
    Args:
        target_einsum_ops: Maximum allowed einsum operations
        quality_priority: 0.0 (circuit priority) to 1.0 (quality priority)
    Returns:
        Dictionary with recommended configuration
    """
    configs = {
        'ultra_light': {
            'class': ZKLiteGenerator,
            'params': {'latent_dim': 64, 'embed_dim': 16, 'ngf': 32},
            'estimated_ops': 30,
            'expected_quality': 'Low (IS ~4-5)'
        },
        'minimal': {
            'class': ZKOptimizedGeneratorV1,
            'params': {'latent_dim': 80, 'embed_dim': 20, 'ngf': 32},
            'estimated_ops': 45,
            'expected_quality': 'Moderate (IS ~5-6)'
        },
        'balanced': {
            'class': ZKOptimizedGeneratorV2,
            'params': {'latent_dim': 100, 'embed_dim': 50, 'ngf': 48},
            'estimated_ops': 65,
            'expected_quality': 'Good (IS ~6-7)'
        },
        'quality': {
            'class': ZKOptimizedGeneratorV3,
            'params': {'latent_dim': 100, 'embed_dim': 50, 'ngf': 64},
            'estimated_ops': 85,
            'expected_quality': 'Best (IS ~7-8)'
        }
    }

    # Select configuration based on constraints
    selected = None
    for name, config in configs.items():
        if config['estimated_ops'] <= target_einsum_ops:
            selected = (name, config)

    if selected is None:
        selected = ('ultra_light', configs['ultra_light'])

    # Adjust based on quality priority
    if quality_priority > 0.7 and target_einsum_ops >= 85:
        selected = ('quality', configs['quality'])
    elif quality_priority > 0.5 and target_einsum_ops >= 65:
        selected = ('balanced', configs['balanced'])
    elif quality_priority > 0.3 and target_einsum_ops >= 45:
        selected = ('minimal', configs['minimal'])

    name, config = selected
    print(f"Recommended configuration: {name}")
    print(f"  Estimated ops: {config['estimated_ops']}")
    print(f"  Expected quality: {config['expected_quality']}")

    return config

def test_architecture_sizes():
    """Test and compare all architectures"""
    architectures = [
        ('Ultra-Light', ZKLiteGenerator()),
        ('Minimal (V1)', ZKOptimizedGeneratorV1()),
        ('Balanced (V2)', ZKOptimizedGeneratorV2()),
        ('Quality (V3)', ZKOptimizedGeneratorV3()),
    ]

    print("="*60)
    print("ZK-OPTIMIZED GAN ARCHITECTURE COMPARISON")
    print("="*60)

    for name, model in architectures:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        estimated_ops = model.get_estimated_einsum_ops()

        print(f"\n{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Estimated base einsum ops: {estimated_ops}")

        # Count layers
        conv_layers = sum(1 for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)))
        bn_layers = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))

        print(f"  Conv/ConvTranspose layers: {conv_layers}")
        print(f"  BatchNorm layers: {bn_layers}")

        # Test forward pass
        batch_size = 4
        if hasattr(model, 'latent_dim'):
            noise = torch.randn(batch_size, model.latent_dim, 1, 1)
        else:
            noise = torch.randn(batch_size, 100, 1, 1)
        labels = torch.randint(0, 10, (batch_size,))

        try:
            output = model(noise, labels)
            print(f"  Output shape: {output.shape}")
            print(f"  ✓ Forward pass successful")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")

    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("• For minimal circuit size (testnet): Use Ultra-Light or V1")
    print("• For production with <100 ops: Use V2 (Balanced)")
    print("• For best quality with relaxed constraints: Use V3")
    print("• Always verify actual einsum count with EZKL compilation")

if __name__ == '__main__':
    # Test all architectures
    test_architecture_sizes()

    # Example: Get recommended config for specific constraints
    print("\n" + "="*60)
    print("CONFIGURATION RECOMMENDATION")
    print("="*60)
    config = create_generator_config(target_einsum_ops=75, quality_priority=0.6)