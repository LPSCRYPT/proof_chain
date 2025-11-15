#!/usr/bin/env python3
"""
Automated training pipeline for all 30 GAN experiments
Trains models, generates samples, and collects metrics
"""

import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import time

# Add path for imports
sys.path.insert(0, '/root/proof_chain')
sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')

def train_single_experiment(config, tier_num, device='cuda'):
    """Train a single GAN experiment"""

    print(f"\n{'='*70}")
    print(f"TRAINING: {config['experiment_id']} - {config['name']}")
    print(f"Tier: {tier_num} | Expected ops: {config['expected_ops']}")
    print(f"{'='*70}")

    # Import architecture
    from flexible_gan_architectures import create_generator, create_discriminator, estimate_einsum_ops

    # Create models
    generator = create_generator(config).to(device)
    discriminator = create_discriminator(config).to(device)

    # Count parameters and ops
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    estimated_ops = estimate_einsum_ops(generator)

    print(f"Generator params: {gen_params:,}")
    print(f"Discriminator params: {disc_params:,}")
    print(f"Estimated ops: {estimated_ops}")

    # Setup training
    train_config = config['training']
    epochs = min(train_config['epochs'], 30)  # Limit for quick testing
    lr_g = train_config['lr_g']
    lr_d = train_config['lr_d']

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Data loader (use existing CIFAR-10)
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

    # Training metrics
    metrics = {
        'experiment_id': config['experiment_id'],
        'name': config['name'],
        'tier': tier_num,
        'gen_params': gen_params,
        'disc_params': disc_params,
        'estimated_ops': estimated_ops,
        'expected_ops': config['expected_ops'],
        'losses_g': [],
        'losses_d': [],
        'training_time': 0
    }

    # Quick training loop
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss_g = []
        epoch_loss_d = []

        for i, (real_images, labels) in enumerate(dataloader):
            if i > 50:  # Limit iterations for quick testing
                break

            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Labels for loss
            real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # Label smoothing
            fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1

            # Train Discriminator
            optimizer_d.zero_grad()

            # Real images
            outputs_real = discriminator(real_images, labels)
            loss_real = criterion(outputs_real, real_labels)

            # Fake images
            noise = torch.randn(batch_size, config['architecture']['latent_dim'], 1, 1).to(device)
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

        # Record epoch metrics
        metrics['losses_g'].append(np.mean(epoch_loss_g))
        metrics['losses_d'].append(np.mean(epoch_loss_d))

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] Loss_G: {metrics['losses_g'][-1]:.4f}, Loss_D: {metrics['losses_d'][-1]:.4f}")

    metrics['training_time'] = time.time() - start_time

    # Generate samples (10 per class)
    print("\nGenerating samples...")
    generator.eval()
    samples = []

    with torch.no_grad():
        for class_idx in range(10):
            noise = torch.randn(10, config['architecture']['latent_dim'], 1, 1).to(device)
            labels = torch.full((10,), class_idx, dtype=torch.long).to(device)
            fake_images = generator(noise, labels)
            samples.append(fake_images.cpu())

    samples = torch.cat(samples, dim=0)

    # Calculate diversity score
    diversity_scores = []
    for i in range(10):
        class_samples = samples[i*10:(i+1)*10]
        if len(class_samples) > 1:
            pairwise_dists = []
            for j in range(len(class_samples)):
                for k in range(j+1, len(class_samples)):
                    dist = torch.mean((class_samples[j] - class_samples[k]) ** 2).item()
                    pairwise_dists.append(dist)
            diversity_scores.append(np.mean(pairwise_dists) if pairwise_dists else 0)

    metrics['diversity'] = np.mean(diversity_scores)

    # Save model and samples
    output_dir = f"/root/proof_chain/gan_experiments/tier{tier_num}/models"
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = f"{output_dir}/{config['experiment_id']}_generator.pth"
    torch.save(generator.state_dict(), model_path)
    print(f"✓ Model saved: {model_path}")

    # Save samples
    samples_dir = f"/root/proof_chain/gan_experiments/tier{tier_num}/samples"
    os.makedirs(samples_dir, exist_ok=True)
    samples_path = f"{samples_dir}/{config['experiment_id']}_samples.pt"
    torch.save(samples, samples_path)
    print(f"✓ Samples saved: {samples_path}")

    # Save metrics
    metrics['final_loss_g'] = metrics['losses_g'][-1] if metrics['losses_g'] else 0
    metrics['final_loss_d'] = metrics['losses_d'][-1] if metrics['losses_d'] else 0

    print(f"\n✓ Training complete for {config['experiment_id']}")
    print(f"  Time: {metrics['training_time']:.1f}s")
    print(f"  Diversity: {metrics['diversity']:.4f}")
    print(f"  Ops: {estimated_ops} (expected: {config['expected_ops']})")

    return metrics


def train_tier(tier_num, configs, device='cuda'):
    """Train all experiments in a tier"""
    print(f"\n{'='*80}")
    print(f"TRAINING TIER {tier_num} - {len(configs)} experiments")
    print(f"{'='*80}")

    tier_results = []

    for config in configs:
        try:
            metrics = train_single_experiment(config, tier_num, device)
            tier_results.append(metrics)

            # Save intermediate results
            results_path = f"/root/proof_chain/gan_experiments/tier{tier_num}/tier{tier_num}_results.json"
            with open(results_path, 'w') as f:
                json.dump(tier_results, f, indent=2)

        except Exception as e:
            print(f"⚠️ Error training {config['experiment_id']}: {e}")
            continue

    return tier_results


def main():
    """Main training pipeline"""
    print("="*80)
    print("GAN ARCHITECTURE EXPERIMENTS - AUTOMATED TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load configurations
    config_dir = "/root/proof_chain/gan_experiments/scripts"
    all_results = []

    # Train each tier
    for tier_num in range(1, 5):
        config_file = f"{config_dir}/tier{tier_num}_configs.json"

        if not os.path.exists(config_file):
            print(f"⚠️ Config file not found: {config_file}")
            continue

        with open(config_file, 'r') as f:
            configs = json.load(f)

        # For quick testing, only train first 2 experiments per tier
        configs_to_train = configs[:2]  # Remove this limit for full training

        tier_results = train_tier(tier_num, configs_to_train, device)
        all_results.extend(tier_results)

        # Save cumulative results
        results_path = "/root/proof_chain/gan_experiments/results/all_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Generate summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    print(f"\nTotal experiments completed: {len(all_results)}")

    # Find best models
    if all_results:
        # Best for circuit size
        best_circuit = min(all_results, key=lambda x: x['estimated_ops'])
        print(f"\nBest circuit efficiency:")
        print(f"  {best_circuit['experiment_id']}: {best_circuit['estimated_ops']} ops")

        # Best for diversity
        best_diversity = max(all_results, key=lambda x: x['diversity'])
        print(f"\nBest diversity:")
        print(f"  {best_diversity['experiment_id']}: {best_diversity['diversity']:.4f}")

        # Best balance (low ops, high diversity)
        for r in all_results:
            r['balance_score'] = r['diversity'] / (r['estimated_ops'] / 100)
        best_balance = max(all_results, key=lambda x: x['balance_score'])
        print(f"\nBest balance (quality/ops):")
        print(f"  {best_balance['experiment_id']}: score={best_balance['balance_score']:.3f}")

    print(f"\n✓ All training complete!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()