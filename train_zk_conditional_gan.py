#!/usr/bin/env python3
"""
Anti-Collapse Training Pipeline for ZK-Optimized Conditional GAN
Includes mode collapse detection, auto-recovery, and quality monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
from datetime import datetime
from collections import deque, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from zk_optimized_conditional_gan import (
    ZKOptimizedGeneratorV1, ZKOptimizedGeneratorV2,
    ZKOptimizedGeneratorV3, ZKOptimizedDiscriminator
)
from gan_evaluation_metrics import GANEvaluator, calculate_mode_entropy, calculate_diversity_score

class AntiCollapseTrainer:
    """
    GAN trainer with anti-collapse measures and quality monitoring
    """

    def __init__(self, generator, discriminator, device='cuda'):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        # Move models to device
        self.generator.to(device)
        self.discriminator.to(device)

        # Training state
        self.epoch = 0
        self.best_is_score = 0.0
        self.collapse_counter = 0
        self.recovery_attempts = 0

        # Monitoring
        self.loss_history = {'g_loss': [], 'd_loss': [], 'd_real': [], 'd_fake': []}
        self.quality_history = {'is_score': [], 'diversity': [], 'entropy': []}
        self.collapse_events = []

        # Anti-collapse parameters
        self.label_smoothing = 0.1
        self.instance_noise_std = 0.1
        self.instance_noise_decay = 0.995

        # Checkpoints
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def setup_optimizers(self, lr_g=0.0001, lr_d=0.0002, beta1=0.5):
        """Setup optimizers with recommended hyperparameters"""
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(beta1, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, 0.999)
        )

        # Learning rate schedulers for stability
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=100, eta_min=lr_g * 0.1
        )
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=100, eta_min=lr_d * 0.1
        )

    def apply_label_smoothing(self, labels, smoothing=0.1):
        """Apply label smoothing to prevent overconfidence"""
        with torch.no_grad():
            confidence = 1.0 - smoothing
            smoothed = torch.full_like(labels, smoothing / 2)
            smoothed.masked_fill_(labels.bool(), confidence - smoothing / 2)
        return smoothed

    def add_instance_noise(self, images, std=0.1):
        """Add noise to discriminator inputs to prevent overfitting"""
        if std > 0:
            noise = torch.randn_like(images) * std
            images = images + noise
        return images

    def calculate_gradient_penalty(self, real_data, fake_data, labels):
        """Calculate gradient penalty for Wasserstein loss"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated, labels)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def detect_mode_collapse(self, generated_samples, threshold_entropy=1.5, threshold_diversity=0.05):
        """
        Detect mode collapse based on entropy and diversity metrics
        Returns: (is_collapsed, metrics)
        """
        with torch.no_grad():
            # Predict classes for generated samples
            # Note: This assumes a classifier is available
            # For now, we'll use diversity as primary metric
            diversity = calculate_diversity_score(generated_samples)

            # Check for collapse
            is_collapsed = diversity < threshold_diversity

            metrics = {
                'diversity': diversity,
                'threshold': threshold_diversity,
                'is_collapsed': is_collapsed
            }

        return is_collapsed, metrics

    def auto_recover_from_collapse(self):
        """Automatic recovery when collapse is detected"""
        print(f"\n⚠️ Mode collapse detected! Attempting recovery (attempt #{self.recovery_attempts + 1})")

        # 1. Reduce learning rates
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] *= 0.5
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] *= 0.5
        print(f"  → Reduced learning rates by 50%")

        # 2. Increase instance noise
        self.instance_noise_std = min(0.2, self.instance_noise_std * 1.5)
        print(f"  → Increased instance noise to {self.instance_noise_std:.3f}")

        # 3. Increase label smoothing
        self.label_smoothing = min(0.2, self.label_smoothing * 1.2)
        print(f"  → Increased label smoothing to {self.label_smoothing:.3f}")

        # 4. Load last good checkpoint if available
        checkpoint_path = os.path.join(self.checkpoint_dir, 'last_good.pth')
        if os.path.exists(checkpoint_path) and self.recovery_attempts > 1:
            self.load_checkpoint(checkpoint_path)
            print(f"  → Loaded last good checkpoint")

        self.recovery_attempts += 1
        self.collapse_counter = 0

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with anti-collapse measures"""
        self.generator.train()
        self.discriminator.train()

        epoch_stats = {
            'g_loss': [], 'd_loss': [],
            'd_real': [], 'd_fake': [],
            'diversity': []
        }

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (real_images, real_labels) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            real_labels = real_labels.to(self.device)

            # ============================================
            # Train Discriminator
            # ============================================
            self.optimizer_d.zero_grad()

            # Real images
            real_images_noisy = self.add_instance_noise(real_images, self.instance_noise_std)
            real_validity = self.discriminator(real_images_noisy, real_labels)
            real_labels_smooth = self.apply_label_smoothing(
                torch.ones_like(real_validity), self.label_smoothing
            )
            d_real_loss = F.binary_cross_entropy(real_validity, real_labels_smooth)

            # Fake images
            noise = torch.randn(batch_size, self.generator.latent_dim, 1, 1).to(self.device)
            fake_labels = torch.randint(0, 10, (batch_size,)).to(self.device)
            fake_images = self.generator(noise, fake_labels)
            fake_images_noisy = self.add_instance_noise(fake_images.detach(), self.instance_noise_std)
            fake_validity = self.discriminator(fake_images_noisy, fake_labels)
            fake_labels_smooth = torch.zeros_like(fake_validity) + self.label_smoothing / 2
            d_fake_loss = F.binary_cross_entropy(fake_validity, fake_labels_smooth)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
            self.optimizer_d.step()

            # ============================================
            # Train Generator
            # ============================================
            self.optimizer_g.zero_grad()

            # Generate fake images
            noise = torch.randn(batch_size, self.generator.latent_dim, 1, 1).to(self.device)
            gen_labels = torch.randint(0, 10, (batch_size,)).to(self.device)
            gen_images = self.generator(noise, gen_labels)
            gen_validity = self.discriminator(gen_images, gen_labels)
            g_loss = F.binary_cross_entropy(
                gen_validity,
                torch.ones_like(gen_validity) - self.label_smoothing / 2
            )

            # Add diversity regularization
            if batch_size > 1:
                diversity = calculate_diversity_score(gen_images)
                diversity_penalty = max(0, 0.1 - diversity) * 10  # Penalize low diversity
                g_loss = g_loss + diversity_penalty
                epoch_stats['diversity'].append(diversity)

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.0)
            self.optimizer_g.step()

            # Record stats
            epoch_stats['g_loss'].append(g_loss.item())
            epoch_stats['d_loss'].append(d_loss.item())
            epoch_stats['d_real'].append(real_validity.mean().item())
            epoch_stats['d_fake'].append(fake_validity.mean().item())

            # Update progress bar
            progress_bar.set_postfix({
                'G': f"{g_loss.item():.3f}",
                'D': f"{d_loss.item():.3f}",
                'D(x)': f"{real_validity.mean().item():.3f}",
                'D(G(z))': f"{fake_validity.mean().item():.3f}"
            })

            # Periodic collapse check
            if batch_idx % 100 == 0 and batch_idx > 0:
                with torch.no_grad():
                    test_noise = torch.randn(100, self.generator.latent_dim, 1, 1).to(self.device)
                    test_labels = torch.arange(10).repeat(10).to(self.device)
                    test_images = self.generator(test_noise, test_labels)

                    is_collapsed, collapse_metrics = self.detect_mode_collapse(test_images)
                    if is_collapsed:
                        self.collapse_counter += 1
                        if self.collapse_counter >= 3:
                            self.auto_recover_from_collapse()
                            self.collapse_events.append({
                                'epoch': epoch,
                                'batch': batch_idx,
                                'metrics': collapse_metrics
                            })

        # Decay instance noise
        self.instance_noise_std *= self.instance_noise_decay

        return epoch_stats

    def evaluate(self, num_samples=1000):
        """Evaluate generator quality"""
        self.generator.eval()

        with torch.no_grad():
            # Generate samples from all classes
            samples_per_class = num_samples // 10
            all_samples = []

            for class_idx in range(10):
                noise = torch.randn(samples_per_class, self.generator.latent_dim, 1, 1).to(self.device)
                labels = torch.full((samples_per_class,), class_idx).to(self.device)
                samples = self.generator(noise, labels)
                all_samples.append(samples)

            all_samples = torch.cat(all_samples, dim=0)

            # Calculate metrics
            diversity = calculate_diversity_score(all_samples)

            # Note: IS and FID would require more complex evaluation
            metrics = {
                'diversity': diversity,
                'num_samples': num_samples
            }

        return metrics

    def save_checkpoint(self, path):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_d_state': self.optimizer_d.state_dict(),
            'loss_history': self.loss_history,
            'quality_history': self.quality_history,
            'collapse_events': self.collapse_events,
            'best_is_score': self.best_is_score
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
        self.epoch = checkpoint['epoch']
        self.loss_history = checkpoint['loss_history']
        self.quality_history = checkpoint['quality_history']
        self.collapse_events = checkpoint.get('collapse_events', [])
        self.best_is_score = checkpoint.get('best_is_score', 0.0)

    def train(self, num_epochs=200, dataloader=None, eval_frequency=10):
        """Full training loop with monitoring"""

        # Create dummy dataloader if not provided
        if dataloader is None:
            print("Creating CIFAR-10 dataloader...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

        print(f"\nStarting training for {num_epochs} epochs")
        print("="*60)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train one epoch
            epoch_stats = self.train_epoch(dataloader, epoch)

            # Record statistics
            for key in ['g_loss', 'd_loss', 'd_real', 'd_fake']:
                if key in epoch_stats and epoch_stats[key]:
                    self.loss_history[key].append(np.mean(epoch_stats[key]))

            # Evaluate periodically
            if (epoch + 1) % eval_frequency == 0:
                print(f"\nEvaluating at epoch {epoch + 1}...")
                eval_metrics = self.evaluate()

                self.quality_history['diversity'].append(eval_metrics['diversity'])

                print(f"  Diversity: {eval_metrics['diversity']:.4f}")

                # Save checkpoint if good
                if eval_metrics['diversity'] > 0.08:  # Good diversity threshold
                    checkpoint_path = os.path.join(self.checkpoint_dir, 'last_good.pth')
                    self.save_checkpoint(checkpoint_path)
                    print(f"  ✓ Saved good checkpoint")

            # Regular checkpoint
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch + 1}.pth')
                self.save_checkpoint(checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")

            # Update schedulers
            self.scheduler_g.step()
            self.scheduler_d.step()

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total epochs: {num_epochs}")
        print(f"Mode collapse events: {len(self.collapse_events)}")
        print(f"Recovery attempts: {self.recovery_attempts}")

        # Save final model
        final_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        self.save_checkpoint(final_path)
        print(f"Final model saved to: {final_path}")

        return self.loss_history, self.quality_history

def plot_training_history(loss_history, quality_history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss plots
    axes[0, 0].plot(loss_history['g_loss'], label='Generator')
    axes[0, 0].plot(loss_history['d_loss'], label='Discriminator')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Discriminator predictions
    axes[0, 1].plot(loss_history['d_real'], label='D(real)')
    axes[0, 1].plot(loss_history['d_fake'], label='D(fake)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Discriminator Output')
    axes[0, 1].set_title('Discriminator Predictions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Diversity over time
    if 'diversity' in quality_history:
        axes[1, 0].plot(quality_history['diversity'])
        axes[1, 0].set_xlabel('Evaluation Step')
        axes[1, 0].set_ylabel('Diversity Score')
        axes[1, 0].set_title('Image Diversity')
        axes[1, 0].axhline(y=0.05, color='r', linestyle='--', label='Collapse Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # IS score if available
    if 'is_score' in quality_history and quality_history['is_score']:
        axes[1, 1].plot(quality_history['is_score'])
        axes[1, 1].set_xlabel('Evaluation Step')
        axes[1, 1].set_ylabel('Inception Score')
        axes[1, 1].set_title('Inception Score')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'IS Score not available',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    return fig

if __name__ == '__main__':
    # Example usage
    print("="*60)
    print("ANTI-COLLAPSE GAN TRAINING PIPELINE")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize models (using balanced architecture)
    print("\nInitializing models...")
    generator = ZKOptimizedGeneratorV2(latent_dim=100, embed_dim=50, ngf=48)
    discriminator = ZKOptimizedDiscriminator(num_classes=10, ndf=64)

    # Create trainer
    trainer = AntiCollapseTrainer(generator, discriminator, device)
    trainer.setup_optimizers(lr_g=0.0001, lr_d=0.0002)

    print("✓ Models initialized")
    print("✓ Trainer configured with anti-collapse measures:")
    print(f"  - Label smoothing: {trainer.label_smoothing}")
    print(f"  - Instance noise: {trainer.instance_noise_std}")
    print(f"  - Auto-recovery enabled")
    print(f"  - Gradient clipping enabled")

    # Ready to train
    print("\nTrainer ready. To start training, call:")
    print("  loss_history, quality_history = trainer.train(num_epochs=200)")

    print("\n" + "="*60)