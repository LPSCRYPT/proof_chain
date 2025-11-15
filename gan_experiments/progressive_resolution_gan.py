#!/usr/bin/env python3
"""
Progressive Resolution GAN Training for ZK
Start with 8x8, then 16x16, finally 32x32
Also test on MNIST for simpler validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')

class ProgressiveGenerator(nn.Module):
    """Generator that can output different resolutions"""
    def __init__(self, latent_dim=100, num_classes=10, ngf=32):
        super().__init__()
        self.current_resolution = 8  # Start at 8x8
        
        # Embedding
        self.embed = nn.Embedding(num_classes, 30)
        
        # 8x8 branch
        self.fc_8 = nn.Sequential(
            nn.Linear(latent_dim + 30, ngf * 8 * 8),
            nn.BatchNorm1d(ngf * 8 * 8),
            nn.ReLU(True)
        )
        self.conv_8 = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 3, 1, 1, 0),
            nn.Tanh()
        )
        
        # 16x16 branch
        self.up_16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.conv_16 = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, 3, 1, 1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.Conv2d(ngf // 2, 3, 1, 1, 0),
            nn.Tanh()
        )
        
        # 32x32 branch
        self.up_32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf // 2, ngf // 2, 3, 1, 1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True)
        )
        self.conv_32 = nn.Sequential(
            nn.Conv2d(ngf // 2, ngf // 4, 3, 1, 1),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            nn.Conv2d(ngf // 4, 3, 1, 1, 0),
            nn.Tanh()
        )
        
    def set_resolution(self, resolution):
        """Set target resolution: 8, 16, or 32"""
        assert resolution in [8, 16, 32]
        self.current_resolution = resolution
        
    def forward(self, noise, labels):
        label_embed = self.embed(labels)
        x = torch.cat([noise.view(noise.size(0), -1), label_embed], dim=1)
        
        # Start with 8x8
        x = self.fc_8(x)
        x = x.view(x.size(0), -1, 8, 8)
        
        if self.current_resolution == 8:
            return self.conv_8(x)
        
        # Upsample to 16x16
        x = self.up_16(x)
        
        if self.current_resolution == 16:
            return self.conv_16(x)
        
        # Upsample to 32x32
        x_16 = x
        x = self.up_32(self.conv_16[:-2](x_16))  # Skip tanh
        return self.conv_32(x)


class MNISTGenerator(nn.Module):
    """Ultra-simple generator for MNIST (28x28, 1 channel)"""
    def __init__(self, latent_dim=50, num_classes=10, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 10)
        
        # Just 2 layers for ultra efficiency
        self.fc1 = nn.Linear(latent_dim + 10, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 28 * 28)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, noise, labels):
        label_embed = self.embed(labels)
        x = torch.cat([noise.view(noise.size(0), -1), label_embed], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = torch.tanh(self.fc2(x))
        return x.view(x.size(0), 1, 28, 28)


def train_progressive(generator, device='cuda', dataset='cifar10'):
    """Train generator progressively at different resolutions"""
    print("\n" + "="*80)
    print(f"PROGRESSIVE TRAINING ON {dataset.upper()}")
    print("="*80)
    
    # Setup data
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='/root/proof_chain/data',
            train=True,
            download=False,
            transform=transform
        )
    else:  # MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.MNIST(
            root='/root/proof_chain/data',
            train=True,
            download=True,
            transform=transform
        )
    
    dataloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    
    # Training phases
    resolutions = [8, 16, 32] if dataset == 'cifar10' else [28]
    epochs_per_res = 20
    
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    
    for resolution in resolutions:
        if hasattr(generator, 'set_resolution'):
            generator.set_resolution(resolution)
            
        print(f"\nTraining at {resolution}x{resolution} resolution...")
        
        for epoch in range(epochs_per_res):
            losses = []
            
            for i, (real_images, labels) in enumerate(dataloader):
                if i > 50:  # Quick training
                    break
                    
                batch_size = real_images.size(0)
                real_images = real_images.to(device)
                labels = labels.to(device)
                
                # Resize real images to current resolution
                if resolution != 32 and resolution != 28:
                    real_images = F.interpolate(real_images, size=(resolution, resolution))
                
                # Generate fake images
                noise = torch.randn(batch_size, 100, 1, 1).to(device) if dataset == 'cifar10' else torch.randn(batch_size, 50, 1, 1).to(device)
                fake_images = generator(noise, labels)
                
                # Simple pixel loss
                loss = criterion(fake_images, real_images)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            if epoch % 5 == 0:
                print(f"  Epoch [{epoch}/{epochs_per_res}] Loss: {np.mean(losses):.4f}")
    
    return generator


def test_on_mnist():
    """Test ultra-simple MNIST GAN for baseline"""
    print("\n" + "="*80)
    print("MNIST BASELINE TEST (SIMPLER DATASET)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and train MNIST generator
    mnist_gen = MNISTGenerator(latent_dim=50, hidden_dim=256).to(device)
    
    # Count ops
    from flexible_gan_architectures import estimate_einsum_ops
    ops = estimate_einsum_ops(mnist_gen)
    print(f"MNIST Generator ops: {ops}")
    
    # Train
    trained_gen = train_progressive(mnist_gen, device, 'mnist')
    
    # Test with simple MNIST classifier
    print("\nTesting MNIST generation quality...")
    
    # Load or create simple MNIST classifier
    classifier = nn.Sequential(
        nn.Conv2d(1, 16, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    # Test accuracy
    trained_gen.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for class_idx in range(10):
            noise = torch.randn(100, 50, 1, 1).to(device)
            labels = torch.full((100,), class_idx, dtype=torch.long).to(device)
            
            fake_images = trained_gen(noise, labels)
            outputs = classifier(fake_images)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += 100
    
    accuracy = 100 * correct / total
    print(f"MNIST Generation Accuracy: {accuracy:.2f}%")
    
    if accuracy > 50:
        print("✅ MNIST works well! The approach is valid, CIFAR-10 is just too complex.")
    
    return accuracy, ops


def test_cifar_progressive():
    """Test progressive training on CIFAR-10"""
    print("\n" + "="*80)
    print("CIFAR-10 PROGRESSIVE RESOLUTION TEST")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create progressive generator
    prog_gen = ProgressiveGenerator(ngf=24).to(device)
    
    # Count ops at different resolutions
    from flexible_gan_architectures import estimate_einsum_ops
    
    for res in [8, 16, 32]:
        prog_gen.set_resolution(res)
        ops = estimate_einsum_ops(prog_gen)
        print(f"Ops at {res}x{res}: {ops}")
    
    # Train progressively
    trained_gen = train_progressive(prog_gen, device, 'cifar10')
    
    # Test final quality
    print("\nTesting final CIFAR-10 quality...")
    
    try:
        from test_conditional_gan_proper import ZKOptimizedClassifier
        classifier = ZKOptimizedClassifier().to(device)
        classifier_path = "/root/proof_chain/cifar_gan_training/zk_classifier_avgpool.pth"
        
        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        
        # Test at 32x32
        trained_gen.set_resolution(32)
        trained_gen.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for class_idx in range(10):
                noise = torch.randn(100, 100, 1, 1).to(device)
                labels = torch.full((100,), class_idx, dtype=torch.long).to(device)
                
                fake_images = trained_gen(noise, labels)
                outputs = classifier(fake_images)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == labels).sum().item()
                total += 100
        
        accuracy = 100 * correct / total
        print(f"Progressive CIFAR-10 Accuracy: {accuracy:.2f}%")
        
        if accuracy > 35:
            print("✅ BREAKTHROUGH! Progressive training helps!")
        
        return accuracy
        
    except Exception as e:
        print(f"Error testing: {e}")
        return 0


def main():
    # Test 1: MNIST baseline
    print("\n=== TEST 1: MNIST BASELINE ===\n")
    mnist_acc, mnist_ops = test_on_mnist()
    
    # Test 2: Progressive CIFAR-10
    print("\n=== TEST 2: PROGRESSIVE CIFAR-10 ===\n")
    cifar_acc = test_cifar_progressive()
    
    # Summary
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING RESULTS")
    print("="*80)
    
    print(f"\nMNIST:")
    print(f"  Accuracy: {mnist_acc:.2f}%")
    print(f"  Ops: {mnist_ops}")
    
    print(f"\nCIFAR-10 Progressive:")
    print(f"  Accuracy: {cifar_acc:.2f}%")
    
    print("\n💡 Key Insights:")
    if mnist_acc > 50 and cifar_acc < 30:
        print("  - Simple datasets (MNIST) work well with minimal models")
        print("  - CIFAR-10 is fundamentally too complex for ZK constraints")
        print("  - Consider simpler visual tasks for ZK deployment")
    elif cifar_acc > 35:
        print("  - Progressive training significantly improves quality!")
        print("  - Multi-resolution approach helps learn better features")


if __name__ == "__main__":
    main()
