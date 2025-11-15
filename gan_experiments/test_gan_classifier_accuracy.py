#!/usr/bin/env python3
"""
Test classifier accuracy on GAN-generated images vs real CIFAR-10 images
Evaluates how well generated images match their intended classes
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')
sys.path.insert(0, '/root/proof_chain')

def test_classifier_on_gan_images():
    """Test classifier accuracy on generated images"""
    print("="*80)
    print("CLASSIFIER ACCURACY TEST: GAN vs REAL IMAGES")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load the best MLP GAN (exp_026)
    from flexible_gan_architectures import create_generator

    config = {
        "experiment_id": "exp_026",
        "name": "mlp_only",
        "architecture": {
            "model_type": "mlp",
            "latent_dim": 100,
            "embed_dim": 50,
            "hidden_dims": [512, 1024, 2048, 3072],
            "use_batchnorm": True,
            "use_bias": False,
            "activation": "relu"
        }
    }

    generator = create_generator(config).to(device)
    gen_path = "/root/proof_chain/gan_experiments/tier4/models/exp_026_generator.pth"

    if os.path.exists(gen_path):
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        print(f"✓ MLP Generator loaded from {gen_path}")
    else:
        print(f"⚠️ Generator not found at {gen_path}")
        return

    generator.eval()

    # Load ZK-optimized classifier
    try:
        from test_conditional_gan_proper import ZKOptimizedClassifier
        classifier = ZKOptimizedClassifier().to(device)
        classifier_path = "/root/proof_chain/cifar_gan_training/zk_classifier_avgpool.pth"

        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
            print(f"✓ Classifier loaded from {classifier_path}\n")
        else:
            print(f"⚠️ Classifier not found, training new one...")
            # Train a simple classifier if needed
            classifier = train_simple_classifier(device)
    except:
        print("Using simple classifier...")
        classifier = SimpleClassifier().to(device)
        classifier_path = "/root/proof_chain/cifar_gan_training/zk_classifier_avgpool.pth"
        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    classifier.eval()

    # Test 1: Accuracy on Real Images
    print("="*80)
    print("TEST 1: CLASSIFIER ACCURACY ON REAL CIFAR-10 IMAGES")
    print("="*80)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='/root/proof_chain/data',
        train=False,
        download=False,
        transform=transform
    )
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    correct_real = 0
    total_real = 0
    class_correct_real = [0] * 10
    class_total_real = [0] * 10

    with torch.no_grad():
        for images, labels in testloader:
            if total_real >= 1000:  # Test on 1000 images
                break

            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)

            total_real += labels.size(0)
            correct_real += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total_real[label] += 1
                if predicted[i] == labels[i]:
                    class_correct_real[label] += 1

    real_accuracy = 100 * correct_real / total_real
    print(f"Overall accuracy on real images: {real_accuracy:.2f}%")

    print("\nPer-class accuracy on real images:")
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        if class_total_real[i] > 0:
            acc = 100 * class_correct_real[i] / class_total_real[i]
            print(f"  {classes[i]:10s}: {acc:.1f}%")

    # Test 2: Accuracy on Generated Images
    print("\n" + "="*80)
    print("TEST 2: CLASSIFIER ACCURACY ON GAN-GENERATED IMAGES")
    print("="*80)

    samples_per_class = 100
    correct_gan = 0
    total_gan = 0
    class_correct_gan = [0] * 10
    class_total_gan = [0] * 10

    with torch.no_grad():
        for class_idx in range(10):
            # Generate images for this class
            noise = torch.randn(samples_per_class, 100, 1, 1).to(device)
            labels = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)

            fake_images = generator(noise, labels)

            # Classify generated images
            outputs = classifier(fake_images)
            _, predicted = torch.max(outputs, 1)

            # Count correct predictions
            correct = (predicted == labels).sum().item()
            correct_gan += correct
            total_gan += samples_per_class

            class_correct_gan[class_idx] = correct
            class_total_gan[class_idx] = samples_per_class

            acc = 100 * correct / samples_per_class
            print(f"  {classes[class_idx]:10s}: {acc:.1f}% ({correct}/{samples_per_class} correct)")

    gan_accuracy = 100 * correct_gan / total_gan
    print(f"\nOverall accuracy on GAN images: {gan_accuracy:.2f}%")

    # Test 3: Real vs Fake Detection
    print("\n" + "="*80)
    print("TEST 3: REAL vs FAKE DETECTION")
    print("="*80)

    # Train a simple discriminator to distinguish real vs fake
    print("Testing if classifier can distinguish real vs generated...")

    # Get features from classifier's penultimate layer
    def get_features(model, images):
        # Get features before final classification layer
        x = model.features(images) if hasattr(model, 'features') else images
        return x.view(x.size(0), -1)

    # Collect features from real and fake images
    real_features = []
    fake_features = []

    with torch.no_grad():
        # Real images
        for images, _ in testloader:
            if len(real_features) * 100 >= 500:
                break
            images = images.to(device)
            # Get raw classifier outputs as features
            features = classifier(images)
            real_features.append(features.cpu())

        # Fake images
        for _ in range(5):
            noise = torch.randn(100, 100, 1, 1).to(device)
            labels = torch.randint(0, 10, (100,)).to(device)
            fake_images = generator(noise, labels)
            features = classifier(fake_images)
            fake_features.append(features.cpu())

    real_features = torch.cat(real_features)[:500]
    fake_features = torch.cat(fake_features)[:500]

    # Simple statistical test - compare mean activations
    real_mean = real_features.mean(0)
    fake_mean = fake_features.mean(0)

    difference = torch.abs(real_mean - fake_mean).mean().item()
    print(f"Mean feature difference (real vs fake): {difference:.4f}")

    if difference < 0.1:
        print("✓ Generated images have similar features to real images")
    else:
        print("⚠️ Generated images have different feature distributions")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n1. Real Image Classification:")
    print(f"   Accuracy: {real_accuracy:.2f}%")

    print(f"\n2. GAN Image Classification:")
    print(f"   Accuracy: {gan_accuracy:.2f}%")
    print(f"   Quality ratio: {gan_accuracy/real_accuracy:.2f}")

    print(f"\n3. Feature Similarity:")
    print(f"   Mean difference: {difference:.4f}")

    # Determine quality assessment
    quality_score = gan_accuracy / real_accuracy

    if quality_score > 0.8:
        print(f"\n✅ EXCELLENT: GAN produces highly realistic class-specific images")
        print(f"   {quality_score:.1%} of real classifier performance")
    elif quality_score > 0.6:
        print(f"\n✓ GOOD: GAN produces recognizable class-specific images")
        print(f"   {quality_score:.1%} of real classifier performance")
    elif quality_score > 0.4:
        print(f"\n⚠️ MODERATE: GAN images partially match intended classes")
        print(f"   {quality_score:.1%} of real classifier performance")
    else:
        print(f"\n❌ POOR: GAN struggles to produce class-specific features")
        print(f"   {quality_score:.1%} of real classifier performance")

    return {
        'real_accuracy': real_accuracy,
        'gan_accuracy': gan_accuracy,
        'quality_ratio': quality_score,
        'feature_difference': difference,
        'per_class_gan': class_correct_gan
    }


class SimpleClassifier(nn.Module):
    """Simple CNN classifier for CIFAR-10"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    results = test_classifier_on_gan_images()