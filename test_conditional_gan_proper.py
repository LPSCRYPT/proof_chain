#!/usr/bin/env python3
"""
Proper test for Conditional GAN with class conditioning
Tests the actual deployed conditional GAN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

class ProperConditionalGAN(nn.Module):
    """Conditional GAN with proper label embedding"""
    def __init__(self, latent_dim=100, num_classes=10, embed_dim=50, ngf=64):
        super(ProperConditionalGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Label embedding layer - CRITICAL FOR CONDITIONING
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Combined input: latent_dim + embed_dim
        input_dim = latent_dim + embed_dim

        # Generator layers with proper channel progression
        self.main = nn.Sequential(
            # Input: (latent_dim + embed_dim) → 4×4 feature map
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 4×4 → 8×8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Final layer: 32×32 RGB image
            nn.ConvTranspose2d(ngf, 3, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        Forward pass WITH CLASS CONDITIONING
        Args:
            noise: Random noise tensor [batch_size, latent_dim, 1, 1]
            labels: Class labels tensor [batch_size]
        Returns:
            Generated images [batch_size, 3, 32, 32]
        """
        # Embed class labels
        label_embed = self.label_embedding(labels)
        label_embed = label_embed.view(label_embed.size(0), self.embed_dim, 1, 1)

        # Concatenate noise and embedded labels
        gen_input = torch.cat([noise, label_embed], dim=1)

        # Generate image
        return self.main(gen_input)

class ZKOptimizedClassifier(nn.Module):
    """ZK-optimized classifier with AvgPool"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),  # ZK-friendly pooling

            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def test_mode_collapse(generator, device='cuda'):
    """Test if generator has mode collapse"""
    generator.eval()

    diversity_scores = []
    with torch.no_grad():
        for class_idx in range(10):
            # Generate 10 images with different noise but same class
            images = []
            for _ in range(10):
                noise = torch.randn(1, 100, 1, 1).to(device)
                labels = torch.tensor([class_idx], dtype=torch.long).to(device)
                img = generator(noise, labels)
                images.append(img)

            # Calculate pixel-wise standard deviation
            images_tensor = torch.cat(images, dim=0)
            std_per_pixel = torch.std(images_tensor, dim=0)
            mean_std = std_per_pixel.mean().item()
            diversity_scores.append(mean_std)

    return diversity_scores

def test_class_conditioning(generator, device='cuda'):
    """Test if generator properly responds to class labels"""
    generator.eval()

    # Use same noise for all classes
    fixed_noise = torch.randn(1, 100, 1, 1).to(device)

    class_differences = []
    images_by_class = []

    with torch.no_grad():
        for class_idx in range(10):
            # Expand noise to batch size of 1
            noise = fixed_noise.clone()
            labels = torch.tensor([class_idx], dtype=torch.long).to(device)
            img = generator(noise, labels)
            images_by_class.append(img)

    # Calculate differences between classes
    for i in range(10):
        for j in range(i+1, 10):
            diff = torch.mean(torch.abs(images_by_class[i] - images_by_class[j]))
            class_differences.append(diff.item())

    return np.mean(class_differences), np.min(class_differences), np.max(class_differences)

def test_classifier_accuracy(generator, classifier, num_samples=100, device='cuda'):
    """Test classifier accuracy on generated images"""
    generator.eval()
    classifier.eval()

    results_by_class = {}

    with torch.no_grad():
        for class_idx in range(10):
            correct = 0
            predictions = []

            for _ in range(num_samples):
                # Generate image
                noise = torch.randn(1, 100, 1, 1).to(device)
                labels = torch.tensor([class_idx], dtype=torch.long).to(device)
                fake_img = generator(noise, labels)

                # Classify
                output = classifier(fake_img)
                pred = torch.argmax(output, dim=1).item()
                predictions.append(pred)

                if pred == class_idx:
                    correct += 1

            accuracy = correct / num_samples * 100
            pred_distribution = Counter(predictions)

            results_by_class[CIFAR10_CLASSES[class_idx]] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': num_samples,
                'predictions': dict(pred_distribution)
            }

    return results_by_class

def calculate_entropy(predictions):
    """Calculate entropy of prediction distribution"""
    counts = np.bincount(predictions, minlength=10)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zero probabilities
    entropy = -np.sum(probs * np.log(probs))
    return entropy

def main():
    print("="*70)
    print("PROPER CONDITIONAL GAN TEST")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Initialize models
    print("Initializing models...")
    generator = ProperConditionalGAN(latent_dim=100, num_classes=10, embed_dim=50, ngf=64).to(device)
    classifier = ZKOptimizedClassifier(num_classes=10).to(device)

    # Try to load existing weights if available
    import os
    gan_path = 'cifar_gan_training/final_generator.pth'
    classifier_path = 'cifar_gan_training/zk_classifier_avgpool.pth'

    if os.path.exists(gan_path):
        print(f"Loading GAN from {gan_path}")
        state = torch.load(gan_path, map_location=device, weights_only=False)
        if list(state.keys())[0].startswith('module.'):
            state = {k[7:]: v for k, v in state.items()}
        try:
            generator.load_state_dict(state)
            print("✓ GAN loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load GAN: {e}")
            print("  Using random initialization")
    else:
        print(f"✗ GAN weights not found at {gan_path}")
        print("  Using random initialization")

    if os.path.exists(classifier_path):
        print(f"Loading classifier from {classifier_path}")
        state = torch.load(classifier_path, map_location=device, weights_only=False)
        if list(state.keys())[0].startswith('module.'):
            state = {k[7:]: v for k, v in state.items()}
        classifier.load_state_dict(state)
        print("✓ Classifier loaded successfully")
    else:
        print(f"✗ Classifier weights not found at {classifier_path}")

    print("\n" + "="*70)
    print("TEST 1: MODE COLLAPSE DETECTION")
    print("="*70)

    diversity_scores = test_mode_collapse(generator, device)
    mean_diversity = np.mean(diversity_scores)

    print("Intra-class diversity (std dev of pixel values):")
    for i, score in enumerate(diversity_scores):
        status = "✓" if score > 0.05 else "✗ COLLAPSED"
        print(f"  {CIFAR10_CLASSES[i]:12s}: {score:.4f} {status}")

    print(f"\nMean diversity: {mean_diversity:.4f}")
    if mean_diversity < 0.05:
        print("⚠️ WARNING: Possible mode collapse detected!")
    else:
        print("✓ Generator shows healthy diversity")

    print("\n" + "="*70)
    print("TEST 2: CLASS CONDITIONING")
    print("="*70)

    mean_diff, min_diff, max_diff = test_class_conditioning(generator, device)
    print(f"Using same noise, different classes:")
    print(f"  Mean difference between classes: {mean_diff:.4f}")
    print(f"  Min difference: {min_diff:.4f}")
    print(f"  Max difference: {max_diff:.4f}")

    if mean_diff < 0.1:
        print("⚠️ WARNING: Classes are too similar - conditioning not working!")
    else:
        print("✓ Generator properly responds to class labels")

    print("\n" + "="*70)
    print("TEST 3: CLASSIFIER ACCURACY")
    print("="*70)

    results = test_classifier_accuracy(generator, classifier, num_samples=100, device=device)

    accuracies = []
    all_predictions = []

    for class_name, metrics in results.items():
        acc = metrics['accuracy']
        accuracies.append(acc)

        # Collect all predictions
        for pred_class, count in metrics['predictions'].items():
            all_predictions.extend([pred_class] * count)

        # Display results
        status = "✓" if acc > 50 else "✗"
        bar = '█' * int(acc / 2)
        print(f"{status} {class_name:12s}: {acc:6.2f}% {bar}")

    # Calculate overall metrics
    mean_accuracy = np.mean(accuracies)
    entropy = calculate_entropy(all_predictions)

    print(f"\nOverall Metrics:")
    print(f"  Mean accuracy: {mean_accuracy:.2f}%")
    print(f"  Prediction entropy: {entropy:.2f} (max: 2.30 for uniform)")

    # Check for classifier bias
    pred_counts = Counter(all_predictions)
    most_common = pred_counts.most_common(1)[0]
    if most_common[1] > 300:  # More than 30% of all predictions
        print(f"⚠️ WARNING: Classifier biased towards class {most_common[0]} ({most_common[1]/10}%)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Overall assessment
    issues = []
    if mean_diversity < 0.05:
        issues.append("Mode collapse detected")
    if mean_diff < 0.1:
        issues.append("Class conditioning not working")
    if mean_accuracy < 30:
        issues.append("Very poor classifier accuracy")
    elif mean_accuracy < 50:
        issues.append("Poor classifier accuracy")
    if entropy < 1.5:
        issues.append("Classifier shows strong bias")

    if not issues:
        print("✓ Conditional GAN is working properly!")
        print(f"  - Diversity: {mean_diversity:.4f}")
        print(f"  - Class separation: {mean_diff:.4f}")
        print(f"  - Accuracy: {mean_accuracy:.2f}%")
    else:
        print("✗ Critical issues detected:")
        for issue in issues:
            print(f"  - {issue}")

    # Save results
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'mean_diversity': float(mean_diversity),
        'class_conditioning': {
            'mean_diff': float(mean_diff),
            'min_diff': float(min_diff),
            'max_diff': float(max_diff)
        },
        'classifier_accuracy': {
            'mean': float(mean_accuracy),
            'by_class': results,
            'entropy': float(entropy)
        },
        'issues': issues
    }

    with open('conditional_gan_test_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✓ Results saved to conditional_gan_test_results.json")

    return mean_accuracy, mean_diversity, mean_diff

if __name__ == '__main__':
    main()