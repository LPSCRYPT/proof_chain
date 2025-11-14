#!/usr/bin/env python3
"""
Comprehensive evaluation metrics for GAN quality assessment
Includes FID, IS, diversity metrics, and visual inspection tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import models, transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class InceptionV3Features(nn.Module):
    """InceptionV3 feature extractor for FID and IS calculation"""
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        # For FID: features from last average pooling
        self.feature_extractor = nn.Sequential(*list(inception.children())[:-1])
        # For IS: full model with logits
        self.classifier = inception
        self.classifier.eval()
        self.feature_extractor.eval()

    @torch.no_grad()
    def get_features(self, x):
        """Extract features for FID calculation"""
        # Resize to 299x299 for Inception
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features

    @torch.no_grad()
    def get_predictions(self, x):
        """Get predictions for IS calculation"""
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.classifier(x)

def calculate_fid(real_features, fake_features):
    """
    Calculate Fréchet Inception Distance
    Lower is better, typical good scores for CIFAR-10: 10-30
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate FID
    diff = mu_real - mu_fake

    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.dot(diff, diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)

def calculate_inception_score(predictions, num_splits=10):
    """
    Calculate Inception Score
    Higher is better, typical good scores for CIFAR-10: 6-9
    """
    # Convert to probabilities
    preds = F.softmax(predictions, dim=1).cpu().numpy()

    # Split for stability
    scores = []
    for i in range(num_splits):
        part = preds[i::num_splits]
        py = np.mean(part, axis=0)

        # Calculate KL divergence for each sample
        kl_divs = []
        for px in part:
            kl_div = entropy(px, py)
            kl_divs.append(kl_div)

        scores.append(np.exp(np.mean(kl_divs)))

    return float(np.mean(scores)), float(np.std(scores))

def calculate_diversity_score(images):
    """
    Calculate diversity within a batch of images
    Returns pixel-wise standard deviation
    """
    # Calculate std across batch dimension
    std = torch.std(images, dim=0)
    return float(std.mean())

def calculate_mode_entropy(class_predictions):
    """
    Calculate entropy of predicted class distribution
    Max entropy = log(10) ≈ 2.30 for 10 classes
    """
    counts = np.bincount(class_predictions, minlength=10)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    if len(probs) <= 1:
        return 0.0

    return float(-np.sum(probs * np.log(probs)))

def create_visual_grid(images, labels=None, title="Generated Images", save_path=None):
    """
    Create a visual grid of images for inspection
    Args:
        images: Tensor of shape [N, 3, 32, 32]
        labels: Optional class labels
        title: Title for the figure
        save_path: Path to save the figure
    """
    # Denormalize images from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    # Create grid
    n_images = min(100, len(images))
    grid_size = int(np.ceil(np.sqrt(n_images)))

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(title, fontsize=16)

    for i in range(n_images):
        ax = plt.subplot(grid_size, grid_size, i + 1)

        # Convert to numpy and transpose for matplotlib
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
        ax.axis('off')

        if labels is not None and i < len(labels):
            ax.set_title(f"Class {labels[i]}", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Visual grid saved to {save_path}")

    return fig

def create_per_class_grid(generator, device='cuda', save_path=None):
    """
    Generate and visualize samples from each class
    """
    generator.eval()

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Generated Samples Per Class", fontsize=16)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    with torch.no_grad():
        for class_idx in range(10):
            # Generate 10 samples for this class
            noise = torch.randn(10, 100, 1, 1).to(device)
            labels = torch.full((10,), class_idx, dtype=torch.long).to(device)

            fake_images = generator(noise, labels)
            fake_images = (fake_images + 1) / 2  # Denormalize

            for sample_idx in range(10):
                ax = plt.subplot(10, 10, class_idx * 10 + sample_idx + 1)
                img = fake_images[sample_idx].cpu().numpy().transpose(1, 2, 0)
                ax.imshow(img)
                ax.axis('off')

                if sample_idx == 0:
                    ax.set_ylabel(class_names[class_idx], rotation=0, labelpad=40, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Per-class grid saved to {save_path}")

    return fig

class GANEvaluator:
    """Complete evaluation suite for conditional GANs"""

    def __init__(self, device='cuda'):
        self.device = device
        self.inception_model = None

    def _ensure_inception(self):
        """Lazy load Inception model"""
        if self.inception_model is None:
            print("Loading InceptionV3 for metrics...")
            self.inception_model = InceptionV3Features().to(self.device)
            self.inception_model.eval()

    @torch.no_grad()
    def evaluate_generator(self, generator, num_samples=5000, batch_size=50):
        """
        Comprehensive evaluation of generator quality
        Returns dictionary with all metrics
        """
        self._ensure_inception()
        generator.eval()

        print("Evaluating generator quality...")

        # Storage for metrics
        all_features = []
        all_predictions = []
        all_images = []
        diversity_scores = []
        class_predictions = []

        # Generate samples from each class
        samples_per_class = num_samples // 10

        for class_idx in range(10):
            class_features = []
            class_preds = []
            class_images = []

            for batch_start in range(0, samples_per_class, batch_size):
                batch_end = min(batch_start + batch_size, samples_per_class)
                batch_size_actual = batch_end - batch_start

                # Generate images
                noise = torch.randn(batch_size_actual, 100, 1, 1).to(self.device)
                labels = torch.full((batch_size_actual,), class_idx, dtype=torch.long).to(self.device)

                fake_images = generator(noise, labels)

                # Extract features and predictions
                features = self.inception_model.get_features(fake_images)
                predictions = self.inception_model.get_predictions(fake_images)

                class_features.append(features.cpu())
                class_preds.append(predictions.cpu())
                class_images.append(fake_images.cpu())

                # Predict class for mode entropy
                pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                class_predictions.extend(pred_classes)

            # Calculate intra-class diversity
            class_images_tensor = torch.cat(class_images, dim=0)
            diversity = calculate_diversity_score(class_images_tensor)
            diversity_scores.append(diversity)

            # Store for overall metrics
            all_features.append(torch.cat(class_features, dim=0))
            all_predictions.append(torch.cat(class_preds, dim=0))
            all_images.append(class_images_tensor[:10])  # Keep 10 per class for visualization

        # Combine all features and predictions
        all_features = torch.cat(all_features, dim=0).numpy()
        all_predictions = torch.cat(all_predictions, dim=0)
        sample_images = torch.cat(all_images, dim=0)

        # Calculate metrics
        print("Calculating Inception Score...")
        is_mean, is_std = calculate_inception_score(all_predictions)

        print("Calculating mode entropy...")
        mode_entropy = calculate_mode_entropy(class_predictions)

        # Note: FID requires real image features (not calculated here without real data)

        # Compile results
        results = {
            'inception_score': {
                'mean': is_mean,
                'std': is_std
            },
            'diversity': {
                'mean': float(np.mean(diversity_scores)),
                'per_class': diversity_scores
            },
            'mode_entropy': mode_entropy,
            'max_entropy': 2.3026,  # log(10)
            'num_samples': num_samples,
            'sample_images': sample_images
        }

        # Quality assessment
        quality_assessment = []
        if is_mean > 7.0:
            quality_assessment.append("Excellent IS (>7.0)")
        elif is_mean > 5.0:
            quality_assessment.append("Good IS (5.0-7.0)")
        else:
            quality_assessment.append("Poor IS (<5.0)")

        if mode_entropy > 2.0:
            quality_assessment.append("Good class diversity")
        elif mode_entropy > 1.5:
            quality_assessment.append("Moderate class diversity")
        else:
            quality_assessment.append("Poor class diversity - possible mode collapse")

        if np.mean(diversity_scores) > 0.1:
            quality_assessment.append("Good intra-class diversity")
        elif np.mean(diversity_scores) > 0.05:
            quality_assessment.append("Moderate intra-class diversity")
        else:
            quality_assessment.append("Poor intra-class diversity - possible mode collapse")

        results['quality_assessment'] = quality_assessment

        return results

    def compare_with_baseline(self, generator, baseline_is=7.0, baseline_fid=30.0):
        """
        Compare generator metrics with baseline/target values
        """
        results = self.evaluate_generator(generator)

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        # Inception Score
        is_mean = results['inception_score']['mean']
        is_std = results['inception_score']['std']
        print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
        print(f"  Target: {baseline_is:.2f}")
        print(f"  Status: {'✓ PASS' if is_mean >= baseline_is else '✗ FAIL'}")

        # Diversity
        mean_diversity = results['diversity']['mean']
        print(f"\nDiversity Score: {mean_diversity:.4f}")
        print(f"  Status: {'✓ Good' if mean_diversity > 0.1 else '⚠️ Low'}")

        # Mode Entropy
        mode_entropy = results['mode_entropy']
        max_entropy = results['max_entropy']
        entropy_ratio = mode_entropy / max_entropy * 100
        print(f"\nMode Entropy: {mode_entropy:.2f} / {max_entropy:.2f} ({entropy_ratio:.1f}%)")
        print(f"  Status: {'✓ Good' if entropy_ratio > 80 else '⚠️ Mode imbalance'}")

        # Per-class diversity
        print("\nPer-Class Diversity:")
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        for i, (name, score) in enumerate(zip(class_names, results['diversity']['per_class'])):
            status = "✓" if score > 0.05 else "✗"
            print(f"  {status} {name:12s}: {score:.4f}")

        # Overall assessment
        print("\n" + "="*60)
        print("QUALITY ASSESSMENT")
        print("="*60)
        for assessment in results['quality_assessment']:
            print(f"• {assessment}")

        return results

def count_einsum_operations(model):
    """
    Estimate einsum operations for EZKL circuit compilation
    This is a simplified estimation - actual count requires EZKL compilation
    """
    einsum_count = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            # Each conv operation contributes einsum ops
            einsum_count += 1
        elif isinstance(module, nn.Linear):
            # Linear layers also contribute
            einsum_count += 1
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm adds a small number of ops
            einsum_count += 0.5
        elif isinstance(module, nn.MaxPool2d):
            # MaxPool is very expensive!
            einsum_count += 500  # Approximate penalty
        elif isinstance(module, nn.AvgPool2d):
            # AvgPool is much cheaper
            einsum_count += 1

    return int(einsum_count)

def test_circuit_compatibility(model, target_ops=100):
    """
    Test if model is compatible with ZK circuit constraints
    """
    estimated_ops = count_einsum_operations(model)

    print(f"Estimated einsum operations: {estimated_ops}")
    print(f"Target maximum: {target_ops}")

    if estimated_ops <= target_ops:
        print("✓ Model is ZK-circuit compatible")
        return True
    else:
        print(f"✗ Model exceeds target by {estimated_ops - target_ops} operations")
        print("  Suggestions:")
        if any(isinstance(m, nn.MaxPool2d) for m in model.modules()):
            print("  - Replace MaxPool2d with AvgPool2d")
        print("  - Reduce number of layers")
        print("  - Reduce channel dimensions")
        return False

if __name__ == '__main__':
    # Example usage
    print("GAN Evaluation Metrics Module")
    print("="*60)
    print("Available functions:")
    print("  - calculate_fid(): Fréchet Inception Distance")
    print("  - calculate_inception_score(): Inception Score")
    print("  - calculate_diversity_score(): Intra-class diversity")
    print("  - calculate_mode_entropy(): Class distribution entropy")
    print("  - create_visual_grid(): Generate visual inspection grid")
    print("  - create_per_class_grid(): Visualize all classes")
    print("  - count_einsum_operations(): Estimate ZK circuit complexity")
    print("  - GANEvaluator: Complete evaluation suite")
    print("="*60)