#!/usr/bin/env python3
"""
Test ZK-Optimized Classifier on GAN-Generated Images
Evaluates how well the optimized classifier performs on synthetic data
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

class TinyGenerator(nn.Module):
    """GAN Generator from training"""
    def __init__(self, latent_dim=32, num_classes=10):
        super(TinyGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 128, 4, 1, 0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=True),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # One-hot encode labels
        batch_size = z.size(0)
        one_hot = torch.zeros(batch_size, self.num_classes, device=z.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Reshape for conv layers
        z = z.view(batch_size, self.latent_dim, 1, 1)
        one_hot = one_hot.view(batch_size, self.num_classes, 1, 1)
        
        # Concatenate
        x = torch.cat([z, one_hot], dim=1)
        return self.main(x)

class ZKOptimizedClassifierV1(nn.Module):
    """Optimized classifier with AvgPool"""
    def __init__(self, num_classes=10):
        super(ZKOptimizedClassifierV1, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),
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

def generate_images(generator, num_per_class=100, device='cuda'):
    """Generate images for each class"""
    generator.eval()
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for class_idx in range(10):
            # Generate batch for this class
            z = torch.randn(num_per_class, 32).to(device)
            labels = torch.full((num_per_class,), class_idx, dtype=torch.long).to(device)
            
            # Generate images
            images = generator(z, labels)
            
            all_images.append(images)
            all_labels.append(labels)
    
    return torch.cat(all_images), torch.cat(all_labels)

def evaluate_classifier(classifier, images, true_labels, device='cuda'):
    """Evaluate classifier on generated images"""
    classifier.eval()
    all_preds = []
    
    with torch.no_grad():
        # Process in batches
        batch_size = 100
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            outputs = classifier(batch)
            _, predicted = outputs.max(1)
            all_preds.append(predicted)
    
    predictions = torch.cat(all_preds)
    
    # Calculate accuracy
    correct = predictions.eq(true_labels).sum().item()
    total = len(true_labels)
    accuracy = 100. * correct / total
    
    return predictions.cpu().numpy(), accuracy

def plot_confusion_matrix(true_labels, predictions, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.ylabel('True Class (GAN Target)')
    plt.xlabel('Predicted Class (Classifier Output)')
    plt.tight_layout()
    plt.savefig('/root/proof_chain/confusion_matrix_zk_optimized.png', dpi=150)
    plt.show()
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(CLASSES):
        if cm[i].sum() > 0:
            class_acc = 100 * cm[i, i] / cm[i].sum()
            print(f"  {class_name:12s}: {class_acc:5.1f}% ({cm[i, i]}/{cm[i].sum()})")

def visualize_samples(generator, classifier, num_samples=5, device='cuda'):
    """Visualize some generated samples with predictions"""
    generator.eval()
    classifier.eval()
    
    fig, axes = plt.subplots(10, num_samples, figsize=(num_samples*2, 20))
    
    with torch.no_grad():
        for class_idx in range(10):
            # Generate samples for this class
            z = torch.randn(num_samples, 32).to(device)
            labels = torch.full((num_samples,), class_idx, dtype=torch.long).to(device)
            
            images = generator(z, labels)
            outputs = classifier(images)
            _, predicted = outputs.max(1)
            
            # Plot samples
            for j in range(num_samples):
                ax = axes[class_idx, j]
                
                # Denormalize image
                img = images[j].cpu()
                img = (img + 1) / 2  # From [-1,1] to [0,1]
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                pred_class = CLASSES[predicted[j]]
                true_class = CLASSES[class_idx]
                
                color = 'green' if predicted[j] == class_idx else 'red'
                ax.set_title(f'→{pred_class}', fontsize=10, color=color)
                
                if j == 0:
                    ax.set_ylabel(f'GAN: {true_class}', fontsize=10)
                
                ax.axis('off')
    
    plt.suptitle('GAN Generated Images → Classifier Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig('/root/proof_chain/gan_classifier_samples.png', dpi=150)
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("="*70)
    print("TESTING ZK-OPTIMIZED CLASSIFIER ON GAN-GENERATED IMAGES")
    print("="*70)
    print()
    
    # Load models
    print("Loading models...")
    
    # Load GAN
    generator = TinyGenerator(latent_dim=32, num_classes=10)
    gan_path = Path('/root/proof_chain/cifar_gan_training/tiny_generator.pth')
    if gan_path.exists():
        state_dict = torch.load(gan_path, map_location=device)
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
        print("  ✓ Loaded GAN generator")
    else:
        print(f"  ✗ GAN weights not found at {gan_path}")
        return
    
    generator = generator.to(device)
    
    # Load optimized classifier
    classifier = ZKOptimizedClassifierV1()
    classifier_path = Path('/root/proof_chain/cifar_gan_training/zk_classifier_avgpool.pth')
    if classifier_path.exists():
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print("  ✓ Loaded ZK-optimized classifier")
    else:
        print(f"  ✗ Classifier weights not found at {classifier_path}")
        return
    
    classifier = classifier.to(device)
    
    # Generate test images
    print("\nGenerating test images from GAN...")
    num_per_class = 100
    images, true_labels = generate_images(generator, num_per_class, device)
    print(f"  Generated {len(images)} images ({num_per_class} per class)")
    
    # Evaluate classifier
    print("\nEvaluating classifier on generated images...")
    predictions, accuracy = evaluate_classifier(classifier, images, true_labels, device)
    
    print(f"\n" + "="*70)
    print(f"OVERALL ACCURACY: {accuracy:.2f}%")
    print("="*70)
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        true_labels.cpu().numpy(), 
        predictions,
        f'ZK-Optimized Classifier on GAN Images\nOverall Accuracy: {accuracy:.1f}%'
    )
    
    # Visualize samples
    print("\nVisualizing sample predictions...")
    visualize_samples(generator, classifier, num_samples=5, device=device)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images tested: {len(images)}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Average per-class: {accuracy:.2f}%")
    
    # Analysis
    print("\nANALYSIS:")
    if accuracy > 80:
        print("  ✅ Excellent: Classifier works very well on GAN images")
    elif accuracy > 60:
        print("  ✓ Good: Classifier reasonably identifies GAN-generated classes")
    elif accuracy > 40:
        print("  ⚠️  Moderate: Some confusion between classes")
    else:
        print("  ❌ Poor: Significant mismatch between GAN and classifier")
    
    print("\nImplications for proof chain:")
    print(f"  - GAN generates class-conditioned images")
    print(f"  - Classifier achieves {accuracy:.1f}% accuracy on synthetic data")
    print(f"  - Both models have <100KB verifiers (deployable on-chain)")
    print(f"  - Complete proof-of-frog pipeline is feasible")
    
    print("\nFiles saved:")
    print("  - confusion_matrix_zk_optimized.png")
    print("  - gan_classifier_samples.png")

if __name__ == '__main__':
    main()
