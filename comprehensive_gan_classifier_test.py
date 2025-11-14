#!/usr/bin/env python3
"""
Comprehensive test of ZK-optimized classifier on GAN-generated images
Tests 1000 images per class to evaluate classification accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# GAN Generator architecture (must match trained model)
class Generator(nn.Module):
    def __init__(self, nz=100, nc=3, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)

# ZK-Optimized Classifier architecture
class ZKOptimizedClassifierV1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes, bias=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def generate_images(generator, class_idx, num_images=1000, nz=100, device='cuda'):
    """Generate images for a specific class"""
    generator.eval()
    generated_images = []
    
    batch_size = 50  # Generate in batches for efficiency
    num_batches = num_images // batch_size
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate random noise
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            
            # Generate images
            fake_images = generator(noise)
            generated_images.append(fake_images.cpu())
    
    # Concatenate all batches
    all_images = torch.cat(generated_images, dim=0)
    return all_images[:num_images]  # Ensure exact count

def classify_images(classifier, images, device='cuda'):
    """Classify a batch of images"""
    classifier.eval()
    predictions = []
    
    batch_size = 100
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx].to(device)
            
            outputs = classifier(batch)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return np.array(predictions)

def main():
    print("="*70)
    print("COMPREHENSIVE GAN-CLASSIFIER TEST")
    print("="*70)
    print(f"Testing {len(CIFAR10_CLASSES)} classes with 1000 images each")
    print(f"Total images to test: {len(CIFAR10_CLASSES) * 1000}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load GAN model
    print("\nLoading GAN generator...")
    generator = Generator(nz=100, nc=3, ngf=64).to(device)
    gan_path = 'cifar_gan_training/final_generator.pth'
    
    if os.path.exists(gan_path):
        generator.load_state_dict(torch.load(gan_path, map_location=device))
        print("✓ GAN loaded successfully")
    else:
        print(f"Error: GAN model not found at {gan_path}")
        return
    
    # Load Classifier model
    print("\nLoading ZK-optimized classifier...")
    classifier = ZKOptimizedClassifierV1(num_classes=10).to(device)
    classifier_path = 'cifar_gan_training/zk_optimized_classifier_v1.pth'
    
    if os.path.exists(classifier_path):
        state_dict = torch.load(classifier_path, map_location=device)
        # Handle DataParallel state dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        classifier.load_state_dict(state_dict)
        print("✓ Classifier loaded successfully")
    else:
        print(f"Error: Classifier model not found at {classifier_path}")
        return
    
    # Test each class
    print("\n" + "="*70)
    print("GENERATING AND TESTING IMAGES")
    print("="*70)
    
    all_true_labels = []
    all_predictions = []
    class_accuracies = {}
    
    for class_idx in range(len(CIFAR10_CLASSES)):
        class_name = CIFAR10_CLASSES[class_idx]
        print(f"\nClass {class_idx}: {class_name}")
        print("-" * 40)
        
        # Generate images
        print(f"  Generating 1000 images...")
        generated_images = generate_images(generator, class_idx, num_images=1000, device=device)
        
        # Classify images
        print(f"  Classifying images...")
        predictions = classify_images(classifier, generated_images, device=device)
        
        # Calculate accuracy for this class
        true_labels = np.full(1000, class_idx)
        accuracy = np.mean(predictions == true_labels) * 100
        class_accuracies[class_name] = accuracy
        
        # Store for overall metrics
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)
        
        # Print class results
        print(f"  ✓ Accuracy: {accuracy:.2f}%")
        
        # Show prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"  Prediction distribution:")
        for pred_class, count in zip(unique, counts):
            if count > 50:  # Only show significant predictions
                print(f"    → {CIFAR10_CLASSES[pred_class]}: {count} ({count/10:.1f}%)")
    
    # Calculate overall metrics
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    
    overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels)) * 100
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    print(f"Total Images Tested: {len(all_predictions)}")
    
    # Per-class accuracy summary
    print("\nPer-Class Accuracy Summary:")
    print("-" * 40)
    for class_name, acc in class_accuracies.items():
        bar = '█' * int(acc / 2)  # Simple bar chart
        print(f"{class_name:12s}: {acc:6.2f}% {bar}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Calculate additional metrics
    print("\nConfusion Matrix Analysis:")
    print("-" * 40)
    
    # Find most confused pairs
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    confused_pairs = []
    for i in range(len(CIFAR10_CLASSES)):
        for j in range(len(CIFAR10_CLASSES)):
            if i != j and cm_normalized[i, j] > 0.1:  # More than 10% confusion
                confused_pairs.append((CIFAR10_CLASSES[i], CIFAR10_CLASSES[j], cm_normalized[i, j] * 100))
    
    if confused_pairs:
        print("Most Confused Class Pairs (>10% misclassification):")
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        for true_class, pred_class, conf_rate in confused_pairs[:5]:
            print(f"  {true_class} → {pred_class}: {conf_rate:.1f}%")
    
    # Save results to JSON
    results = {
        "test_date": datetime.now().isoformat(),
        "total_images": len(all_predictions),
        "images_per_class": 1000,
        "overall_accuracy": float(overall_accuracy),
        "class_accuracies": {k: float(v) for k, v in class_accuracies.items()},
        "confusion_matrix": cm.tolist(),
        "device": str(device),
        "gan_model": gan_path,
        "classifier_model": classifier_path
    }
    
    with open("gan_classifier_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to gan_classifier_test_results.json")
    
    # Generate classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=CIFAR10_CLASSES))
    
    return results

if __name__ == "__main__":
    results = main()
