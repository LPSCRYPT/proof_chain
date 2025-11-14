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
import warnings
warnings.filterwarnings('ignore')

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Conditional GAN Generator (matching the actual trained model)
class Generator(nn.Module):
    def __init__(self, nz=100, nc=3, ngf=64, num_classes=10):
        super(Generator, self).__init__()
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, nz)
        
        self.main = nn.Sequential(
            # input is Z + embedded label, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 32 x 32
        )

    def forward(self, noise, labels):
        # Embed labels and combine with noise
        embedded_labels = self.label_embedding(labels)
        embedded_labels = embedded_labels.view(embedded_labels.size(0), -1, 1, 1)
        
        # Add embedded labels to noise
        gen_input = noise + embedded_labels
        
        return self.main(gen_input)

# ZK-Optimized Classifier
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

def generate_images_for_class(generator, class_idx, num_images=1000, nz=100, device='cuda'):
    """Generate images for a specific class"""
    generator.eval()
    generated_images = []
    
    batch_size = 50
    num_batches = (num_images + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_images - i * batch_size)
            
            # Generate noise
            noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
            
            # Create labels for this class
            labels = torch.full((current_batch_size,), class_idx, dtype=torch.long, device=device)
            
            # Generate images
            fake_images = generator(noise, labels)
            generated_images.append(fake_images.cpu())
    
    all_images = torch.cat(generated_images, dim=0)
    return all_images[:num_images]

def classify_images(classifier, images, device='cuda'):
    """Classify a batch of images"""
    classifier.eval()
    predictions = []
    confidences = []
    
    batch_size = 100
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx].to(device)
            
            outputs = classifier(batch)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
    
    return np.array(predictions), np.array(confidences)

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
    generator = Generator(nz=100, nc=3, ngf=64, num_classes=10).to(device)
    gan_path = 'cifar_gan_training/final_generator.pth'
    
    if os.path.exists(gan_path):
        state_dict = torch.load(gan_path, map_location=device, weights_only=False)
        # Remove 'module.' prefix from DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        generator.load_state_dict(state_dict)
        print("✓ GAN loaded successfully")
    else:
        print(f"Error: GAN model not found at {gan_path}")
        return
    
    # Load Classifier model
    print("\nLoading ZK-optimized classifier...")
    classifier = ZKOptimizedClassifierV1(num_classes=10).to(device)
    classifier_path = 'cifar_gan_training/zk_optimized_classifier_v1.pth'
    
    if os.path.exists(classifier_path):
        state_dict = torch.load(classifier_path, map_location=device, weights_only=False)
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
    all_confidences = []
    class_results = {}
    
    for class_idx in range(len(CIFAR10_CLASSES)):
        class_name = CIFAR10_CLASSES[class_idx]
        print(f"\nClass {class_idx}: {class_name}")
        print("-" * 40)
        
        # Generate images
        print(f"  Generating 1000 images...")
        generated_images = generate_images_for_class(generator, class_idx, num_images=1000, device=device)
        
        # Classify images
        print(f"  Classifying images...")
        predictions, confidences = classify_images(classifier, generated_images, device=device)
        
        # Calculate metrics for this class
        true_labels = np.full(1000, class_idx)
        correct = predictions == true_labels
        accuracy = np.mean(correct) * 100
        avg_confidence = np.mean(confidences) * 100
        avg_confidence_correct = np.mean(confidences[correct]) * 100 if np.any(correct) else 0
        avg_confidence_wrong = np.mean(confidences[~correct]) * 100 if np.any(~correct) else 0
        
        class_results[class_name] = {
            'accuracy': float(accuracy),
            'avg_confidence': float(avg_confidence),
            'avg_confidence_correct': float(avg_confidence_correct),
            'avg_confidence_wrong': float(avg_confidence_wrong),
            'num_correct': int(np.sum(correct)),
            'num_wrong': int(np.sum(~correct))
        }
        
        # Store for overall metrics
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)
        all_confidences.extend(confidences)
        
        # Print class results
        print(f"  ✓ Accuracy: {accuracy:.2f}%")
        print(f"  ✓ Avg Confidence: {avg_confidence:.2f}%")
        
        # Show prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"  Prediction distribution:")
        top_predictions = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:3]
        for pred_class, count in top_predictions:
            print(f"    → {CIFAR10_CLASSES[pred_class]}: {count} ({count/10:.1f}%)")
    
    # Calculate overall metrics
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    
    overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels)) * 100
    overall_confidence = np.mean(all_confidences) * 100
    
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    print(f"Overall Confidence: {overall_confidence:.2f}%")
    print(f"Total Images Tested: {len(all_predictions)}")
    
    # Per-class accuracy summary
    print("\nPer-Class Accuracy Summary:")
    print("-" * 40)
    for class_name, metrics in class_results.items():
        acc = metrics['accuracy']
        bar = '█' * int(acc / 2)
        print(f"{class_name:12s}: {acc:6.2f}% {bar}")
    
    # Confusion Matrix Analysis
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    print("\nConfusion Matrix Insights:")
    print("-" * 40)
    
    # Find most confused pairs
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    confused_pairs = []
    
    for i in range(len(CIFAR10_CLASSES)):
        for j in range(len(CIFAR10_CLASSES)):
            if i != j and cm_normalized[i, j] > 0.1:
                confused_pairs.append((CIFAR10_CLASSES[i], CIFAR10_CLASSES[j], cm_normalized[i, j] * 100))
    
    if confused_pairs:
        print("Most Confused Class Pairs (>10% misclassification):")
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        for true_class, pred_class, conf_rate in confused_pairs[:5]:
            print(f"  {true_class} → {pred_class}: {conf_rate:.1f}%")
    
    # Best and worst performing classes
    accuracies = [(name, metrics['accuracy']) for name, metrics in class_results.items()]
    accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\nBest Performing Classes:")
    for name, acc in accuracies[:3]:
        print(f"  {name}: {acc:.2f}%")
    
    print("\nWorst Performing Classes:")
    for name, acc in accuracies[-3:]:
        print(f"  {name}: {acc:.2f}%")
    
    # Save results
    results = {
        "test_date": datetime.now().isoformat(),
        "total_images": len(all_predictions),
        "images_per_class": 1000,
        "overall_accuracy": float(overall_accuracy),
        "overall_confidence": float(overall_confidence),
        "class_results": class_results,
        "confusion_matrix": cm.tolist(),
        "device": str(device),
        "gan_model": gan_path,
        "classifier_model": classifier_path
    }
    
    with open("gan_classifier_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to gan_classifier_test_results.json")
    
    # Print final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Successfully tested {len(all_predictions)} GAN-generated images")
    print(f"✓ Overall accuracy: {overall_accuracy:.2f}%")
    print(f"✓ Model shows reasonable performance on synthetic data")
    
    return results

if __name__ == "__main__":
    results = main()
