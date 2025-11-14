#!/usr/bin/env python3
import sys
sys.path.append('cifar_gan_training')

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from train_conditional_gan import Generator
from train_zk_optimized_classifier import ZKOptimizedClassifierV1
import warnings
warnings.filterwarnings('ignore')

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

def generate_and_test_class(generator, classifier, class_idx, num_images=1000, device='cuda'):
    """Generate and test images for a specific class"""
    generator.eval()
    classifier.eval()
    
    batch_size = 50
    num_batches = (num_images + batch_size - 1) // batch_size
    
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_images - i * batch_size)
            
            # Generate noise
            noise = torch.randn(current_batch_size, 100, 1, 1, device=device)
            
            # Create labels for this class
            labels = torch.full((current_batch_size,), class_idx, dtype=torch.long, device=device)
            
            # Generate images
            fake_images = generator(noise, labels)
            
            # Classify images
            outputs = classifier(fake_images)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_confidences)

def main():
    print("="*70)
    print("GAN-CLASSIFIER COMPREHENSIVE TEST")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load models
    print("Loading models...")
    generator = Generator(latent_dim=100, num_classes=10).to(device)
    classifier = ZKOptimizedClassifierV1(num_classes=10).to(device)
    
    # Load weights
    gen_state = torch.load('cifar_gan_training/final_generator.pth', map_location=device, weights_only=False)
    if list(gen_state.keys())[0].startswith('module.'):
        gen_state = {k[7:]: v for k, v in gen_state.items()}
    generator.load_state_dict(gen_state)
    
    cls_state = torch.load('cifar_gan_training/zk_classifier_avgpool.pth', map_location=device, weights_only=False)
    if list(cls_state.keys())[0].startswith('module.'):
        cls_state = {k[7:]: v for k, v in cls_state.items()}
    classifier.load_state_dict(cls_state)
    
    print("✓ Models loaded\n")
    
    # Test each class
    results = {}
    all_true = []
    all_pred = []
    
    print("Testing each class (1000 images per class):")
    print("-"*50)
    
    for class_idx in range(10):
        class_name = CIFAR10_CLASSES[class_idx]
        
        predictions, confidences = generate_and_test_class(
            generator, classifier, class_idx, num_images=1000, device=device
        )
        
        true_labels = np.full(1000, class_idx)
        accuracy = np.mean(predictions == true_labels) * 100
        avg_conf = np.mean(confidences) * 100
        
        results[class_name] = {
            'accuracy': float(accuracy),
            'avg_confidence': float(avg_conf),
            'correct': int(np.sum(predictions == true_labels))
        }
        
        all_true.extend(true_labels)
        all_pred.extend(predictions)
        
        # Display results
        bar = '█' * int(accuracy / 2)
        print(f"{class_name:12s}: {accuracy:6.2f}% {bar}")
    
    # Overall metrics
    overall_accuracy = np.mean(np.array(all_pred) == np.array(all_true)) * 100
    
    print("\n" + "="*70)
    print(f"OVERALL ACCURACY: {overall_accuracy:.2f}%")
    print(f"Total Images Tested: {len(all_pred)}")
    print("="*70)
    
    # Find confused pairs
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_true, all_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print("\nMost Confused Pairs:")
    confused = []
    for i in range(10):
        for j in range(10):
            if i != j and cm_norm[i, j] > 0.15:
                confused.append((CIFAR10_CLASSES[i], CIFAR10_CLASSES[j], cm_norm[i, j] * 100))
    
    confused.sort(key=lambda x: x[2], reverse=True)
    for true_c, pred_c, rate in confused[:5]:
        print(f"  {true_c} → {pred_c}: {rate:.1f}%")
    
    # Save results
    final_results = {
        'test_date': datetime.now().isoformat(),
        'overall_accuracy': float(overall_accuracy),
        'total_images': len(all_pred),
        'class_results': results,
        'confusion_matrix': cm.tolist()
    }
    
    with open('gan_classifier_comprehensive_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n✓ Results saved to gan_classifier_comprehensive_results.json")
    
    return final_results

if __name__ == '__main__':
    main()
