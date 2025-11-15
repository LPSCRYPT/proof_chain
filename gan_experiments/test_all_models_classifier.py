#!/usr/bin/env python3
"""
Test classifier accuracy on all trained GAN models
Compare class conditioning across different architectures
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import sys

sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')
sys.path.insert(0, '/root/proof_chain')

def test_model_classifier_accuracy(model_path, config, device='cuda'):
    """Test a single model's classifier accuracy"""

    from flexible_gan_architectures import create_generator

    # Load generator
    generator = create_generator(config).to(device)

    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded: {config['experiment_id']}")
    else:
        print(f"⚠️ Model not found: {model_path}")
        return None

    generator.eval()

    # Load classifier
    try:
        from test_conditional_gan_proper import ZKOptimizedClassifier
        classifier = ZKOptimizedClassifier().to(device)
        classifier_path = "/root/proof_chain/cifar_gan_training/zk_classifier_avgpool.pth"

        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        else:
            print(f"⚠️ Classifier not found")
            return None
    except:
        print("⚠️ Could not load classifier")
        return None

    classifier.eval()

    # Test on generated images
    samples_per_class = 100
    correct_gan = 0
    total_gan = 0
    class_correct_gan = [0] * 10

    with torch.no_grad():
        for class_idx in range(10):
            # Generate images for this class
            noise = torch.randn(samples_per_class, config['architecture']['latent_dim'], 1, 1).to(device)
            labels = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)

            try:
                fake_images = generator(noise, labels)

                # Classify generated images
                outputs = classifier(fake_images)
                _, predicted = torch.max(outputs, 1)

                # Count correct predictions
                correct = (predicted == labels).sum().item()
                correct_gan += correct
                total_gan += samples_per_class

                class_correct_gan[class_idx] = correct

            except Exception as e:
                print(f"  Error generating class {class_idx}: {e}")
                continue

    if total_gan > 0:
        accuracy = 100 * correct_gan / total_gan
        return {
            'experiment_id': config['experiment_id'],
            'name': config['name'],
            'accuracy': accuracy,
            'correct': correct_gan,
            'total': total_gan,
            'per_class': class_correct_gan
        }
    return None

def main():
    """Test all trained models"""
    print("="*80)
    print("CLASSIFIER ACCURACY TEST: ALL TRAINED MODELS")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # First, test real image accuracy as baseline
    print("Testing Real Image Accuracy...")
    print("-"*80)

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

    try:
        from test_conditional_gan_proper import ZKOptimizedClassifier
        classifier = ZKOptimizedClassifier().to(device)
        classifier_path = "/root/proof_chain/cifar_gan_training/zk_classifier_avgpool.pth"

        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))

        classifier.eval()

        correct_real = 0
        total_real = 0

        with torch.no_grad():
            for images, labels in testloader:
                if total_real >= 1000:  # Test on 1000 images
                    break

                images, labels = images.to(device), labels.to(device)
                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)

                total_real += labels.size(0)
                correct_real += (predicted == labels).sum().item()

        real_accuracy = 100 * correct_real / total_real
        print(f"Real Image Accuracy: {real_accuracy:.2f}%\n")

    except Exception as e:
        print(f"Could not test real images: {e}")
        real_accuracy = 0

    # Test all trained models
    all_results = []

    for tier_num in range(1, 5):
        print(f"\nTesting Tier {tier_num} Models...")
        print("-"*80)

        # Load tier configs
        config_file = f"/root/proof_chain/gan_experiments/scripts/tier{tier_num}_configs.json"
        if not os.path.exists(config_file):
            continue

        with open(config_file, 'r') as f:
            configs = json.load(f)

        # Test each model in tier
        for config in configs:
            model_path = f"/root/proof_chain/gan_experiments/tier{tier_num}/models/{config['experiment_id']}_generator.pth"

            if os.path.exists(model_path):
                result = test_model_classifier_accuracy(model_path, config, device)
                if result:
                    # Add model type and estimated ops
                    result['model_type'] = config['architecture'].get('model_type', 'conv')
                    result['expected_ops'] = config['expected_ops']
                    result['tier'] = tier_num
                    all_results.append(result)
                    print(f"  {config['experiment_id']}: {result['accuracy']:.2f}% ({config['name']})")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: CIRCUIT EFFICIENCY vs CLASS CONDITIONING")
    print("="*80)

    if all_results:
        # Sort by accuracy
        all_results.sort(key=lambda x: x['accuracy'], reverse=True)

        print("\nTop 3 Models by Class Conditioning Accuracy:")
        print("-"*80)
        for i, result in enumerate(all_results[:3]):
            print(f"{i+1}. {result['experiment_id']}: {result['accuracy']:.2f}%")
            print(f"   Name: {result['name']}")
            print(f"   Model Type: {result['model_type']}")
            print(f"   Expected Ops: {result['expected_ops']}")
            if real_accuracy > 0:
                print(f"   Quality Ratio: {result['accuracy']/real_accuracy:.1%} of real")

        # Compare MLP vs Conv
        mlp_models = [r for r in all_results if r['model_type'] == 'mlp']
        conv_models = [r for r in all_results if r['model_type'] != 'mlp']

        if mlp_models:
            mlp_avg = np.mean([r['accuracy'] for r in mlp_models])
            mlp_ops_avg = np.mean([r['expected_ops'] for r in mlp_models])
            print(f"\nMLP Models:")
            print(f"  Count: {len(mlp_models)}")
            print(f"  Avg Accuracy: {mlp_avg:.2f}%")
            print(f"  Avg Ops: {mlp_ops_avg:.0f}")

        if conv_models:
            conv_avg = np.mean([r['accuracy'] for r in conv_models])
            conv_ops_avg = np.mean([r['expected_ops'] for r in conv_models])
            print(f"\nConvolutional Models:")
            print(f"  Count: {len(conv_models)}")
            print(f"  Avg Accuracy: {conv_avg:.2f}%")
            print(f"  Avg Ops: {conv_ops_avg:.0f}")

        # Find best trade-off (high accuracy, low ops)
        print("\n" + "="*80)
        print("TRADE-OFF ANALYSIS: Best Balance of Quality and Efficiency")
        print("="*80)

        # Calculate trade-off score
        for r in all_results:
            # Score = accuracy / sqrt(ops) to balance both factors
            r['tradeoff_score'] = r['accuracy'] / np.sqrt(r['expected_ops'])

        all_results.sort(key=lambda x: x['tradeoff_score'], reverse=True)

        print("\nTop 3 Models by Trade-off Score (accuracy/√ops):")
        print("-"*80)
        for i, result in enumerate(all_results[:3]):
            print(f"{i+1}. {result['experiment_id']}: Score={result['tradeoff_score']:.2f}")
            print(f"   Accuracy: {result['accuracy']:.2f}%")
            print(f"   Expected Ops: {result['expected_ops']}")
            print(f"   Name: {result['name']}")

        # Save results
        results_path = "/root/proof_chain/gan_experiments/classifier_accuracy_comparison.json"
        with open(results_path, 'w') as f:
            json.dump({
                'real_accuracy': real_accuracy,
                'model_results': all_results
            }, f, indent=2)

        print(f"\n✓ Results saved to {results_path}")

        # Final recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)

        best_tradeoff = all_results[0]
        print(f"\n🎯 Recommended Model: {best_tradeoff['experiment_id']}")
        print(f"   - Trade-off Score: {best_tradeoff['tradeoff_score']:.2f}")
        print(f"   - Class Accuracy: {best_tradeoff['accuracy']:.2f}%")
        print(f"   - Expected Ops: {best_tradeoff['expected_ops']}")
        print(f"   - Architecture: {best_tradeoff['name']}")

        if best_tradeoff['accuracy'] > 50 and best_tradeoff['expected_ops'] < 100:
            print(f"\n✅ This model achieves good balance:")
            print(f"   - Decent class conditioning ({best_tradeoff['accuracy']:.1f}% accuracy)")
            print(f"   - ZK-friendly circuit size ({best_tradeoff['expected_ops']} ops)")
        elif best_tradeoff['expected_ops'] < 50:
            print(f"\n⚡ Ultra-efficient for ZK but sacrifices quality")
        else:
            print(f"\n⚠️ May need further optimization for ZK deployment")

if __name__ == "__main__":
    main()
