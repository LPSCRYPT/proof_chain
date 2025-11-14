#!/usr/bin/env python3
"""
Comprehensive test suite for the trained ZK-optimized conditional GAN
Tests all three requirements: conditioning, circuit size, and visual quality
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json
import sys
import os

# Import our modules
from zk_optimized_conditional_gan import (
    ZKOptimizedGeneratorV2,
    ZKOptimizedDiscriminator
)
from test_conditional_gan_proper import (
    test_mode_collapse,
    test_class_conditioning,
    test_classifier_accuracy,
    ZKOptimizedClassifier,
    CIFAR10_CLASSES
)
from gan_evaluation_metrics import (
    GANEvaluator,
    create_per_class_grid,
    test_circuit_compatibility,
    count_einsum_operations
)

def load_trained_model(generator_path, device="cuda"):
    """Load the trained generator model"""
    generator = ZKOptimizedGeneratorV2(
        latent_dim=100,
        num_classes=10,
        embed_dim=50,
        ngf=48
    ).to(device)

    if os.path.exists(generator_path):
        print(f"Loading trained model from {generator_path}")
        state_dict = torch.load(generator_path, map_location=device)
        generator.load_state_dict(state_dict)
        print("✓ Model loaded successfully")
    else:
        print(f"✗ Model not found at {generator_path}")
        print("  Using random initialization for testing")

    return generator

def comprehensive_test(generator_path="zk_conditional_gan_v2_final.pth",
                      classifier_path="cifar_gan_training/zk_classifier_avgpool.pth"):
    """Run comprehensive tests on all three requirements"""

    print("="*70)
    print("COMPREHENSIVE ZK-OPTIMIZED CONDITIONAL GAN TEST")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load models
    generator = load_trained_model(generator_path, device)
    generator.eval()

    # Initialize classifier for quality testing
    classifier = ZKOptimizedClassifier(num_classes=10).to(device)
    if os.path.exists(classifier_path):
        print(f"Loading classifier from {classifier_path}")
        state = torch.load(classifier_path, map_location=device)
        if list(state.keys())[0].startswith("module."):
            state = {k[7:]: v for k, v in state.items()}
        classifier.load_state_dict(state)
        print("✓ Classifier loaded\n")
    else:
        print(f"✗ Classifier not found at {classifier_path}\n")

    results = {}

    # =================================================================
    # TEST 1: CIRCUIT COMPATIBILITY (Requirement 2)
    # =================================================================
    print("="*70)
    print("TEST 1: CIRCUIT COMPATIBILITY")
    print("="*70)

    einsum_ops = count_einsum_operations(generator)
    circuit_compatible = test_circuit_compatibility(generator, target_ops=100)

    results["circuit"] = {
        "estimated_ops": einsum_ops,
        "target": 100,
        "compatible": circuit_compatible,
        "status": "✓ PASS" if circuit_compatible else "✗ FAIL"
    }

    print(f"\nCircuit Test Result: {results[\"circuit\"][\"status\"]}")

    # =================================================================
    # TEST 2: CLASS CONDITIONING (Requirement 1)
    # =================================================================
    print("\n" + "="*70)
    print("TEST 2: CLASS CONDITIONING")
    print("="*70)

    # Test mode collapse
    diversity_scores = test_mode_collapse(generator, device)
    mean_diversity = np.mean(diversity_scores)

    print("\nIntra-class diversity:")
    for i, score in enumerate(diversity_scores):
        status = "✓" if score > 0.05 else "✗"
        print(f"  {status} {CIFAR10_CLASSES[i]:12s}: {score:.4f}")

    # Test class separation
    mean_diff, min_diff, max_diff = test_class_conditioning(generator, device)

    print(f"\nClass separation (same noise, different labels):")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Min difference: {min_diff:.4f}")
    print(f"  Max difference: {max_diff:.4f}")

    conditioning_works = mean_diversity > 0.05 and mean_diff > 0.1

    results["conditioning"] = {
        "diversity": float(mean_diversity),
        "diversity_threshold": 0.05,
        "class_separation": float(mean_diff),
        "separation_threshold": 0.1,
        "status": "✓ PASS" if conditioning_works else "✗ FAIL"
    }

    print(f"\nConditioning Test Result: {results[\"conditioning\"][\"status\"]}")

    # =================================================================
    # TEST 3: VISUAL QUALITY (Requirement 3)
    # =================================================================
    print("\n" + "="*70)
    print("TEST 3: VISUAL QUALITY")
    print("="*70)

    # Test classifier accuracy
    print("\nTesting classifier accuracy on generated images...")
    accuracy_results = test_classifier_accuracy(generator, classifier,
                                               num_samples=50, device=device)

    accuracies = []
    print("\nPer-class accuracy:")
    for class_name, metrics in accuracy_results.items():
        acc = metrics["accuracy"]
        accuracies.append(acc)
        status = "✓" if acc > 50 else "✗"
        print(f"  {status} {class_name:12s}: {acc:6.2f}%")

    mean_accuracy = np.mean(accuracies)

    # Calculate more quality metrics
    print("\nCalculating advanced quality metrics...")
    evaluator = GANEvaluator(device=device)
    quality_results = evaluator.evaluate_generator(generator,
                                                  num_samples=1000,
                                                  batch_size=50)

    is_mean = quality_results["inception_score"]["mean"]
    is_std = quality_results["inception_score"]["std"]
    mode_entropy = quality_results["mode_entropy"]

    print(f"\nQuality Metrics:")
    print(f"  Classifier Accuracy: {mean_accuracy:.2f}%")
    print(f"  Inception Score: {is_mean:.2f} ± {is_std:.2f}")
    print(f"  Mode Entropy: {mode_entropy:.2f} / 2.30")

    quality_pass = mean_accuracy > 70 and is_mean > 6.0

    results["quality"] = {
        "classifier_accuracy": float(mean_accuracy),
        "accuracy_threshold": 70,
        "inception_score": float(is_mean),
        "is_threshold": 6.0,
        "mode_entropy": float(mode_entropy),
        "status": "✓ PASS" if quality_pass else "✗ FAIL"
    }

    print(f"\nQuality Test Result: {results[\"quality\"][\"status\"]}")

    # =================================================================
    # OVERALL ASSESSMENT
    # =================================================================
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)

    all_pass = all([
        results["circuit"]["compatible"],
        conditioning_works,
        quality_pass
    ])

    print(f"\n1. Circuit Compatibility (<100 ops): {results[\"circuit\"][\"status\"]}")
    print(f"   - Estimated ops: {results[\"circuit\"][\"estimated_ops\"]}")

    print(f"\n2. Class Conditioning: {results[\"conditioning\"][\"status\"]}")
    print(f"   - Diversity: {results[\"conditioning\"][\"diversity\"]:.4f}")
    print(f"   - Separation: {results[\"conditioning\"][\"class_separation\"]:.4f}")

    print(f"\n3. Visual Quality: {results[\"quality\"][\"status\"]}")
    print(f"   - Accuracy: {results[\"quality\"][\"classifier_accuracy\"]:.2f}%")
    print(f"   - IS: {results[\"quality\"][\"inception_score\"]:.2f}")

    if all_pass:
        print("\n🎉 SUCCESS! All requirements met!")
        print("The model is ready for ZK deployment!")
    else:
        print("\n⚠️ Some requirements not met:")
        if not results["circuit"]["compatible"]:
            print("  - Circuit size exceeds limit")
        if not conditioning_works:
            print("  - Class conditioning needs improvement")
        if not quality_pass:
            print("  - Visual quality below threshold")

    # Save results
    results["timestamp"] = datetime.now().isoformat()
    results["overall_pass"] = all_pass

    with open("zk_gan_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to zk_gan_test_results.json")

    # Generate visual samples
    print("\nGenerating visual samples...")
    fig = create_per_class_grid(generator, device=device,
                               save_path="zk_gan_samples.png")
    print("✓ Visual samples saved to zk_gan_samples.png")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-path",
                       default="zk_conditional_gan_v2_final.pth",
                       help="Path to trained generator")
    parser.add_argument("--classifier-path",
                       default="cifar_gan_training/zk_classifier_avgpool.pth",
                       help="Path to classifier")

    args = parser.parse_args()
    comprehensive_test(args.generator_path, args.classifier_path)
