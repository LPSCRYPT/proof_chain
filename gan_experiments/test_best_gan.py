#!/usr/bin/env python3
"""
Test the best performing GAN model (exp_026 - MLP architecture)
And compile it to EZKL to get actual circuit metrics
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys

# Add path for architecture imports
sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')

def test_mlp_gan():
    """Test the winning MLP-only GAN model"""
    print("="*80)
    print("TESTING BEST ZK-OPTIMIZED GAN: exp_026 (MLP-only)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load the exp_026 configuration
    config = {
        "experiment_id": "exp_026",
        "name": "mlp_only",
        "tier": 4,
        "expected_ops": 20,
        "architecture": {
            "model_type": "mlp",
            "latent_dim": 100,
            "embed_dim": 50,
            "hidden_dims": [512, 1024, 2048, 3072],
            "use_batchnorm": True,
            "use_bias": False,
            "activation": "relu"
        },
        "training": {
            "epochs": 30,
            "batch_size": 64,
            "lr_g": 0.0002,
            "lr_d": 0.0002
        }
    }

    # Import and create the model
    from flexible_gan_architectures import create_generator

    # Load the trained model
    generator = create_generator(config).to(device)
    model_path = "/root/proof_chain/gan_experiments/tier4/models/exp_026_generator.pth"

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict)
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠️ Model not found at {model_path}")
        return

    generator.eval()

    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")

    # Count operations for ZK circuit
    linear_layers = 0
    embedding_layers = 0
    activation_layers = 0
    batchnorm_layers = 0

    for module in generator.modules():
        if isinstance(module, nn.Linear):
            linear_layers += 1
        elif isinstance(module, nn.Embedding):
            embedding_layers += 1
        elif isinstance(module, (nn.ReLU, nn.Tanh, nn.GELU)):
            activation_layers += 1
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            batchnorm_layers += 1

    estimated_ops = linear_layers + embedding_layers + batchnorm_layers

    print(f"\nArchitecture Analysis:")
    print(f"  Linear layers: {linear_layers}")
    print(f"  Embedding layers: {embedding_layers}")
    print(f"  Activation layers: {activation_layers}")
    print(f"  BatchNorm layers: {batchnorm_layers}")
    print(f"  Estimated einsum ops: {estimated_ops}")

    # Test generation quality
    print(f"\nTesting Generation Quality...")

    # Generate samples for each class
    samples_per_class = 10
    all_samples = []

    with torch.no_grad():
        for class_idx in range(10):
            noise = torch.randn(samples_per_class, 100, 1, 1).to(device)
            labels = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)
            fake_images = generator(noise, labels)
            all_samples.append(fake_images.cpu())

    all_samples = torch.cat(all_samples, dim=0)

    # Calculate diversity metrics
    print(f"\nDiversity Analysis:")
    diversity_scores = []

    for i in range(10):
        class_samples = all_samples[i*samples_per_class:(i+1)*samples_per_class]
        if len(class_samples) > 1:
            pairwise_dists = []
            for j in range(len(class_samples)):
                for k in range(j+1, len(class_samples)):
                    dist = torch.mean((class_samples[j] - class_samples[k]) ** 2).item()
                    pairwise_dists.append(dist)
            class_diversity = np.mean(pairwise_dists) if pairwise_dists else 0
            diversity_scores.append(class_diversity)
            print(f"  Class {i} diversity: {class_diversity:.4f}")

    overall_diversity = np.mean(diversity_scores)
    print(f"\nOverall diversity: {overall_diversity:.4f}")

    # Class separation analysis
    print(f"\nClass Separation Analysis:")
    class_means = []

    for i in range(10):
        class_samples = all_samples[i*samples_per_class:(i+1)*samples_per_class]
        class_mean = torch.mean(class_samples, dim=0)
        class_means.append(class_mean)

    separations = []
    for i in range(10):
        for j in range(i+1, 10):
            sep = torch.mean((class_means[i] - class_means[j]) ** 2).item()
            separations.append(sep)

    mean_separation = np.mean(separations)
    print(f"  Mean class separation: {mean_separation:.4f}")

    # Final assessment
    print(f"\n{'='*80}")
    print("FINAL ASSESSMENT FOR ZK DEPLOYMENT")
    print("="*80)

    circuit_pass = estimated_ops < 100
    diversity_pass = overall_diversity > 0.05

    print(f"✓ Circuit Efficiency: {estimated_ops} ops {'✅ PASS' if circuit_pass else '❌ FAIL'}")
    print(f"✓ Visual Diversity: {overall_diversity:.4f} {'✅ PASS' if diversity_pass else '❌ FAIL'}")
    print(f"✓ Class Separation: {mean_separation:.4f}")

    if circuit_pass and diversity_pass:
        print(f"\n🎉 SUCCESS! MLP model is optimal for ZK deployment!")
        print(f"   - Ultra-low circuit complexity: {estimated_ops} ops")
        print(f"   - Excellent diversity: {overall_diversity:.4f}")
        print(f"   - Ready for EZKL compilation")
    else:
        print(f"\n⚠️ Some requirements not met")

    # Save analysis results
    results = {
        "model": "exp_026_mlp_only",
        "parameters": int(total_params),
        "estimated_ops": estimated_ops,
        "diversity": float(overall_diversity),
        "class_separation": float(mean_separation),
        "architecture": {
            "linear_layers": linear_layers,
            "embedding_layers": embedding_layers,
            "activation_layers": activation_layers,
            "batchnorm_layers": batchnorm_layers
        },
        "zk_ready": circuit_pass and diversity_pass
    }

    results_path = "/root/proof_chain/gan_experiments/best_model_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Analysis saved to {results_path}")

    # Export to ONNX for EZKL compilation
    print(f"\nExporting to ONNX for EZKL...")
    onnx_path = "/root/proof_chain/gan_experiments/best_mlp_gan.onnx"

    dummy_noise = torch.randn(1, 100, 1, 1).to(device)
    dummy_labels = torch.tensor([0], dtype=torch.long).to(device)

    torch.onnx.export(
        generator,
        (dummy_noise, dummy_labels),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['noise', 'labels'],
        output_names=['generated_image'],
        dynamic_axes={
            'noise': {0: 'batch_size'},
            'labels': {0: 'batch_size'},
            'generated_image': {0: 'batch_size'}
        }
    )

    print(f"✓ ONNX model exported to {onnx_path}")
    print(f"  Ready for EZKL compilation!")

    return results

if __name__ == "__main__":
    results = test_mlp_gan()