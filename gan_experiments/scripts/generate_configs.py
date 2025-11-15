#!/usr/bin/env python3
"""
Generate 30 GAN configurations for comprehensive architecture testing
Finding the sweet spot between quality and ZK circuit compatibility
"""

import json
import os

def generate_gan_configs():
    """Generate 30 different GAN configurations across 4 tiers"""
    configs = []

    # Tier 1: Ultra-Light Models (15-40 einsum ops)
    tier1_configs = [
        {
            "experiment_id": "exp_001",
            "name": "ultra_minimal",
            "tier": 1,
            "architecture": {
                "latent_dim": 50,
                "embed_dim": 20,
                "ngf": 32,
                "num_layers": 3,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 50,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 15
        },
        {
            "experiment_id": "exp_002",
            "name": "minimal_no_bn",
            "tier": 1,
            "architecture": {
                "latent_dim": 64,
                "embed_dim": 25,
                "ngf": 32,
                "num_layers": 3,
                "use_bias": False,
                "use_batchnorm": False,
                "activation": "leaky_relu"
            },
            "training": {
                "epochs": 60,
                "lr_g": 0.00015,
                "lr_d": 0.00008
            },
            "expected_ops": 20
        },
        {
            "experiment_id": "exp_003",
            "name": "tiny_linear",
            "tier": 1,
            "architecture": {
                "latent_dim": 75,
                "embed_dim": 30,
                "ngf": 24,
                "num_layers": 3,
                "linear_layers": 2,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 70,
                "lr_g": 0.00025,
                "lr_d": 0.00012
            },
            "expected_ops": 25
        },
        {
            "experiment_id": "exp_004",
            "name": "micro_conv",
            "tier": 1,
            "architecture": {
                "latent_dim": 80,
                "embed_dim": 32,
                "ngf": 28,
                "num_layers": 3,
                "use_bias": False,
                "kernel_size": 3,
                "activation": "gelu"
            },
            "training": {
                "epochs": 60,
                "lr_g": 0.0002,
                "lr_d": 0.0001,
                "label_smoothing": 0.05
            },
            "expected_ops": 30
        },
        {
            "experiment_id": "exp_005",
            "name": "ultra_narrow",
            "tier": 1,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 40,
                "ngf": 24,
                "num_layers": 4,
                "use_bias": False,
                "activation": "tanh"
            },
            "training": {
                "epochs": 80,
                "lr_g": 0.00018,
                "lr_d": 0.00009
            },
            "expected_ops": 35
        }
    ]

    # Tier 2: Balanced Models (40-70 einsum ops)
    tier2_configs = [
        {
            "experiment_id": "exp_006",
            "name": "baseline_current",
            "tier": 2,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 50,
                "ngf": 48,
                "num_layers": 4,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 100,
                "lr_g": 0.0002,
                "lr_d": 0.0001,
                "label_smoothing": 0.1
            },
            "expected_ops": 34
        },
        {
            "experiment_id": "exp_007",
            "name": "wider_filters",
            "tier": 2,
            "architecture": {
                "latent_dim": 90,
                "embed_dim": 45,
                "ngf": 56,
                "num_layers": 3,
                "use_bias": False,
                "activation": "leaky_relu"
            },
            "training": {
                "epochs": 90,
                "lr_g": 0.00022,
                "lr_d": 0.00011
            },
            "expected_ops": 42
        },
        {
            "experiment_id": "exp_008",
            "name": "deeper_layers",
            "tier": 2,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 50,
                "ngf": 40,
                "num_layers": 5,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 100,
                "lr_g": 0.00018,
                "lr_d": 0.00009,
                "gradient_clip": 3.0
            },
            "expected_ops": 45
        },
        {
            "experiment_id": "exp_009",
            "name": "mixed_activation",
            "tier": 2,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 48,
                "ngf": 44,
                "num_layers": 4,
                "use_bias": False,
                "activation": "mixed",  # Mix of relu and leaky_relu
                "dropout": 0.1
            },
            "training": {
                "epochs": 95,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 48
        },
        {
            "experiment_id": "exp_010",
            "name": "skip_connections",
            "tier": 2,
            "architecture": {
                "latent_dim": 96,
                "embed_dim": 48,
                "ngf": 48,
                "num_layers": 4,
                "use_skip": True,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 100,
                "lr_g": 0.00021,
                "lr_d": 0.000105
            },
            "expected_ops": 52
        },
        {
            "experiment_id": "exp_011",
            "name": "instance_norm",
            "tier": 2,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 50,
                "ngf": 48,
                "num_layers": 4,
                "normalization": "instance",
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 95,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 54
        },
        {
            "experiment_id": "exp_012",
            "name": "spectral_norm",
            "tier": 2,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 52,
                "ngf": 46,
                "num_layers": 4,
                "use_spectral_norm": True,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 100,
                "lr_g": 0.00019,
                "lr_d": 0.000095
            },
            "expected_ops": 56
        },
        {
            "experiment_id": "exp_013",
            "name": "adaptive_filters",
            "tier": 2,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 50,
                "ngf": [32, 48, 64, 48],  # Adaptive filter sizes
                "num_layers": 4,
                "use_bias": False,
                "activation": "gelu"
            },
            "training": {
                "epochs": 100,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 58
        },
        {
            "experiment_id": "exp_014",
            "name": "attention_light",
            "tier": 2,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 50,
                "ngf": 44,
                "num_layers": 4,
                "use_attention": "light",  # Simplified attention
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 100,
                "lr_g": 0.00018,
                "lr_d": 0.00009
            },
            "expected_ops": 62
        },
        {
            "experiment_id": "exp_015",
            "name": "progressive_growth",
            "tier": 2,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 50,
                "ngf": 48,
                "num_layers": 4,
                "progressive": True,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 120,
                "lr_g": 0.0002,
                "lr_d": 0.0001,
                "progressive_epochs": [30, 60, 90]
            },
            "expected_ops": 65
        }
    ]

    # Tier 3: Quality-Focused (70-100 einsum ops)
    tier3_configs = [
        {
            "experiment_id": "exp_016",
            "name": "high_capacity",
            "tier": 3,
            "architecture": {
                "latent_dim": 128,
                "embed_dim": 64,
                "ngf": 64,
                "num_layers": 4,
                "use_bias": True,
                "activation": "leaky_relu"
            },
            "training": {
                "epochs": 120,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 72
        },
        {
            "experiment_id": "exp_017",
            "name": "wider_deeper",
            "tier": 3,
            "architecture": {
                "latent_dim": 120,
                "embed_dim": 60,
                "ngf": 56,
                "num_layers": 5,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 110,
                "lr_g": 0.00018,
                "lr_d": 0.00009
            },
            "expected_ops": 75
        },
        {
            "experiment_id": "exp_018",
            "name": "residual_blocks",
            "tier": 3,
            "architecture": {
                "latent_dim": 128,
                "embed_dim": 64,
                "ngf": 52,
                "num_layers": 4,
                "residual_blocks": 2,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 120,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 78
        },
        {
            "experiment_id": "exp_019",
            "name": "multi_scale",
            "tier": 3,
            "architecture": {
                "latent_dim": 128,
                "embed_dim": 60,
                "ngf": 48,
                "num_layers": 5,
                "multi_scale": True,
                "use_bias": False,
                "activation": "gelu"
            },
            "training": {
                "epochs": 125,
                "lr_g": 0.00019,
                "lr_d": 0.000095
            },
            "expected_ops": 80
        },
        {
            "experiment_id": "exp_020",
            "name": "style_modulation",
            "tier": 3,
            "architecture": {
                "latent_dim": 128,
                "embed_dim": 64,
                "ngf": 48,
                "num_layers": 4,
                "style_layers": 2,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 120,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 82
        },
        {
            "experiment_id": "exp_021",
            "name": "wavelet_transform",
            "tier": 3,
            "architecture": {
                "latent_dim": 120,
                "embed_dim": 60,
                "ngf": 52,
                "num_layers": 4,
                "use_wavelet": True,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 115,
                "lr_g": 0.00021,
                "lr_d": 0.000105
            },
            "expected_ops": 85
        },
        {
            "experiment_id": "exp_022",
            "name": "dual_discriminator",
            "tier": 3,
            "architecture": {
                "latent_dim": 128,
                "embed_dim": 64,
                "ngf": 56,
                "num_layers": 4,
                "dual_disc": True,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 130,
                "lr_g": 0.00018,
                "lr_d": 0.00009
            },
            "expected_ops": 88
        },
        {
            "experiment_id": "exp_023",
            "name": "conditional_bn",
            "tier": 3,
            "architecture": {
                "latent_dim": 128,
                "embed_dim": 64,
                "ngf": 52,
                "num_layers": 4,
                "conditional_bn": True,
                "use_bias": False,
                "activation": "relu"
            },
            "training": {
                "epochs": 120,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 90
        },
        {
            "experiment_id": "exp_024",
            "name": "pyramid_features",
            "tier": 3,
            "architecture": {
                "latent_dim": 128,
                "embed_dim": 64,
                "ngf": 48,
                "num_layers": 5,
                "feature_pyramid": True,
                "use_bias": False,
                "activation": "leaky_relu"
            },
            "training": {
                "epochs": 125,
                "lr_g": 0.00019,
                "lr_d": 0.000095
            },
            "expected_ops": 93
        },
        {
            "experiment_id": "exp_025",
            "name": "max_quality",
            "tier": 3,
            "architecture": {
                "latent_dim": 150,
                "embed_dim": 75,
                "ngf": 64,
                "num_layers": 5,
                "use_bias": True,
                "activation": "gelu",
                "use_attention": True
            },
            "training": {
                "epochs": 150,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 98
        }
    ]

    # Tier 4: Experimental Variations
    tier4_configs = [
        {
            "experiment_id": "exp_026",
            "name": "mlp_only",
            "tier": 4,
            "architecture": {
                "model_type": "mlp",
                "latent_dim": 100,
                "embed_dim": 50,
                "hidden_dims": [512, 1024, 2048, 3072],
                "activation": "relu"
            },
            "training": {
                "epochs": 80,
                "lr_g": 0.0003,
                "lr_d": 0.00015
            },
            "expected_ops": 20
        },
        {
            "experiment_id": "exp_027",
            "name": "vae_hybrid",
            "tier": 4,
            "architecture": {
                "model_type": "vae_gan",
                "latent_dim": 100,
                "embed_dim": 50,
                "ngf": 48,
                "num_layers": 4,
                "kl_weight": 0.01,
                "activation": "relu"
            },
            "training": {
                "epochs": 100,
                "lr_g": 0.0002,
                "lr_d": 0.0001
            },
            "expected_ops": 45
        },
        {
            "experiment_id": "exp_028",
            "name": "quantized_weights",
            "tier": 4,
            "architecture": {
                "latent_dim": 100,
                "embed_dim": 50,
                "ngf": 48,
                "num_layers": 4,
                "quantization": 8,  # 8-bit quantization
                "activation": "relu"
            },
            "training": {
                "epochs": 120,
                "lr_g": 0.00015,
                "lr_d": 0.000075
            },
            "expected_ops": 30
        },
        {
            "experiment_id": "exp_029",
            "name": "pruned_sparse",
            "tier": 4,
            "architecture": {
                "latent_dim": 128,
                "embed_dim": 64,
                "ngf": 64,
                "num_layers": 4,
                "sparsity": 0.3,  # 30% pruning
                "activation": "relu"
            },
            "training": {
                "epochs": 130,
                "lr_g": 0.0002,
                "lr_d": 0.0001,
                "pruning_epochs": [40, 80]
            },
            "expected_ops": 55
        },
        {
            "experiment_id": "exp_030",
            "name": "knowledge_distilled",
            "tier": 4,
            "architecture": {
                "latent_dim": 80,
                "embed_dim": 40,
                "ngf": 40,
                "num_layers": 3,
                "teacher_model": "exp_025",  # Distill from max_quality
                "activation": "relu"
            },
            "training": {
                "epochs": 100,
                "lr_g": 0.0002,
                "lr_d": 0.0001,
                "distillation_weight": 0.5
            },
            "expected_ops": 28
        }
    ]

    # Combine all configs
    configs = tier1_configs + tier2_configs + tier3_configs + tier4_configs

    return configs

def save_configs(configs):
    """Save configuration files"""
    # Save all configs
    with open('all_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)

    # Save by tier
    for tier in range(1, 5):
        tier_configs = [c for c in configs if c['tier'] == tier]
        with open(f'tier{tier}_configs.json', 'w') as f:
            json.dump(tier_configs, f, indent=2)
        print(f"Tier {tier}: {len(tier_configs)} configurations")

    print(f"\nTotal configurations: {len(configs)}")

    # Create summary
    summary = {
        "total_experiments": len(configs),
        "tier1": len([c for c in configs if c['tier'] == 1]),
        "tier2": len([c for c in configs if c['tier'] == 2]),
        "tier3": len([c for c in configs if c['tier'] == 3]),
        "tier4": len([c for c in configs if c['tier'] == 4]),
        "expected_ops_range": {
            "min": min(c['expected_ops'] for c in configs),
            "max": max(c['expected_ops'] for c in configs),
            "mean": sum(c['expected_ops'] for c in configs) / len(configs)
        }
    }

    with open('experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(f"  Min ops: {summary['expected_ops_range']['min']}")
    print(f"  Max ops: {summary['expected_ops_range']['max']}")
    print(f"  Mean ops: {summary['expected_ops_range']['mean']:.1f}")

if __name__ == "__main__":
    configs = generate_gan_configs()
    save_configs(configs)
    print("\n✓ Generated 30 GAN configurations")
    print("  Files created:")
    print("  - all_configs.json")
    print("  - tier1_configs.json to tier4_configs.json")
    print("  - experiment_summary.json")