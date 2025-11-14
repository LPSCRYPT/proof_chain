#!/usr/bin/env python3
"""
Compile the trained ZK-optimized conditional GAN to EZKL circuit
Get actual constraint counts and proof size metrics
"""

import torch
import numpy as np
import json
import os
from zk_optimized_conditional_gan import ZKOptimizedGeneratorV2

def export_generator_to_onnx():
    """Export the trained generator to ONNX format for EZKL compilation"""
    print("="*70)
    print("EXPORTING GENERATOR TO ONNX FOR EZKL")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained generator
    generator = ZKOptimizedGeneratorV2(
        latent_dim=100,
        num_classes=10,
        embed_dim=50,
        ngf=48
    ).to(device)

    # Load checkpoint
    checkpoint_path = 'checkpoints/final_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'generator_state' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state'])
            print(f"✓ Loaded generator from epoch {checkpoint.get('epoch', '?')}")
        else:
            print("✓ Loaded generator weights")
    else:
        print("⚠️ No checkpoint found, using random weights")

    generator.eval()

    # Create sample inputs
    batch_size = 1
    noise = torch.randn(batch_size, 100, 1, 1).to(device)
    labels = torch.tensor([0], dtype=torch.long).to(device)  # Single class for example

    # Export to ONNX
    onnx_path = 'generator_zk.onnx'
    print(f"\nExporting to {onnx_path}...")

    torch.onnx.export(
        generator,
        (noise, labels),
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

    # Get file size
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ONNX file size: {file_size:.2f} MB")

    return onnx_path

def prepare_ezkl_input():
    """Prepare input data for EZKL compilation"""
    print("\n" + "="*70)
    print("PREPARING EZKL INPUT DATA")
    print("="*70)

    # Create sample input
    noise = np.random.randn(1, 100, 1, 1).astype(np.float32)
    labels = np.array([0], dtype=np.int64)

    # Create input.json for EZKL
    input_data = {
        "input_data": [
            noise.flatten().tolist(),
            labels.tolist()
        ]
    }

    input_path = 'input.json'
    with open(input_path, 'w') as f:
        json.dump(input_data, f)

    print(f"✓ Input data saved to {input_path}")

    # Create calibration dataset
    calibration_data = []
    for i in range(10):  # One sample per class
        noise = np.random.randn(1, 100, 1, 1).astype(np.float32)
        labels = np.array([i], dtype=np.int64)
        calibration_data.append({
            "input_data": [
                noise.flatten().tolist(),
                labels.tolist()
            ]
        })

    calibration_path = 'calibration.json'
    with open(calibration_path, 'w') as f:
        json.dump(calibration_data, f)

    print(f"✓ Calibration data saved to {calibration_path}")

    return input_path, calibration_path

def create_ezkl_settings():
    """Create settings for EZKL compilation"""
    print("\n" + "="*70)
    print("CREATING EZKL SETTINGS")
    print("="*70)

    settings = {
        "run_args": {
            "tolerance": {
                "val": 0.0,
                "scale": 10
            },
            "scale": 10,
            "bits": 20,
            "logrows": 18,
            "num_inner_cols": 2,
            "variables": [
                ["batch_size", 1]
            ],
            "input_visibility": "Private",
            "output_visibility": "Public",
            "param_visibility": "Fixed"
        },
        "num_rows": 262144,
        "total_assignments": 0,
        "total_const_size": 0,
        "model_instance_shapes": [
            [1, 100, 1, 1],
            [1]
        ],
        "model_output_scales": [10],
        "model_input_scales": [10, 0],
        "module_sizes": {},
        "required_lookups": [],
        "required_range_checks": [],
        "check_mode": "UNSAFE",
        "version": "9.1.0",
        "num_blinding_factors": null,
        "timestamp": null
    }

    settings_path = 'settings.json'
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)

    print(f"✓ Settings saved to {settings_path}")
    print(f"  Logrows: {settings['run_args']['logrows']}")
    print(f"  Bits: {settings['run_args']['bits']}")
    print(f"  Scale: {settings['run_args']['scale']}")

    return settings_path

def compile_ezkl_circuit():
    """Compile the ONNX model to EZKL circuit using bash commands"""
    print("\n" + "="*70)
    print("COMPILING TO EZKL CIRCUIT")
    print("="*70)

    commands = [
        # 1. Generate settings from calibration data
        "ezkl gen-settings -M generator_zk.onnx -O settings.json --settings-path settings.json --calibration-path calibration.json",

        # 2. Calibrate settings
        "ezkl calibrate-settings -M generator_zk.onnx --settings-path settings.json --calibration-path calibration.json",

        # 3. Compile the circuit
        "ezkl compile-circuit -M generator_zk.onnx --settings-path settings.json --compiled-circuit network.ezkl",

        # 4. Generate structured reference string
        "ezkl gen-srs --settings-path settings.json --srs-path kzg.srs",

        # 5. Setup proving and verification keys
        "ezkl setup --compiled-circuit network.ezkl --vk-path vk.key --pk-path pk.key --srs-path kzg.srs",

        # 6. Generate witness
        "ezkl gen-witness --data input.json --compiled-circuit network.ezkl --witness witness.json",

        # 7. Generate proof
        "ezkl prove --witness witness.json --compiled-circuit network.ezkl --pk-path pk.key --proof-path proof.json --srs-path kzg.srs",

        # 8. Verify proof
        "ezkl verify --settings-path settings.json --proof-path proof.json --vk-path vk.key --srs-path kzg.srs",

        # 9. Get circuit statistics
        "ezkl table --compiled-circuit network.ezkl"
    ]

    print("\nEZKL compilation commands to run:")
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd}")

    # Save commands to script
    script_path = 'compile_ezkl.sh'
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# EZKL compilation script for ZK-optimized GAN\n\n")
        f.write("set -e  # Exit on error\n\n")

        for cmd in commands:
            f.write(f"echo '\\n>>> {cmd}'\n")
            f.write(f"{cmd}\n\n")

        f.write("echo '\\n✓ EZKL compilation complete!'\n")

    os.chmod(script_path, 0o755)
    print(f"\n✓ Compilation script saved to {script_path}")
    print("  Run: ./compile_ezkl.sh")

    return script_path

def estimate_circuit_metrics():
    """Estimate circuit metrics based on model architecture"""
    print("\n" + "="*70)
    print("ESTIMATED CIRCUIT METRICS")
    print("="*70)

    # Load model to count parameters
    generator = ZKOptimizedGeneratorV2()
    total_params = sum(p.numel() for p in generator.parameters())

    # Estimate based on architecture
    conv_layers = 5  # 4 ConvTranspose + 1 Conv2d
    batchnorm_layers = 4
    activation_layers = 5  # ReLU + Tanh

    # Rough estimation formulas
    estimated_constraints = (
        conv_layers * 50000 +  # Convolutions are expensive
        batchnorm_layers * 10000 +  # BatchNorm operations
        activation_layers * 5000 +  # Activations
        total_params // 100  # Parameter loading
    )

    estimated_proof_size = estimated_constraints // 1000  # Rough estimate in KB
    estimated_proving_time = estimated_constraints // 10000  # Seconds

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Conv/ConvTranspose layers: {conv_layers}")
    print(f"  BatchNorm layers: {batchnorm_layers}")
    print(f"  Activation layers: {activation_layers}")

    print(f"\nEstimated Circuit Metrics:")
    print(f"  Constraints: ~{estimated_constraints:,}")
    print(f"  Proof size: ~{estimated_proof_size} KB")
    print(f"  Proving time: ~{estimated_proving_time} seconds")
    print(f"  Circuit depth: ~{conv_layers * 10}")

    print("\n⚠️ Note: These are rough estimates.")
    print("  Actual metrics will be available after EZKL compilation.")

    return {
        'parameters': total_params,
        'estimated_constraints': estimated_constraints,
        'estimated_proof_size_kb': estimated_proof_size,
        'estimated_proving_time_sec': estimated_proving_time
    }

def main():
    """Main function to prepare GAN for EZKL compilation"""
    print("="*70)
    print("ZK-OPTIMIZED GAN CIRCUIT COMPILATION")
    print("="*70)
    print("Preparing to compile the trained conditional GAN to EZKL circuit")
    print()

    # Step 1: Export to ONNX
    onnx_path = export_generator_to_onnx()

    # Step 2: Prepare input data
    input_path, calibration_path = prepare_ezkl_input()

    # Step 3: Create settings
    settings_path = create_ezkl_settings()

    # Step 4: Generate compilation script
    script_path = compile_ezkl_circuit()

    # Step 5: Estimate metrics
    metrics = estimate_circuit_metrics()

    # Save metrics
    with open('circuit_metrics_estimate.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Install EZKL if not already installed:")
    print("   curl https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash")
    print("\n2. Run the compilation script:")
    print("   ./compile_ezkl.sh")
    print("\n3. Check actual circuit metrics:")
    print("   ezkl table --compiled-circuit network.ezkl")
    print("\n4. Deploy verifier contract:")
    print("   ezkl create-evm-verifier --vk-path vk.key --sol-code-path verifier.sol")
    print("   ezkl deploy-evm-verifier --sol-code-path verifier.sol --rpc-url <RPC_URL>")

    print("\n✓ Preparation complete! Ready for EZKL compilation.")

if __name__ == '__main__':
    main()