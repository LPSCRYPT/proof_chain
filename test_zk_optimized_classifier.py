#!/usr/bin/env python3
"""
Test ZK-Optimized Classifier with EZKL
Generate verifier and compare size
"""

import ezkl
import json
import os
import numpy as np
from pathlib import Path
import subprocess

# Create test directory
test_dir = Path('/root/proof_chain/ezkl_logs/models/ZKOptimized')
test_dir.mkdir(parents=True, exist_ok=True)
os.chdir(test_dir)

print("="*70)
print("ZK-OPTIMIZED CLASSIFIER EZKL TEST")
print("="*70)
print()

# Copy ONNX model
model_path = test_dir / 'classifier.onnx'
subprocess.run(['cp', '/root/proof_chain/cifar_gan_training/zk_classifier_avgpool.onnx', str(model_path)])

# Create test input (flattened RGB image)
test_input = {
    "input_shapes": [[1, 3, 32, 32]],
    "input_data": [np.random.randn(1, 3, 32, 32).flatten().tolist()]
}

with open('input.json', 'w') as f:
    json.dump(test_input, f)

print("Step 1: Generating settings...")
run_args = ezkl.PyRunArgs()
run_args.input_visibility = "polycommit"  # Match original classifier
run_args.output_visibility = "public"
run_args.param_visibility = "fixed"

ezkl.gen_settings(
    model=model_path,
    output='settings.json',
    py_run_args=run_args
)
print("✓ Settings generated")

print("\nStep 2: Calibrating settings...")
ezkl.calibrate_settings(
    data='input.json',
    model=model_path,
    settings='settings.json',
    target='resources'
)
print("✓ Settings calibrated")

# Check the calibrated settings
with open('settings.json', 'r') as f:
    settings = json.load(f)
    print(f"  Logrows: {settings['run_args']['logrows']}")
    print(f"  Num rows: {settings.get('num_rows', 'N/A')}")
    print(f"  Einsum equations: {len(settings.get('einsum_params', {}).get('equations', []))} operations")

print("\nStep 3: Compiling circuit...")
ezkl.compile_circuit(
    model=model_path,
    model='network.ezkl',
    settings_path='settings.json'
)
print("✓ Circuit compiled")

# Generate SRS (smaller for testing)
print("\nStep 4: Generating SRS...")
logrows = settings['run_args']['logrows']
print(f"  Using logrows={logrows} from calibrated settings")
subprocess.run(['/root/.ezkl/ezkl', 'gen-srs', 
                '--logrows', str(logrows),
                '--srs-path', 'kzg.srs'], check=True)
print("✓ SRS generated")

print("\nStep 5: Setting up keys...")
ezkl.setup(
    model='network.ezkl',
    vk_path='vk.key',
    pk_path='pk.key',
    srs_path='kzg.srs'
)

# Check VK size
vk_size = os.path.getsize('vk.key')
print(f"✓ Keys generated")
print(f"  VK size: {vk_size/1024/1024:.1f} MB")

print("\nStep 6: Generating verifier contract...")
subprocess.run(['/root/.ezkl/ezkl', 'create-evm-verifier',
                '--settings-path', 'settings.json',
                '--vk-path', 'vk.key',
                '--srs-path', 'kzg.srs',
                '--sol-code-path', 'verifier.sol',
                '--abi-path', 'verifier_abi.json'], check=True)

# Check verifier size
verifier_size = os.path.getsize('verifier.sol')
verifier_lines = len(open('verifier.sol').readlines())

print(f"✓ Verifier generated")
print(f"  Size: {verifier_size/1024:.1f} KB")
print(f"  Lines: {verifier_lines:,}")

# Count mstore operations
with open('verifier.sol', 'r') as f:
    mstore_count = f.read().count('mstore')
print(f"  mstore operations: {mstore_count}")

print("\n" + "="*70)
print("COMPARISON WITH ORIGINAL CLASSIFIER")
print("="*70)
print()
print("Original Classifier (with MaxPool):")
print("  Verifier: 1,300 KB (22,733 lines)")
print("  Einsum ops: 14,360")
print("  mstore ops: 1,301")
print()
print(f"ZK-Optimized Classifier (with AvgPool):")
print(f"  Verifier: {verifier_size/1024:.1f} KB ({verifier_lines:,} lines)")
print(f"  Einsum ops: {len(settings.get('einsum_params', {}).get('equations', []))}")
print(f"  mstore ops: {mstore_count}")
print()
reduction = (1 - verifier_size/(1300*1024)) * 100
print(f"SIZE REDUCTION: {reduction:.1f}%")
