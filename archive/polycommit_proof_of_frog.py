#!/usr/bin/env python3
"""
Proof-of-Frog with Polycommit Composition

Uses EZKL's KZG commitment scheme to keep the intermediate image private
while cryptographically linking GAN → Classifier proofs.
"""

import json
import shutil
import os
import subprocess
from pathlib import Path
import ezkl

# Setup directories
base_dir = Path('/root/ezkl_logs/models/ProofOfFrog_Polycommit')
gan_dir = base_dir / 'gan'
cls_dir = base_dir / 'classifier'

gan_dir.mkdir(parents=True, exist_ok=True)
cls_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PROOF-OF-FROG: Private Polycommit Composition")
print("="*70)
print()

# Copy models and inputs
print("Step 1: Copying models...")
shutil.copy('/root/cifar_gan_training/tiny_conditional_gan_cifar10.onnx', gan_dir / 'network.onnx')
shutil.copy('/root/cifar_gan_training/tiny_classifier_cifar10.onnx', cls_dir / 'network.onnx')
shutil.copy('/root/ezkl_logs/models/TinyConditionalGAN_32x32/input.json', gan_dir / 'input.json')
print("✓ Models copied\n")

# ============================================================================
# GAN SETUP with polycommit output
# ============================================================================
print("Step 2: Setting up GAN with polycommit output...")
os.chdir(gan_dir)

# Generate settings with polycommit output visibility
run_args = ezkl.PyRunArgs()
run_args.output_visibility = "polycommit"  # Keep image private!
run_args.param_visibility = "fixed"
run_args.input_visibility = "public"  # Input (latent + class) can be public

ezkl.gen_settings(
    model='network.onnx',
    output='settings.json',
    py_run_args=run_args
)

print("  Calibrating GAN...")
ezkl.calibrate_settings(
    data='input.json',
    model='network.onnx',
    settings='settings.json',
    target='resources'
)

with open('settings.json', 'r') as f:
    gan_settings = json.load(f)
    logrows = gan_settings.get('run_args', {}).get('logrows', '?')
    print(f"  ✓ GAN ready: logrows={logrows}, output_visibility=polycommit\n")

# ============================================================================
# CLASSIFIER SETUP with polycommit input
# ============================================================================
print("Step 3: Setting up Classifier with polycommit input...")
os.chdir(cls_dir)

# Create placeholder input (will be replaced with GAN output)
cls_input = {
    'input_shapes': [[1, 3, 32, 32]],
    'input_data': [[0.0] * 3072]
}
with open('input.json', 'w') as f:
    json.dump(cls_input, f)

# Generate settings with polycommit input visibility
run_args = ezkl.PyRunArgs()
run_args.input_visibility = "polycommit"  # Match GAN's polycommit output!
run_args.param_visibility = "fixed"
run_args.output_visibility = "public"  # Classification result can be public

ezkl.gen_settings(
    model='network.onnx',
    output='settings.json',
    py_run_args=run_args
)

print("  Calibrating Classifier...")
ezkl.calibrate_settings(
    data='input.json',
    model='network.onnx',
    settings='settings.json',
    target='resources'
)

with open('settings.json', 'r') as f:
    cls_settings = json.load(f)
    logrows = cls_settings.get('run_args', {}).get('logrows', '?')
    print(f"  ✓ Classifier ready: logrows={logrows}, input_visibility=polycommit\n")

# ============================================================================
# COMPILE CIRCUITS
# ============================================================================
print("Step 4: Compiling circuits...")
print("  Compiling GAN...")
os.chdir(gan_dir)
subprocess.run(['/root/.ezkl/ezkl', 'compile-circuit', '-M', 'network.onnx', '--settings-path', 'settings.json', '--compiled-circuit', 'network.ezkl'], check=True, capture_output=True)

print("  Compiling Classifier...")
os.chdir(cls_dir)
subprocess.run(['/root/.ezkl/ezkl', 'compile-circuit', '-M', 'network.onnx', '--settings-path', 'settings.json', '--compiled-circuit', 'network.ezkl'], check=True, capture_output=True)

print("✓ Both circuits compiled\n")

print("="*70)
print("SETUP COMPLETE - Polycommit Composition Ready")
print("="*70)
print()
print("Configuration:")
print(f"  GAN: output_visibility = polycommit (image stays private)")
print(f"  Classifier: input_visibility = polycommit (matches GAN output)")
print()
print("Next steps:")
print("  1. Generate SRS for both circuits")
print("  2. Setup proving/verification keys")
print("  3. Generate GAN witness and proof")
print("  4. Generate Classifier witness (will link to GAN via commitments)")
print("  5. Use swap_proof_commitments() to cryptographically link them")
print("  6. Single verification proves entire GAN→Classifier pipeline!")
print()
print(f"Working directory: {base_dir}")
