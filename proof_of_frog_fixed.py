#!/usr/bin/env python3
"""
Proof-of-Frog with Corrected Polycommit Setup

KEY FIX: Added calibrate_settings() after gen_settings() to properly
generate KZG commitments. This was missing from the original implementation.

EZKL workflow should be:
1. gen_settings() - create initial settings
2. calibrate_settings() - finalize and calibrate (THIS WAS MISSING!)
3. compile_circuit() - compile with calibrated settings
4. setup() - generate keys
5. prove() - generate proof (now with proper commitments!)
"""

import json
import os
import sys
from pathlib import Path

try:
    import ezkl
except ImportError:
    print("ERROR: ezkl not available")
    sys.exit(1)

print("="*70)
print("PROOF-OF-FROG: Fixed Polycommit Setup with Calibration")
print("="*70)
print()
print("KEY FIX: Adding calibrate_settings() to generate KZG commitments")
print("="*70)
print()

base_dir = Path('/root/ezkl_logs/models/ProofOfFrog_Fixed')
gan_dir = base_dir / 'gan'
cls_dir = base_dir / 'classifier'

for d in [gan_dir, cls_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Model paths
gan_model = Path('/root/cifar_gan_training/tiny_conditional_gan_cifar10.onnx')
cls_model = Path('/root/cifar_gan_training/tiny_classifier_cifar10.onnx')

if not gan_model.exists() or not cls_model.exists():
    print("ERROR: Models not found")
    sys.exit(1)

print(f"✓ GAN model: {gan_model}")
print(f"✓ Classifier model: {cls_model}")
print()

# ============================================================================
# STEP 1: GAN SETUP with polycommit output
# ============================================================================
print("="*70)
print("STEP 1: GAN Setup with Polycommit Output")
print("="*70)
print()

os.chdir(gan_dir)

# Copy model
import shutil
shutil.copy(gan_model, 'network.onnx')

# Create input
import numpy as np
np.random.seed(42)
latent = np.random.randn(32).astype(np.float32)
class_onehot = np.zeros(10, dtype=np.float32)
class_onehot[6] = 1.0  # frog
gan_input = np.concatenate([latent, class_onehot])

with open('input.json', 'w') as f:
    json.dump({'input_data': [gan_input.tolist()]}, f)

print("1a. Generating settings with polycommit output...")
run_args = ezkl.PyRunArgs()
run_args.output_visibility = "polycommit"  # Private output via KZG
run_args.param_visibility = "fixed"
run_args.input_visibility = "public"

ezkl.gen_settings(
    model='network.onnx',
    output='settings.json',
    py_run_args=run_args
)
print("  ✓ Settings generated")

print("1b. Calibrating settings (THIS WAS MISSING!)...")
ezkl.calibrate_settings(
    data='input.json',
    model='network.onnx',
    settings='settings.json',
    target='resources'
)
print("  ✓ Settings calibrated - KZG commitments configured")

print("1c. Compiling circuit...")
ezkl.compile_circuit(
    model='network.onnx',
    compiled_circuit='network.ezkl',
    settings_path='settings.json'
)
print("  ✓ Circuit compiled")

# Verify settings
with open('settings.json', 'r') as f:
    settings = json.load(f)
    print(f"  ✓ GAN logrows: {settings['run_args']['logrows']}")
    print(f"  ✓ GAN output_visibility: {settings['run_args']['output_visibility']}")

print()

# ============================================================================
# STEP 2: CLASSIFIER SETUP with polycommit input
# ============================================================================
print("="*70)
print("STEP 2: Classifier Setup with Polycommit Input")
print("="*70)
print()

os.chdir(cls_dir)

# Copy model
shutil.copy(cls_model, 'network.onnx')

# Create dummy input (3072 values from GAN output)
cls_input_data = [0.0] * 3072
with open('input.json', 'w') as f:
    json.dump({'input_data': [cls_input_data]}, f)

print("2a. Generating settings with polycommit input...")
run_args = ezkl.PyRunArgs()
run_args.input_visibility = "polycommit"  # Match GAN's polycommit output
run_args.param_visibility = "fixed"
run_args.output_visibility = "public"  # Classification can be public

ezkl.gen_settings(
    model='network.onnx',
    output='settings.json',
    py_run_args=run_args
)
print("  ✓ Settings generated")

print("2b. Calibrating settings (THIS WAS MISSING!)...")
ezkl.calibrate_settings(
    data='input.json',
    model='network.onnx',
    settings='settings.json',
    target='resources'
)
print("  ✓ Settings calibrated - KZG commitment input configured")

print("2c. Compiling circuit...")
ezkl.compile_circuit(
    model='network.onnx',
    compiled_circuit='network.ezkl',
    settings_path='settings.json'
)
print("  ✓ Circuit compiled")

# Verify settings
with open('settings.json', 'r') as f:
    settings = json.load(f)
    print(f"  ✓ Classifier logrows: {settings['run_args']['logrows']}")
    print(f"  ✓ Classifier input_visibility: {settings['run_args']['input_visibility']}")

print()

# ============================================================================
# STEP 3: GENERATE SRS
# ============================================================================
print("="*70)
print("STEP 3: Generating Shared SRS")
print("="*70)
print()

os.chdir(gan_dir)
with open('settings.json', 'r') as f:
    gan_logrows = json.load(f)['run_args']['logrows']

os.chdir(cls_dir)
with open('settings.json', 'r') as f:
    cls_logrows = json.load(f)['run_args']['logrows']

max_logrows = max(gan_logrows, cls_logrows)
print(f"Using logrows={max_logrows} (max of GAN:{gan_logrows}, Classifier:{cls_logrows})")

srs_path = base_dir / 'kzg.srs'
ezkl.get_srs(srs_path=str(srs_path), logrows=max_logrows)
print(f"  ✓ SRS generated: {srs_path.stat().st_size/(1024**3):.2f} GB")

# Copy to both directories
shutil.copy(srs_path, gan_dir / 'kzg.srs')
shutil.copy(srs_path, cls_dir / 'kzg.srs')
print(f"  ✓ SRS copied to both directories")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
print("SETUP COMPLETE - Ready for Key Generation")
print("="*70)
print()
print("Next steps:")
print("  1. Generate keys: setup --compiled-circuit network.ezkl")
print("  2. Generate proofs: prove --compiled-circuit network.ezkl")
print("  3. Verify commitments in witness files")
print()
print("Expected difference from before:")
print("  OLD: processed_outputs: {'polycommit': None}  ❌")
print("  NEW: processed_outputs: {'polycommit': <actual commitment>}  ✅")
print()
print("Directories:")
print(f"  GAN: {gan_dir}")
print(f"  Classifier: {cls_dir}")
print()
