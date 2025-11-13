#!/usr/bin/env python3
"""
Fix Classifier Logrows - Regenerate classifier circuit with optimized logrows
Keeps GAN untouched (already working with 6 min proof time)
"""

import ezkl
import json
import os
from pathlib import Path

BASE_DIR = "/root/ezkl_logs/models/ProofOfFrog_Fixed"
CLASSIFIER_DIR = f"{BASE_DIR}/classifier"

print("=" * 70)
print("CLASSIFIER LOGROWS OPTIMIZATION")
print("=" * 70)
print()
print("Goal: Reduce classifier logrows from 23 to 18")
print("Expected improvement: 3+ hours → 6-12 minutes")
print()

# Backup existing classifier settings
print("Step 1: Backing up existing classifier settings...")
os.system(f"cp {CLASSIFIER_DIR}/settings.json {CLASSIFIER_DIR}/settings_logrows23_backup.json")
print("  ✓ Backup created: settings_logrows23_backup.json")
print()

# Read current settings to preserve visibility configurations
print("Step 2: Reading current classifier settings...")
with open(f"{CLASSIFIER_DIR}/settings.json") as f:
    current_settings = json.load(f)

print(f"  Current logrows: {current_settings['run_args']['logrows']}")
print(f"  Current total_assignments: {current_settings['total_assignments']:,}")
print()

# Generate new settings with reduced logrows
print("Step 3: Generating new settings with logrows=18...")
os.chdir(CLASSIFIER_DIR)

run_args = ezkl.PyRunArgs()
run_args.logrows = 18  # Reduce from 23 to 18 (32x smaller circuit)
run_args.input_visibility = "polycommit"  # CRITICAL: maintain KZG input
run_args.param_visibility = "fixed"
run_args.output_visibility = "public"

ezkl.gen_settings(
    model='network.onnx',
    output='settings_new.json',
    py_run_args=run_args
)
print("  ✓ New settings generated")
print()

# Calibrate with target='resources' for speed
print("Step 4: Calibrating settings (target='resources' for speed)...")
ezkl.calibrate_settings(
    data='input.json',
    model='network.onnx',
    settings='settings_new.json',
    target='resources'  # Optimize for speed/memory
)
print("  ✓ Calibration complete")
print()

# Read new settings to show improvements
with open(f"{CLASSIFIER_DIR}/settings_new.json") as f:
    new_settings = json.load(f)

print("Settings Comparison:")
print(f"  Old logrows: {current_settings['run_args']['logrows']} → New logrows: {new_settings['run_args']['logrows']}")
print(f"  Old assignments: {current_settings['total_assignments']:,} → New assignments: {new_settings['total_assignments']:,}")
reduction = (current_settings['total_assignments'] - new_settings['total_assignments']) / current_settings['total_assignments'] * 100
print(f"  Reduction: {reduction:.1f}%")
print()

# Compile new circuit
print("Step 5: Compiling new classifier circuit...")
ezkl.compile_circuit(
    model='network.onnx',
    compiled_circuit='network_logrows18.ezkl',
    settings_path='settings_new.json'
)
print("  ✓ Circuit compiled: network_logrows18.ezkl")
print()

# Backup old keys
print("Step 6: Backing up old keys...")
os.system(f"mv {CLASSIFIER_DIR}/pk.key {CLASSIFIER_DIR}/pk_logrows23.key")
os.system(f"mv {CLASSIFIER_DIR}/vk.key {CLASSIFIER_DIR}/vk_logrows23.key")
print("  ✓ Old keys backed up as pk_logrows23.key, vk_logrows23.key")
print()

# Generate new keys
print("Step 7: Generating new keys (expected: 2-3 minutes)...")
print("  This will be much faster than the previous 7 minutes!")
import time
start_time = time.time()

ezkl.setup(
    compiled_circuit='network_logrows18.ezkl',
    srs_path='kzg.srs',
    vk_path='vk_new.key',
    pk_path='pk_new.key'
)

setup_time = time.time() - start_time
print(f"  ✓ Keys generated in {setup_time/60:.1f} minutes")
print()

# Check new key sizes
import subprocess
pk_size = subprocess.check_output(['du', '-h', f'{CLASSIFIER_DIR}/pk_new.key']).split()[0].decode()
vk_size = subprocess.check_output(['du', '-h', f'{CLASSIFIER_DIR}/vk_new.key']).split()[0].decode()

print("Key Size Comparison:")
print(f"  Old PK: 72GB → New PK: {pk_size}")
print(f"  Old VK: 20MB → New VK: {vk_size}")
print()

# Move new files into place
print("Step 8: Installing new circuit and keys...")
os.system(f"mv {CLASSIFIER_DIR}/settings_new.json {CLASSIFIER_DIR}/settings.json")
os.system(f"mv {CLASSIFIER_DIR}/network_logrows18.ezkl {CLASSIFIER_DIR}/network.ezkl")
os.system(f"mv {CLASSIFIER_DIR}/pk_new.key {CLASSIFIER_DIR}/pk.key")
os.system(f"mv {CLASSIFIER_DIR}/vk_new.key {CLASSIFIER_DIR}/vk.key")
print("  ✓ New files installed")
print()

print("=" * 70)
print("CLASSIFIER OPTIMIZATION COMPLETE!")
print("=" * 70)
print()
print("Next steps:")
print("  1. GAN proof: Already complete (use existing proof.json)")
print("  2. Run classifier proof generation (expected: 6-12 minutes)")
print("  3. Link proofs via swap_proof_commitments()")
print()
print("To test the optimized classifier:")
print(f"  cd {CLASSIFIER_DIR}")
print("  /root/.ezkl/ezkl gen-witness --data input_from_gan.json \\")
print("    --compiled-circuit network.ezkl --output witness_from_gan.json \\")
print("    --vk-path vk.key --srs-path kzg.srs")
print()
print("  /usr/bin/time -v /root/.ezkl/ezkl prove \\")
print("    --compiled-circuit network.ezkl --pk-path pk.key \\")
print("    --proof-path proof_test.json --srs-path kzg.srs \\")
print("    --witness witness_from_gan.json")
print()
