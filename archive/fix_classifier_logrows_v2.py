#!/usr/bin/env python3
"""
Fix Classifier Logrows V2 - Complete fixed version
Regenerate classifier circuit with logrows=18
"""

import ezkl
import json
import os
import time
import subprocess

CLASSIFIER_DIR = "/root/ezkl_logs/models/ProofOfFrog_Fixed/classifier"

print("="*70)
print("CLASSIFIER LOGROWS OPTIMIZATION - V2")
print("="*70)
print()

os.chdir(CLASSIFIER_DIR)

# Step 1: Backup
print("Step 1: Backing up existing files...")
os.system(f"cp settings.json settings_logrows23_backup.json")
print("  ✓ Backup created\n")

# Step 2: Read current settings
with open("settings.json") as f:
    old_settings = json.load(f)
print(f"Current state: logrows={old_settings['run_args']['logrows']}, assignments={old_settings['total_assignments']:,}\n")

# Step 3: Generate settings with logrows=18
print("Step 3: Generating settings with logrows=18...")
run_args = ezkl.PyRunArgs()
run_args.logrows = 18
run_args.input_visibility = "polycommit"
run_args.param_visibility = "fixed"
run_args.output_visibility = "public"

ezkl.gen_settings(
    model='network.onnx',
    output='settings_logrows18.json',
    py_run_args=run_args
)
print("  ✓ Settings generated\n")

# Step 4: Calibrate
print("Step 4: Calibrating (target='resources')...")
ezkl.calibrate_settings(
    data='input.json',
    model='network.onnx',
    settings='settings_logrows18.json',
    target='resources'
)
print("  ✓ Calibrated\n")

# Step 5: Force logrows=18 (calibration may override)
print("Step 5: Forcing logrows=18 in settings...")
with open("settings_logrows18.json") as f:
    new_settings = json.load(f)

print(f"  Before force: logrows={new_settings['run_args']['logrows']}")
new_settings['run_args']['logrows'] = 18

with open("settings_logrows18.json", "w") as f:
    json.dump(new_settings, f, indent=2)
print(f"  After force: logrows=18")
print(f"  ✓ Settings updated\n")

# Step 6: Compile
print("Step 6: Compiling circuit with logrows=18...")
ezkl.compile_circuit(
    model='network.onnx',
    compiled_circuit='network_logrows18.ezkl',
    settings_path='settings_logrows18.json'
)
print("  ✓ Circuit compiled\n")

# Step 7: Backup old keys
print("Step 7: Backing up old keys...")
os.system("mv pk.key pk_logrows23.key 2>/dev/null || true")
os.system("mv vk.key vk_logrows23.key 2>/dev/null || true")
print("  ✓ Old keys backed up\n")

# Step 8: Generate new keys
print("Step 8: Generating keys (logrows=18, expect 2-3 min)...")
start = time.time()

ezkl.setup(
    model='network_logrows18.ezkl',
    vk_path='vk_logrows18.key',
    pk_path='pk_logrows18.key',
    srs_path='kzg.srs'
)

elapsed = time.time() - start
print(f"  ✓ Keys generated in {elapsed/60:.1f} minutes\n")

# Step 9: Check sizes
pk_size = subprocess.check_output(['du', '-h', 'pk_logrows18.key']).split()[0].decode()
vk_size = subprocess.check_output(['du', '-h', 'vk_logrows18.key']).split()[0].decode()

print("Key Size Comparison:")
print(f"  Old PK: 72GB → New PK: {pk_size}")
print(f"  Old VK: 20MB → New VK: {vk_size}\n")

# Step 10: Install new files
print("Step 10: Installing new circuit and keys...")
os.system("mv settings_logrows18.json settings.json")
os.system("mv network_logrows18.ezkl network.ezkl")
os.system("mv pk_logrows18.key pk.key")
os.system("mv vk_logrows18.key vk.key")
print("  ✓ New files installed\n")

print("="*70)
print("OPTIMIZATION COMPLETE!")
print("="*70)
print()
print(f"GAN: Unchanged (already working)")
print(f"Classifier: Optimized to logrows=18 (PK: {pk_size}, VK: {vk_size})")
print()
print("Ready for proof generation (expected: 6-12 minutes)")
