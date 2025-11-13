#!/usr/bin/env python3
"""
Complete Classifier Key Generation (After Disk Space Fix)
The circuit and settings are already generated, just need keys
"""

import ezkl
import os
import time
import subprocess

CLASSIFIER_DIR = "/root/ezkl_logs/models/ProofOfFrog_Fixed/classifier"

print("="*70)
print("CLASSIFIER KEY GENERATION - Completing Optimization")
print("="*70)
print()

os.chdir(CLASSIFIER_DIR)

# Verify files exist
print("Verifying optimized circuit files...")
required_files = ['settings_logrows18.json', 'network_logrows18.ezkl', 'kzg.srs']
for f in required_files:
    if not os.path.exists(f):
        print(f"  ✗ ERROR: Missing {f}")
        exit(1)
    size = subprocess.check_output(['du', '-h', f]).split()[0].decode()
    print(f"  ✓ {f}: {size}")
print()

# Generate keys
print("Generating keys for logrows=18 circuit (expected: 2-3 minutes)...")
start = time.time()

ezkl.setup(
    model='network_logrows18.ezkl',
    vk_path='vk_logrows18.key',
    pk_path='pk_logrows18.key',
    srs_path='kzg.srs'
)

elapsed = time.time() - start
print(f"  ✓ Keys generated in {elapsed/60:.1f} minutes")
print()

# Check sizes
pk_size = subprocess.check_output(['du', '-h', 'pk_logrows18.key']).split()[0].decode()
vk_size = subprocess.check_output(['du', '-h', 'vk_logrows18.key']).split()[0].decode()

print("Key Size Comparison:")
print(f"  Old PK: 72GB → New PK: {pk_size}")
print(f"  Old VK: 20MB → New VK: {vk_size}")
print()

# Install new files
print("Installing optimized circuit and keys...")
os.system("mv settings_logrows18.json settings.json")
os.system("mv network_logrows18.ezkl network.ezkl")
os.system("mv pk_logrows18.key pk.key")
os.system("mv vk_logrows18.key vk.key")
print("  ✓ New files installed")
print()

print("="*70)
print("CLASSIFIER OPTIMIZATION COMPLETE!")
print("="*70)
print()
print(f"GAN: Unchanged (logrows=24, working fine)")
print(f"Classifier: Optimized to logrows=18 (PK: {pk_size}, VK: {vk_size})")
print()
print("Ready for proof generation (expected: 6-12 minutes)")
print()
print("Disk usage:")
os.system("df -h / | grep overlay")
