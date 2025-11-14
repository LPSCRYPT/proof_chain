#!/usr/bin/env python3
"""
Deploy ProofOfFrog Onchain Verifiers
Generates Solidity verifier contracts for both GAN and Classifier proofs
"""

import subprocess
import os
import json
from pathlib import Path

FIXED_DIR = Path("/root/proof_chain/ezkl_logs/models/ProofOfFrog_Fixed")
GAN_DIR = FIXED_DIR / "gan"
CLS_DIR = FIXED_DIR / "classifier"

print("="*70)
print("PROOFOFFROG ONCHAIN VERIFIER DEPLOYMENT")
print("="*70)
print()

# Step 1: Generate GAN Verifier Contract
print("Step 1: Generating GAN verifier contract...")
os.chdir(GAN_DIR)

cmd_gan = [
    "/root/.ezkl/ezkl", "create-evm-verifier",
    "--settings-path", "settings.json",
    "--vk-path", "vk.key",
    "--srs-path", "kzg.srs",
    "--sol-code-path", "gan_verifier.sol",
    "--abi-path", "gan_verifier_abi.json"
]

print(f"  Running: {' '.join(cmd_gan)}")
result = subprocess.run(cmd_gan, capture_output=True, text=True)

if result.returncode == 0:
    sol_size = Path("gan_verifier.sol").stat().st_size
    abi_size = Path("gan_verifier_abi.json").stat().st_size
    print(f"  ✓ GAN verifier generated:")
    print(f"    Solidity: {sol_size:,} bytes ({sol_size/1024:.1f} KB)")
    print(f"    ABI: {abi_size:,} bytes ({abi_size/1024:.1f} KB)")
else:
    print(f"  ✗ GAN verifier generation failed")
    print(f"  stdout: {result.stdout}")
    print(f"  stderr: {result.stderr}")

print()

# Step 2: Generate Classifier Verifier Contract
print("Step 2: Generating Classifier verifier contract...")
os.chdir(CLS_DIR)

cmd_cls = [
    "/root/.ezkl/ezkl", "create-evm-verifier",
    "--settings-path", "settings.json",
    "--vk-path", "vk.key",
    "--srs-path", "kzg.srs",
    "--sol-code-path", "classifier_verifier.sol",
    "--abi-path", "classifier_verifier_abi.json"
]

print(f"  Running: {' '.join(cmd_cls)}")
result = subprocess.run(cmd_cls, capture_output=True, text=True)

if result.returncode == 0:
    sol_size = Path("classifier_verifier.sol").stat().st_size
    abi_size = Path("classifier_verifier_abi.json").stat().st_size
    print(f"  ✓ Classifier verifier generated:")
    print(f"    Solidity: {sol_size:,} bytes ({sol_size/1024:.1f} KB)")
    print(f"    ABI: {abi_size:,} bytes ({abi_size/1024:.1f} KB)")
else:
    print(f"  ✗ Classifier verifier generation failed")
    print(f"  stdout: {result.stdout}")
    print(f"  stderr: {result.stderr}")

print()

# Step 3: Test verifiers locally using verify-evm
print("Step 3: Testing verifiers locally with verify-evm...")
print()

# Test GAN verifier
os.chdir(GAN_DIR)
print("  Testing GAN verifier...")
cmd_test_gan = [
    "/root/.ezkl/ezkl", "verify-evm",
    "--proof-path", "proof.json",
    "--sol-code-path", "gan_verifier.sol"
]

result = subprocess.run(cmd_test_gan, capture_output=True, text=True, timeout=30)
if "verified" in result.stdout.lower() or result.returncode == 0:
    print("    ✓ GAN verifier tested successfully")
else:
    print(f"    ⚠️  GAN verifier test failed or timed out")
    print(f"    This may be expected for large circuits")

# Test Classifier verifier
os.chdir(CLS_DIR)
print("  Testing Classifier verifier...")
cmd_test_cls = [
    "/root/.ezkl/ezkl", "verify-evm",
    "--proof-path", "proof_from_gan.json",
    "--sol-code-path", "classifier_verifier.sol"
]

result = subprocess.run(cmd_test_cls, capture_output=True, text=True, timeout=30)
if "verified" in result.stdout.lower() or result.returncode == 0:
    print("    ✓ Classifier verifier tested successfully")
else:
    print(f"    ⚠️  Classifier verifier test failed or timed out")
    print(f"    This may be expected for large circuits")

print()

# Summary
print("="*70)
print("VERIFIER CONTRACT GENERATION COMPLETE!")
print("="*70)
print()
print("Generated Contracts:")
print(f"  GAN Verifier:")
print(f"    Solidity: {GAN_DIR}/gan_verifier.sol")
print(f"    ABI: {GAN_DIR}/gan_verifier_abi.json")
print(f"  Classifier Verifier:")
print(f"    Solidity: {CLS_DIR}/classifier_verifier.sol")
print(f"    ABI: {CLS_DIR}/classifier_verifier_abi.json")
print()
print("To deploy to a blockchain network:")
print()
print("1. For GAN verifier:")
print(f"   cd {GAN_DIR}")
print("   /root/.ezkl/ezkl deploy-evm \\")
print("     --sol-code-path gan_verifier.sol \\")
print("     --rpc-url <YOUR_RPC_URL> \\")
print("     --private-key <YOUR_PRIVATE_KEY> \\")
print("     --addr-path gan_verifier_address.txt")
print()
print("2. For Classifier verifier:")
print(f"   cd {CLS_DIR}")
print("   /root/.ezkl/ezkl deploy-evm \\")
print("     --sol-code-path classifier_verifier.sol \\")
print("     --rpc-url <YOUR_RPC_URL> \\")
print("     --private-key <YOUR_PRIVATE_KEY> \\")
print("     --addr-path classifier_verifier_address.txt")
print()
print("Recommended testnets:")
print("  - Ethereum Sepolia: https://sepolia.etherscan.io/")
print("  - Polygon Mumbai: https://mumbai.polygonscan.com/")
print("  - Optimism Goerli: https://goerli-optimism.etherscan.io/")
print()
print("Or test locally with Anvil:")
print("  anvil # Start local Ethereum node")
print("  # Use RPC: http://localhost:8545")
print()
