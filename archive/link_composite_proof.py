#!/usr/bin/env python3
"""
Link GAN and Classifier Proofs using swap_proof_commitments

This creates a cryptographically-linked composite proof where:
1. GAN generates image with output_visibility = "KZGCommit" (private)
2. Classifier processes image with input_visibility = "KZGCommit"
3. swap_proof_commitments() links them via KZG commitments

After swapping, the classifier proof's input commitment will match the GAN's
output commitment, proving they're operating on the same image without revealing it.
"""

import json
import os
import sys
from pathlib import Path

# Try to import ezkl
try:
    import ezkl
except ImportError:
    print("ERROR: ezkl module not available locally")
    print("This script must be run on the server with ezkl installed")
    sys.exit(1)

base_dir = Path('/root/ezkl_logs/models/ProofOfFrog_Polycommit')
gan_dir = base_dir / 'gan'
cls_dir = base_dir / 'classifier'

print("="*70)
print("PROOF-OF-FROG: Commitment Swapping for Composite Proof")
print("="*70)
print()

# Step 1: Verify files exist
print("Step 1: Checking required files...")
required_files = {
    'GAN proof': gan_dir / 'proof.json',
    'GAN witness': gan_dir / 'witness.json',
    'Classifier proof': cls_dir / 'proof_from_gan.json',
    'Classifier witness': cls_dir / 'witness_from_gan.json'
}

for name, path in required_files.items():
    if not path.exists():
        print(f"✗ {name} not found: {path}")
        sys.exit(1)
    size = path.stat().st_size
    print(f"✓ {name}: {size/1024:.1f} KB")

print()

# Step 2: Load witnesses
print("Step 2: Loading witnesses...")
with open(gan_dir / 'witness.json', 'r') as f:
    gan_witness = json.load(f)

with open(cls_dir / 'witness_from_gan.json', 'r') as f:
    cls_witness = json.load(f)

print(f"  GAN witness outputs: {len(gan_witness.get('outputs', [[]])[0])} values")
print(f"  Classifier witness inputs: {len(cls_witness.get('inputs', [[]])[0])} values")
print()

# Step 3: Verify data consistency
print("Step 3: Verifying witness data consistency...")
gan_output = gan_witness['outputs'][0]
cls_input = cls_witness['inputs'][0]

if gan_output == cls_input:
    print("✓ GAN output matches Classifier input (3072 values)")
else:
    print("✗ WARNING: GAN output does NOT match Classifier input!")
    print(f"  GAN output: {len(gan_output)} values")
    print(f"  Classifier input: {len(cls_input)} values")

print()

# Step 4: Swap proof commitments
print("Step 4: Swapping proof commitments...")
print("  This links the proofs cryptographically via KZG commitments")
print()

# According to EZKL docs, we need to:
# 1. Update the classifier witness to reference GAN's processed outputs
# 2. Call swap_proof_commitments to update the proof

# Check if 'processed_outputs' exists in GAN witness
if 'processed_outputs' in gan_witness:
    print("  Found processed_outputs in GAN witness")
    cls_witness['processed_inputs'] = gan_witness['processed_outputs']

    # Save updated witness
    cls_witness_updated_path = cls_dir / 'witness_from_gan_linked.json'
    with open(cls_witness_updated_path, 'w') as f:
        json.dump(cls_witness, f, indent=2)
    print(f"✓ Updated classifier witness saved: {cls_witness_updated_path}")

    # Now swap commitments in the proof
    cls_proof_path = cls_dir / 'proof_from_gan.json'
    cls_proof_linked_path = cls_dir / 'proof_from_gan_linked.json'

    # Make a copy first
    with open(cls_proof_path, 'r') as f:
        proof_data = f.read()
    with open(cls_proof_linked_path, 'w') as f:
        f.write(proof_data)

    print(f"  Calling ezkl.swap_proof_commitments()...")
    try:
        ezkl.swap_proof_commitments(
            str(cls_proof_linked_path),
            str(cls_witness_updated_path)
        )
        print("✓ Commitments swapped successfully!")
    except Exception as e:
        print(f"✗ swap_proof_commitments failed: {e}")
        sys.exit(1)

else:
    print("✗ WARNING: No 'processed_outputs' found in GAN witness")
    print("  This may indicate polycommit wasn't properly configured")
    print()
    print("  GAN witness keys:", list(gan_witness.keys()))
    print("  Classifier witness keys:", list(cls_witness.keys()))
    print()
    print("  Checking for KZG commitment data...")

    # Look for commitment data in different locations
    if 'instances' in gan_witness:
        print(f"  GAN instances: {gan_witness['instances']}")

    print()
    print("  NOTE: May need to regenerate proofs with correct polycommit settings")

print()

# Step 5: Summary
print("="*70)
print("COMPOSITE PROOF LINKING: Status")
print("="*70)
print()

if 'processed_outputs' in gan_witness:
    print("✓ SUCCESS: Proofs are now cryptographically linked!")
    print()
    print("Verification workflow:")
    print("  1. Verify GAN proof with gan/proof.json")
    print("  2. Verify Classifier proof with classifier/proof_from_gan_linked.json")
    print("  3. KZG commitments ensure same image was used")
    print()
    print("Artifacts:")
    print(f"  GAN proof: {gan_dir / 'proof.json'}")
    print(f"  Classifier proof (linked): {cls_proof_linked_path}")
    print(f"  GAN witness: {gan_dir / 'witness.json'}")
    print(f"  Classifier witness (linked): {cls_witness_updated_path}")
else:
    print("⚠ INCOMPLETE: Commitments not swapped")
    print()
    print("Possible issues:")
    print("  - Polycommit visibility may not be properly configured")
    print("  - Proofs may need regeneration with correct settings")
    print("  - EZKL version may not support processed_outputs/inputs")
    print()
    print("Current configuration:")
    print("  GAN: output_visibility should be 'KZGCommit'")
    print("  Classifier: input_visibility should be 'KZGCommit'")

print()
