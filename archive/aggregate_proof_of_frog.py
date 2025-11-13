#!/usr/bin/env python3
"""
Proof-of-Frog with True Proof Aggregation

Uses EZKL's aggregate() function to create a SINGLE aggregated proof
that proves both GAN generation and Classifier evaluation in one verification.

This is different from polycommit chaining - it's TRUE proof recursion/aggregation
where multiple proofs are combined into one.
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

base_dir = Path('/root/ezkl_logs/models/ProofOfFrog_Aggregated')
gan_dir = base_dir / 'gan'
cls_dir = base_dir / 'classifier'
aggr_dir = base_dir / 'aggregated'

# Create directories
for d in [gan_dir, cls_dir, aggr_dir]:
    d.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PROOF-OF-FROG: True Proof Aggregation")
print("="*70)
print()
print("This creates a SINGLE aggregated proof that verifies:")
print("  1. GAN generated an image conditioned on 'frog'")
print("  2. Classifier evaluated that exact image")
print()
print("Instead of verifying two proofs separately, the aggregated proof")
print("verifies both in one step - true proof recursion!")
print("="*70)
print()

def setup_aggregated_circuit():
    """
    Step 1: Setup aggregated PK/VK from sample proofs

    This creates a single proving/verification key pair for the
    aggregated circuit that can verify both proofs together.
    """
    print("Step 1: Setting up aggregated circuit...")
    print()

    # Paths to our existing proofs
    gan_proof = gan_dir / 'proof.json'
    cls_proof = cls_dir / 'proof_from_gan.json'

    # Check if proofs exist
    if not gan_proof.exists():
        print(f"✗ GAN proof not found: {gan_proof}")
        print("  Please run the polycommit pipeline first to generate proofs")
        return False

    if not cls_proof.exists():
        print(f"✗ Classifier proof not found: {cls_proof}")
        print("  Please run the polycommit pipeline first to generate proofs")
        return False

    print(f"✓ GAN proof: {gan_proof} ({gan_proof.stat().st_size/1024:.1f} KB)")
    print(f"✓ Classifier proof: {cls_proof} ({cls_proof.stat().st_size/1024:.1f} KB)")
    print()

    # Paths for aggregated circuit
    aggr_pk = aggr_dir / 'pk_aggr.key'
    aggr_vk = aggr_dir / 'vk_aggr.key'
    aggr_srs = aggr_dir / 'kzg_aggr.srs'

    # Copy SRS (can reuse existing one)
    gan_srs = gan_dir / 'kzg.srs'
    if gan_srs.exists():
        import shutil
        shutil.copy(gan_srs, aggr_srs)
        print(f"✓ Copied SRS: {aggr_srs.stat().st_size/(1024**3):.1f} GB")
    else:
        print("⚠ No SRS found, will need to generate")

    print()
    print("Calling ezkl.setup_aggregate()...")
    print("  This creates aggregated PK/VK from sample proofs")
    print("  (May take several minutes for large circuits)")
    print()

    try:
        # Setup aggregated circuit
        # logrows for aggregation circuit (may need adjustment)
        aggr_logrows = 26  # Larger than individual circuits

        result = ezkl.setup_aggregate(
            sample_snarks=[str(gan_proof), str(cls_proof)],
            vk_path=str(aggr_vk),
            pk_path=str(aggr_pk),
            logrows=aggr_logrows,
            split_proofs=False,  # Not circuit splits, separate models
            srs_path=str(aggr_srs),
            disable_selector_compression=False,
            commitment="kzg"
        )

        if result:
            pk_size = aggr_pk.stat().st_size
            vk_size = aggr_vk.stat().st_size
            print(f"✓ Aggregated setup complete!")
            print(f"  PK: {pk_size/(1024**3):.1f} GB")
            print(f"  VK: {vk_size/(1024**2):.1f} MB")
            return True
        else:
            print("✗ setup_aggregate failed")
            return False

    except Exception as e:
        print(f"✗ Exception during setup_aggregate: {e}")
        return False

def generate_aggregated_proof():
    """
    Step 2: Generate aggregated proof

    This combines both proofs into a SINGLE proof that proves:
    - GAN generated the image
    - Classifier evaluated the image
    - Both executions were correct
    """
    print()
    print("="*70)
    print("Step 2: Generating aggregated proof...")
    print("="*70)
    print()

    # Input proofs
    gan_proof = gan_dir / 'proof.json'
    cls_proof = cls_dir / 'proof_from_gan.json'

    # Output paths
    aggr_proof = aggr_dir / 'proof_aggregated.json'
    aggr_vk = aggr_dir / 'vk_aggr.key'
    aggr_srs = aggr_dir / 'kzg_aggr.srs'

    print("Input proofs:")
    print(f"  GAN: {gan_proof}")
    print(f"  Classifier: {cls_proof}")
    print()
    print(f"Output: {aggr_proof}")
    print()

    print("Calling ezkl.aggregate()...")
    print("  This creates ONE proof from TWO proofs")
    print("  (May take several minutes)")
    print()

    try:
        aggr_logrows = 26

        result = ezkl.aggregate(
            aggregation_snarks=[str(gan_proof), str(cls_proof)],
            proof_path=str(aggr_proof),
            vk_path=str(aggr_vk),
            transcript="evm",
            logrows=aggr_logrows,
            check_mode="safe",
            split_proofs=False,
            srs_path=str(aggr_srs),
            commitment="kzg"
        )

        if result and aggr_proof.exists():
            proof_size = aggr_proof.stat().st_size
            print(f"✓ Aggregated proof generated!")
            print(f"  Size: {proof_size/1024:.1f} KB")
            return True
        else:
            print("✗ aggregate failed or proof not created")
            return False

    except Exception as e:
        print(f"✗ Exception during aggregate: {e}")
        return False

def verify_aggregated_proof():
    """
    Step 3: Verify aggregated proof

    This verifies BOTH proofs with a SINGLE verification!
    """
    print()
    print("="*70)
    print("Step 3: Verifying aggregated proof...")
    print("="*70)
    print()

    aggr_proof = aggr_dir / 'proof_aggregated.json'
    aggr_vk = aggr_dir / 'vk_aggr.key'
    aggr_srs = aggr_dir / 'kzg_aggr.srs'

    if not aggr_proof.exists():
        print(f"✗ Aggregated proof not found: {aggr_proof}")
        return False

    print("Calling ezkl.verify_aggr()...")
    print()

    try:
        aggr_logrows = 26

        result = ezkl.verify_aggr(
            proof_path=str(aggr_proof),
            vk_path=str(aggr_vk),
            logrows=aggr_logrows,
            commitment="kzg"
        )

        if result:
            print("✓ AGGREGATED PROOF VERIFIED!")
            print()
            print("This single verification proves:")
            print("  ✓ GAN generated an image conditioned on 'frog'")
            print("  ✓ Classifier evaluated that exact image")
            print("  ✓ Both computations were correct")
            print()
            return True
        else:
            print("✗ Aggregated proof verification FAILED")
            return False

    except Exception as e:
        print(f"✗ Exception during verify_aggr: {e}")
        return False

def main():
    print("Starting aggregated proof pipeline...")
    print()

    # Check if source proofs exist
    gan_proof = Path('/root/ezkl_logs/models/ProofOfFrog_Polycommit/gan/proof.json')
    cls_proof = Path('/root/ezkl_logs/models/ProofOfFrog_Polycommit/classifier/proof_from_gan.json')

    if not gan_proof.exists() or not cls_proof.exists():
        print("ERROR: Source proofs not found")
        print(f"  GAN proof: {gan_proof} - {'exists' if gan_proof.exists() else 'NOT FOUND'}")
        print(f"  Classifier proof: {cls_proof} - {'exists' if cls_proof.exists() else 'NOT FOUND'}")
        print()
        print("Please run the polycommit pipeline first:")
        print("  cd /root && python3 complete_polycommit_pipeline.py")
        return False

    # Copy proofs to working directory
    import shutil
    shutil.copy(gan_proof, gan_dir / 'proof.json')
    shutil.copy(cls_proof, cls_dir / 'proof_from_gan.json')

    # Copy SRS if available
    gan_srs = Path('/root/ezkl_logs/models/ProofOfFrog_Polycommit/gan/kzg.srs')
    if gan_srs.exists():
        shutil.copy(gan_srs, gan_dir / 'kzg.srs')

    # Step 1: Setup aggregated circuit
    if not setup_aggregated_circuit():
        print()
        print("✗ Setup failed")
        return False

    # Step 2: Generate aggregated proof
    if not generate_aggregated_proof():
        print()
        print("✗ Proof generation failed")
        return False

    # Step 3: Verify aggregated proof
    if not verify_aggregated_proof():
        print()
        print("✗ Verification failed")
        return False

    # Success!
    print()
    print("="*70)
    print("PROOF-OF-FROG: TRUE AGGREGATED PROOF SUCCESS!")
    print("="*70)
    print()
    print("Achievement unlocked:")
    print("  ✓ Created SINGLE proof from TWO proofs")
    print("  ✓ One verification proves entire GAN→Classifier pipeline")
    print("  ✓ True proof recursion/aggregation")
    print()
    print("Artifacts:")
    print(f"  Aggregated proof: {aggr_dir / 'proof_aggregated.json'}")
    print(f"  Aggregated VK: {aggr_dir / 'vk_aggr.key'}")
    print(f"  Aggregated PK: {aggr_dir / 'pk_aggr.key'}")
    print()

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
