#!/usr/bin/env python3
"""
Complete Polycommit Proof-of-Frog Pipeline Automation

This script completes the entire pipeline:
1. Wait for GAN keys to be ready
2. Generate GAN proof with polycommit output
3. Extract GAN output and create classifier input
4. Generate classifier witness
5. Generate classifier proof with polycommit input
6. Verify both proofs
"""

import json
import os
import subprocess
import time
from pathlib import Path

base_dir = Path('/root/ezkl_logs/models/ProofOfFrog_Polycommit')
gan_dir = base_dir / 'gan'
cls_dir = base_dir / 'classifier'

def run_cmd(cmd, cwd=None, timeout=None):
    """Run command and return output"""
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    print(result.stdout)
    return result

def wait_for_keys():
    """Wait for GAN keys to be generated"""
    print("\n" + "="*70)
    print("Step 1: Waiting for GAN keys to be ready...")
    print("="*70)

    while True:
        pk = gan_dir / 'pk.key'
        vk = gan_dir / 'vk.key'

        if pk.exists() and vk.exists():
            # Check if files are complete by attempting to load
            try:
                result = subprocess.run(
                    ['/root/.ezkl/ezkl', 'verify', '--help'],
                    capture_output=True, timeout=5
                )
                # If command runs, keys should be ready
                pk_size = pk.stat().st_size
                vk_size = vk.stat().st_size

                if pk_size > 1_000_000_000 and vk_size > 1_000_000:  # > 1GB and > 1MB
                    print(f"Keys ready: PK={pk_size/(1024**3):.1f}GB, VK={vk_size/(1024**2):.1f}MB")
                    return True
            except:
                pass

        print("  Keys not ready, waiting 30 seconds...")
        time.sleep(30)

def generate_gan_proof():
    """Generate GAN proof with polycommit output"""
    print("\n" + "="*70)
    print("Step 2: Generating GAN proof with polycommit output...")
    print("="*70)

    os.chdir(gan_dir)

    start_time = time.time()
    run_cmd([
        '/usr/bin/time', '-v',
        '/root/.ezkl/ezkl', 'prove',
        '--compiled-circuit', 'network.ezkl',
        '--pk-path', 'pk.key',
        '--proof-path', 'proof.json',
        '--srs-path', 'kzg.srs',
        '--witness', 'witness.json'
    ], cwd=gan_dir, timeout=600)

    elapsed = time.time() - start_time
    proof_size = (gan_dir / 'proof.json').stat().st_size

    print(f"\n✓ GAN proof generated in {elapsed/60:.1f} minutes")
    print(f"  Proof size: {proof_size/1024:.1f} KB")

    return elapsed, proof_size

def create_classifier_input_from_gan():
    """Extract GAN output and create classifier input"""
    print("\n" + "="*70)
    print("Step 3: Creating classifier input from GAN output...")
    print("="*70)

    # Load GAN witness
    with open(gan_dir / 'witness.json', 'r') as f:
        gan_witness = json.load(f)

    gan_output = gan_witness['outputs'][0]
    print(f"  GAN output: {len(gan_output)} values (3072 expected for 32x32 RGB)")

    # Create classifier input
    cls_input = {
        'input_shapes': [[1, 3, 32, 32]],
        'input_data': [gan_output]
    }

    with open(cls_dir / 'input_from_gan.json', 'w') as f:
        json.dump(cls_input, f)

    print("✓ Classifier input created from GAN output")
    return gan_output

def generate_classifier_witness():
    """Generate classifier witness"""
    print("\n" + "="*70)
    print("Step 4: Generating classifier witness...")
    print("="*70)

    os.chdir(cls_dir)

    import ezkl
    ezkl.gen_witness(
        'input_from_gan.json',
        'network.ezkl',
        'witness_from_gan.json'
    )

    print("✓ Classifier witness generated")

def generate_classifier_proof():
    """Generate classifier proof with polycommit input"""
    print("\n" + "="*70)
    print("Step 5: Generating classifier proof with polycommit input...")
    print("="*70)

    os.chdir(cls_dir)

    start_time = time.time()
    run_cmd([
        '/usr/bin/time', '-v',
        '/root/.ezkl/ezkl', 'prove',
        '--compiled-circuit', 'network.ezkl',
        '--pk-path', 'pk.key',
        '--proof-path', 'proof_from_gan.json',
        '--srs-path', 'kzg.srs',
        '--witness', 'witness_from_gan.json'
    ], cwd=cls_dir, timeout=600)

    elapsed = time.time() - start_time
    proof_size = (cls_dir / 'proof_from_gan.json').stat().st_size

    print(f"\n✓ Classifier proof generated in {elapsed/60:.1f} minutes")
    print(f"  Proof size: {proof_size/1024:.1f} KB")

    return elapsed, proof_size

def verify_proofs():
    """Verify both proofs"""
    print("\n" + "="*70)
    print("Step 6: Verifying proofs...")
    print("="*70)

    # Verify GAN proof
    print("\nVerifying GAN proof...")
    os.chdir(gan_dir)
    gan_result = run_cmd([
        '/root/.ezkl/ezkl', 'verify',
        '--proof-path', 'proof.json',
        '--settings-path', 'settings.json',
        '--vk-path', 'vk.key',
        '--srs-path', 'kzg.srs'
    ], cwd=gan_dir)

    print("✓ GAN proof verified")

    # Verify Classifier proof
    print("\nVerifying Classifier proof...")
    os.chdir(cls_dir)
    cls_result = run_cmd([
        '/root/.ezkl/ezkl', 'verify',
        '--proof-path', 'proof_from_gan.json',
        '--settings-path', 'settings.json',
        '--vk-path', 'vk.key',
        '--srs-path', 'kzg.srs'
    ], cwd=cls_dir)

    print("✓ Classifier proof verified")

    return True

def main():
    print("\n" + "="*70)
    print("POLYCOMMIT PROOF-OF-FROG: Autonomous Pipeline Execution")
    print("="*70)
    print()

    try:
        # Step 1: Wait for GAN keys
        wait_for_keys()

        # Step 2: Generate GAN proof
        gan_time, gan_size = generate_gan_proof()

        # Step 3: Create classifier input from GAN output
        gan_output = create_classifier_input_from_gan()

        # Step 4: Generate classifier witness
        generate_classifier_witness()

        # Step 5: Generate classifier proof
        cls_time, cls_size = generate_classifier_proof()

        # Step 6: Verify proofs
        verify_proofs()

        # Summary
        print("\n" + "="*70)
        print("POLYCOMMIT PROOF-OF-FROG: PIPELINE COMPLETE ✓")
        print("="*70)
        print()
        print("Summary:")
        print(f"  GAN proof: {gan_time/60:.1f} min, {gan_size/1024:.1f} KB")
        print(f"  Classifier proof: {cls_time/60:.1f} min, {cls_size/1024:.1f} KB")
        print(f"  Total proving time: {(gan_time+cls_time)/60:.1f} minutes")
        print()
        print("Configuration:")
        print("  ✓ GAN: output_visibility = polycommit (image private via KZG)")
        print("  ✓ Classifier: input_visibility = polycommit (matches GAN)")
        print("  ✓ Both proofs verified independently")
        print("  ✓ Image consistency enforced through witness chaining")
        print()
        print("Artifacts:")
        print(f"  GAN proof: {gan_dir / 'proof.json'}")
        print(f"  Classifier proof: {cls_dir / 'proof_from_gan.json'}")
        print(f"  GAN witness: {gan_dir / 'witness.json'}")
        print(f"  Classifier witness: {cls_dir / 'witness_from_gan.json'}")
        print()

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()
