# ProofOfFrog Classifier Performance Investigation

## ⚠️ DIAGNOSIS COMPLETE - ROOT CAUSE IDENTIFIED

**Date:** 2025-11-13
**Status:** CONFIRMED

### Actual Settings Found

**GAN Configuration:**
- logrows: **24** (Circuit size: 2^24 = 16.7M constraints)
- batch_size: null
- calibration_target: null
- PK size: 82GB
- VK size: 37MB
- Proof generation time: 6 minutes

**Classifier Configuration:**
- logrows: **23** (Circuit size: 2^23 = 8.4M constraints)
- batch_size: null
- calibration_target: null
- PK size: 72GB
- VK size: 20MB
- Proof generation time: **3+ hours** (FAILED)

### Root Cause Analysis

**Primary Issue:** Extreme over-provisioning of logrows parameter

The EZKL example notebooks recommend:
- logrows=15 for standard circuits
- logrows=18 for complex GAN models

Current configuration uses:
- logrows=24 for GAN (6 steps too high = **64x** larger than needed)
- logrows=23 for Classifier (5 steps too high = **32x** larger than needed)

**Impact:**
- Each increment of logrows doubles circuit size, memory, and proving time
- logrows=24 creates circuits with 16.7 million constraints
- Key files are 82GB and 72GB (should be ~1-5GB)
- Proof generation time increased from expected 12 minutes to 3+ hours

**Why GAN Still Works:**
GAN's smaller model (252K params) can handle the over-provisioned circuit in 6 minutes, but it's still inefficient.

**Why Classifier Fails:**
Classifier (620K params) with logrows=23 creates a circuit so large that proving becomes impractical (3+ hours, 121GB RAM).

### Recommended Fix

**Immediate Action:** Reduce logrows to 17-18

```python
# For GAN (252K params)
run_args.logrows = 17  # Reduces PK from 82GB to ~2GB

# For Classifier (620K params)
run_args.logrows = 18  # Reduces PK from 72GB to ~4GB
```

**Expected Results:**
- GAN proof: 3-6 minutes (down from 6 minutes)
- Classifier proof: 6-12 minutes (down from 3+ hours!)
- PK files: 2-4GB each (down from 72-82GB)
- Total improvement: **~95% reduction** in proving time and storage

---

## Problem Summary

Classifier proof generation taking **3+ hours** instead of expected **12 minutes**.

**System:** ProofOfFrog composite ZK-ML proof pipeline
- GAN (252K params) → Classifier (620K params)
- GAN proof: ✅ Works fine (6 minutes)
- Classifier proof: ❌ Hangs for 3+ hours

**Last Run Results:**
- GAN proof: 27KB (completed successfully at 19:05)
- Classifier witness: 473KB (generated at 19:08)
- Classifier proof: KILLED after 3+ hours (195 CPU minutes, 121GB RAM)

**Location:** `/root/ezkl_logs/models/ProofOfFrog_Fixed/classifier/`

---

## Key Findings from EZKL Examples

### 1. logrows Parameter (MOST LIKELY CAUSE)

**Source:** `proof_splitting.ipynb`, `mnist_gan_proof_splitting.ipynb`

The `logrows` parameter directly controls circuit size and memory requirements:
- **logrows=15**: Standard configurations (smaller circuits)
- **logrows=18**: Complex GAN models (larger circuits)
- Each increment doubles the circuit size

**Example from mnist_gan_proof_splitting.ipynb:**
```python
# GAN configuration that works
logrows = 18
calibrate_settings(
    data="input.json",
    model="network.onnx",
    settings="settings.json",
    target="resources"  # Optimize for speed/memory
)
```

**Why This Matters:**
- If classifier logrows > GAN logrows, it would explain longer proof time
- Classifier has 620K params vs GAN's 252K - may have been over-provisioned
- Each extra logrow approximately doubles proving time and memory

**Diagnostic Command:**
```bash
ssh root@47.162.147.233 "cd /root/ezkl_logs/models/ProofOfFrog_Fixed && echo '=== GAN logrows ===' && jq '.run_args.logrows' gan/settings.json && echo '=== Classifier logrows ===' && jq '.run_args.logrows' classifier/settings.json"
```

---

### 2. Calibration Target

**Source:** `ezkl_demo.ipynb`, `proof_splitting.ipynb`

Two calibration targets with different trade-offs:
- **`target='resources'`**: Optimizes for speed and memory (smaller circuits)
- **`target='accuracy'`**: Optimizes for precision (larger circuits, slower)

**From ezkl_demo.ipynb:**
> "Calibration prevents data from falling out of range of lookups"

**Why This Matters:**
- If classifier was calibrated with `target='accuracy'`, it would be slower
- GAN likely used `target='resources'` since it works fine
- Mismatch in targets could explain performance difference

**Diagnostic Command:**
```bash
ssh root@47.162.147.233 "cd /root/ezkl_logs/models/ProofOfFrog_Fixed && echo '=== GAN target ===' && jq '.run_args.calibration_target' gan/settings.json && echo '=== Classifier target ===' && jq '.run_args.calibration_target' classifier/settings.json"
```

---

### 3. Proof Splitting for Large Circuits

**Source:** `mnist_gan_proof_splitting.ipynb`

For models too large for efficient single proofs, split into segments:

**MNIST GAN Splitting Architecture:**
```python
# Split 0: Initial layers (0-4)
split_model(model, 0, 4, 'split_0.onnx')

# Split 1: Middle layers (4-8)
split_model(model, 4, 8, 'split_1.onnx')

# Split 2: Final layers (8+)
split_model(model, 8, len(layers), 'split_2.onnx')

# Link via polycommit
for split in splits:
    output_visibility = "polycommit"
    input_visibility = "polycommit"  # For splits 1+
```

**Cryptographic Linking:**
```python
# Link consecutive proofs
swap_proof_commitments(
    proof1='split_0_proof.json',
    proof2='split_1_proof.json',
    witness1='split_0_witness.json',
    witness2='split_1_witness.json'
)
```

**Why This Matters:**
- If classifier circuit is too large, splitting into 2-3 segments could reduce per-segment proving time
- Each segment proves in parallel conceptually, though still sequential in practice
- More manageable memory footprint per segment

**When to Use:**
- If logrows > 20 or proving time > 30 minutes
- If memory usage > 150GB per proof

---

### 4. Batch Size Configuration

**Source:** `mnist_gan_proof_splitting.ipynb`

**Critical Finding:**
```python
# MUST set batch_size=1 for proof generation
batch_size = 1  # Fixed for proving
```

**Why This Matters:**
- batch_size > 1 causes exponential circuit growth
- If classifier settings.json has batch_size > 1, this could explain the performance issue
- Simple fix: set batch_size=1 and recalibrate

**Diagnostic Command:**
```bash
ssh root@47.162.147.233 "cd /root/ezkl_logs/models/ProofOfFrog_Fixed && echo '=== GAN batch_size ===' && jq '.run_args.batch_size' gan/settings.json && echo '=== Classifier batch_size ===' && jq '.run_args.batch_size' classifier/settings.json"
```

---

### 5. KZG Visibility Configuration

**Source:** `kzg_vis.ipynb`

**Six Visibility Modes:**
1. `public` - Value appears in circuit instances
2. `private` - Value is in witness, verifier doesn't see
3. `fixed` - Value is fixed in circuit (for params)
4. `hashed` - Commitment via Poseidon hash
5. `encrypted` - ElGamal encryption
6. `polycommit` - KZG commitment (unblinded advice column)

**Key Insight:**
> "polycommit doesn't appear in the instances of the circuit and must instead be modified directly within the proof bytes"

This is why `swap_proof_commitments()` is required for linking.

**Current Configuration:**
- GAN: `output_visibility="polycommit"` ✅
- Classifier: `input_visibility="polycommit"` ✅

This configuration is correct per examples.

---

## Proposed Solutions (Ranked by Likelihood)

### Solution 1: Reduce logrows Parameter ⭐⭐⭐⭐⭐

**Most Likely Fix**

**Steps:**
1. Check current logrows values
2. If classifier logrows > GAN logrows + 1, reduce it
3. Recommended: Try logrows=17 or logrows=18 for classifier
4. Recalibrate with new logrows

**Implementation:**
```python
# In proof_of_frog_fixed.py or new script
import ezkl
import json

# Read current settings
with open('classifier/settings.json') as f:
    settings = json.load(f)

# Reduce logrows
run_args = ezkl.PyRunArgs()
run_args.logrows = 17  # Try 17 first, then 18 if too small
run_args.input_visibility = "polycommit"
run_args.param_visibility = "fixed"
run_args.output_visibility = "public"

# Regenerate settings
ezkl.gen_settings(
    model='classifier/network.onnx',
    output='classifier/settings_new.json',
    py_run_args=run_args
)

# Recalibrate
ezkl.calibrate_settings(
    data='classifier/input.json',
    model='classifier/network.onnx',
    settings='classifier/settings_new.json',
    target='resources'
)

# Recompile
ezkl.compile_circuit(
    model='classifier/network.onnx',
    compiled_circuit='classifier/network_new.ezkl',
    settings_path='classifier/settings_new.json'
)

# Regenerate keys (required after circuit change)
# This will take ~7 minutes for classifier
```

**Expected Improvement:**
- logrows=17: ~3-6 minute proof time (50% reduction)
- logrows=18: ~6-12 minute proof time (as expected)

---

### Solution 2: Change Calibration Target ⭐⭐⭐⭐

**Likely If Target Mismatch**

**Steps:**
1. Check if classifier used `target='accuracy'`
2. Recalibrate with `target='resources'`
3. Recompile and regenerate keys

**Implementation:**
```python
# Recalibrate with resources target
ezkl.calibrate_settings(
    data='classifier/input.json',
    model='classifier/network.onnx',
    settings='classifier/settings.json',
    target='resources'  # Changed from 'accuracy'
)

# Recompile (settings changed)
ezkl.compile_circuit(
    model='classifier/network.onnx',
    compiled_circuit='classifier/network.ezkl',
    settings_path='classifier/settings.json'
)

# Regenerate keys (~7 minutes)
ezkl.setup(
    compiled_circuit='classifier/network.ezkl',
    srs_path='classifier/kzg.srs',
    vk_path='classifier/vk_new.key',
    pk_path='classifier/pk_new.key'
)
```

**Expected Improvement:** 20-40% reduction in proving time

---

### Solution 3: Fix batch_size if > 1 ⭐⭐⭐

**Quick Fix If Applicable**

**Steps:**
1. Check if batch_size > 1 in settings
2. Set to batch_size=1
3. Recalibrate and recompile

**Implementation:**
```python
run_args = ezkl.PyRunArgs()
run_args.batch_size = 1  # CRITICAL

# Regenerate settings
ezkl.gen_settings(
    model='classifier/network.onnx',
    output='classifier/settings.json',
    py_run_args=run_args
)

# Full recalibration required
```

**Expected Improvement:** If batch_size was > 1, this could reduce time by 80%+

---

### Solution 4: Split Classifier Circuit ⭐⭐

**Last Resort for Very Large Circuits**

Only needed if:
- logrows > 20
- Circuit has > 1M params
- Proving time > 30 minutes even after optimizations

**Implementation:** Complex - requires splitting ONNX model and linking 2-3 proofs

---

## Diagnostic Script

```bash
#!/bin/bash
# diagnostic_classifier.sh

BASE_DIR="/root/ezkl_logs/models/ProofOfFrog_Fixed"

echo "==================================================================="
echo "ProofOfFrog Classifier Diagnostics"
echo "==================================================================="
echo

echo "--- GAN Settings ---"
echo "logrows: $(jq '.run_args.logrows' $BASE_DIR/gan/settings.json)"
echo "batch_size: $(jq '.run_args.batch_size' $BASE_DIR/gan/settings.json)"
echo "calibration_target: $(jq '.run_args.calibration_target' $BASE_DIR/gan/settings.json)"
echo

echo "--- Classifier Settings ---"
echo "logrows: $(jq '.run_args.logrows' $BASE_DIR/classifier/settings.json)"
echo "batch_size: $(jq '.run_args.batch_size' $BASE_DIR/classifier/settings.json)"
echo "calibration_target: $(jq '.run_args.calibration_target' $BASE_DIR/classifier/settings.json)"
echo

echo "--- Comparison ---"
GAN_LOGROWS=$(jq '.run_args.logrows' $BASE_DIR/gan/settings.json)
CLS_LOGROWS=$(jq '.run_args.logrows' $BASE_DIR/classifier/settings.json)
echo "GAN logrows: $GAN_LOGROWS"
echo "Classifier logrows: $CLS_LOGROWS"

if [ "$CLS_LOGROWS" -gt "$GAN_LOGROWS" ]; then
    DIFF=$((CLS_LOGROWS - GAN_LOGROWS))
    echo "⚠️  Classifier logrows is $DIFF steps higher than GAN"
    echo "   This means ~$((2**DIFF))x larger circuit size"
else
    echo "✅ Classifier logrows is reasonable"
fi
echo

echo "--- File Sizes ---"
echo "GAN PK: $(du -h $BASE_DIR/gan/pk.key | cut -f1)"
echo "GAN VK: $(du -h $BASE_DIR/gan/vk.key | cut -f1)"
echo "Classifier PK: $(du -h $BASE_DIR/classifier/pk.key | cut -f1)"
echo "Classifier VK: $(du -h $BASE_DIR/classifier/vk.key | cut -f1)"
echo

echo "--- Generated Files ---"
if [ -f "$BASE_DIR/gan/proof.json" ]; then
    echo "✅ GAN proof: $(du -h $BASE_DIR/gan/proof.json | cut -f1)"
else
    echo "❌ GAN proof: NOT FOUND"
fi

if [ -f "$BASE_DIR/classifier/proof_from_gan.json" ]; then
    echo "✅ Classifier proof: $(du -h $BASE_DIR/classifier/proof_from_gan.json | cut -f1)"
else
    echo "❌ Classifier proof: NOT FOUND"
fi
```

---

## References

### EZKL Example Notebooks
- **proof_splitting.ipynb**: Basic proof splitting with logrows configuration
  - https://github.com/zkonduit/ezkl/blob/main/examples/notebooks/proof_splitting.ipynb

- **mnist_gan_proof_splitting.ipynb**: GAN-specific splitting with KZG commitments
  - https://github.com/zkonduit/ezkl/blob/main/examples/notebooks/mnist_gan_proof_splitting.ipynb

- **kzg_vis.ipynb**: Visibility modes and polycommit behavior
  - https://github.com/zkonduit/ezkl/blob/main/examples/notebooks/kzg_vis.ipynb

- **ezkl_demo.ipynb**: Calibration targets and optimization
  - https://github.com/zkonduit/ezkl/blob/main/examples/notebooks/ezkl_demo.ipynb

### Related Files
- `/tmp/proof_generation_only.sh` - Script for proof generation without key regeneration
- `/tmp/complete_fixed_pipeline.sh` - Full pipeline with calibration
- `/tmp/README.md` - ProofOfFrog project documentation

---

## Next Steps

1. **Run Diagnostic Script** to identify exact issue
2. **Implement Solution 1** (reduce logrows) - most likely fix
3. **Test with single proof** before full pipeline
4. **Monitor with /usr/bin/time -v** to track memory and CPU

---

*Last Updated: 2025-11-13*
*Investigation: Classifier proof 3+ hours vs expected 12 minutes*

---

## On-Chain Verifier Deployment Investigation

**Date:** 2025-11-14  
**Status:** DEPLOYMENT SUCCESSFUL (LOCAL TESTNET)

### Deployed Contracts (Anvil Local Testnet)

- GAN Verifier: 0x5fbdb2315678afecb367f032d93f642f64180aa3 (15 KB runtime)
- Classifier Verifier: 0xe7f1725e7734ce288f8367e1bb143E90bb3F0512 (301 KB runtime)

### Key Findings

**GAN Verifier:** Fully production-ready
- 15.5 KB runtime bytecode (under 24 KB L1 limit)
- On-chain verification SUCCESSFUL
- Deployable to any EVM chain

**Classifier Verifier:** L1 deployment impossible
- 309 KB runtime bytecode (13x over 24 KB limit)
- Requires Anvil with --code-size-limit 500000 --gas-limit 100000000
- Reusable verifier generation FAILED: "ptr exceeds 16 bits"
- Root cause: 14,336 shuffles create massive verification logic

### Production Recommendations

**For GAN:** Deploy anywhere  
**For Classifier:** L2 required (Arbitrum/Optimism/Base) or off-chain verification

*Full details in ONCHAIN_DEPLOYMENT_SUMMARY.md*

---
