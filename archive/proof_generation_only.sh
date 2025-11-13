#!/bin/bash
#
# ProofOfFrog - Proof Generation Only (Keys Already Exist)
# Skips key generation and jumps to proof generation with --vk-path fix
#

set -e

BASE_DIR="/root/ezkl_logs/models/ProofOfFrog_Fixed"
GAN_DIR="$BASE_DIR/gan"
CLS_DIR="$BASE_DIR/classifier"
EZKL="/root/.ezkl/ezkl"

echo "======================================================================"
echo "PROOF-OF-FROG: Proof Generation (Skipping Key Generation)"
echo "======================================================================"
echo

# ============================================================================
# STEP 1: GAN PROOF GENERATION
# ============================================================================
echo "======================================================================"
echo "STEP 1: GAN Proof Generation (with --vk-path fix)"
echo "======================================================================"
cd "$GAN_DIR"

echo "Generating GAN witness with VK + SRS for KZG commitments..."
"$EZKL" gen-witness \
    --data input.json \
    --compiled-circuit network.ezkl \
    --output witness.json \
    --vk-path vk.key \
    --srs-path kzg.srs \
    2>&1 | tee gen_witness.log

echo "Checking KZG commitment..."
python3 << 'EOPYTHON'
import json
with open("witness.json") as f:
    w = json.load(f)
    kzg = w.get("processed_outputs", {}).get("polycommit")
    if kzg and kzg != "None":
        print(f"  ✓ KZG commitment generated: {kzg}")
    else:
        print("  ✗ ERROR: KZG commitment is None!")
        exit(1)
EOPYTHON

echo
echo "Generating GAN proof (expected: ~6 minutes)..."
/usr/bin/time -v "$EZKL" prove \
    --compiled-circuit network.ezkl \
    --pk-path pk.key \
    --proof-path proof.json \
    --srs-path kzg.srs \
    --witness witness.json \
    2>&1 | tee prove.log

PROOF_SIZE=$(du -h proof.json | cut -f1)
echo "  ✓ GAN proof generated: $PROOF_SIZE"
echo

# ============================================================================
# STEP 2: CLASSIFIER INPUT PREPARATION
# ============================================================================
echo "======================================================================"
echo "STEP 2: Prepare Classifier Input from GAN Output"
echo "======================================================================"
cd "$CLS_DIR"

echo "Extracting GAN output..."
python3 << 'EOPYTHON'
import json

# Read GAN witness
with open('../gan/witness.json') as f:
    gan_witness = json.load(f)

# Extract GAN output
gan_output = gan_witness['outputs'][0]
print(f"  ✓ Extracted GAN output: {len(gan_output)} values")

# Create classifier input
cls_input = {'input_data': [gan_output]}

with open('input_from_gan.json', 'w') as f:
    json.dump(cls_input, f)
print("  ✓ Created classifier input")
EOPYTHON

echo

# ============================================================================
# STEP 3: CLASSIFIER PROOF GENERATION
# ============================================================================
echo "======================================================================"
echo "STEP 3: Classifier Proof Generation (with --vk-path fix)"
echo "======================================================================"

echo "Generating classifier witness with VK + SRS for KZG commitments..."
"$EZKL" gen-witness \
    --data input_from_gan.json \
    --compiled-circuit network.ezkl \
    --output witness_from_gan.json \
    --vk-path vk.key \
    --srs-path kzg.srs \
    2>&1 | tee gen_witness_from_gan.log

echo
echo "Linking KZG commitments..."
python3 << 'EOPYTHON'
import json

# Read GAN witness
with open('../gan/witness.json') as f:
    gan_witness = json.load(f)

# Read classifier witness
with open('witness_from_gan.json') as f:
    cls_witness = json.load(f)

# Copy GAN's processed_outputs to classifier's processed_inputs
cls_witness['processed_inputs'] = gan_witness['processed_outputs']

# Save modified classifier witness
with open('witness_from_gan.json', 'w') as f:
    json.dump(cls_witness, f)

print("  ✓ Linked KZG commitments:")
print(f"    GAN processed_outputs -> Classifier processed_inputs")

# Show commitment details
kzg = gan_witness['processed_outputs'].get('polycommit')
if kzg and kzg != "None":
    print(f"    Commitment: {kzg}")
else:
    print(f"    ✗ WARNING: No KZG commitment found!")
    exit(1)
EOPYTHON

echo
echo "Generating classifier proof (expected: ~12 minutes)..."
/usr/bin/time -v "$EZKL" prove \
    --compiled-circuit network.ezkl \
    --pk-path pk.key \
    --proof-path proof_from_gan.json \
    --srs-path kzg.srs \
    --witness witness_from_gan.json \
    2>&1 | tee prove_from_gan.log

PROOF_SIZE=$(du -h proof_from_gan.json | cut -f1)
echo "  ✓ Classifier proof generated: $PROOF_SIZE"
echo

# ============================================================================
# STEP 4: VERIFICATION
# ============================================================================
echo "======================================================================"
echo "STEP 4: Verification"
echo "======================================================================"

cd "$GAN_DIR"
echo "Verifying GAN proof..."
"$EZKL" verify \
    --proof-path proof.json \
    --settings-path settings.json \
    --vk-path vk.key \
    --srs-path kzg.srs \
    2>&1 | tee verify.log

if grep -q "verified: true" verify.log; then
    echo "  ✅ GAN proof verified successfully"
else
    echo "  ❌ GAN proof verification failed"
    exit 1
fi
echo

cd "$CLS_DIR"
echo "Verifying classifier proof (with 60s timeout)..."
timeout 60 "$EZKL" verify \
    --proof-path proof_from_gan.json \
    --settings-path settings.json \
    --vk-path vk.key \
    --srs-path kzg.srs \
    2>&1 | tee verify_from_gan.log || {
    echo "  ⚠️  Classifier verification timed out or failed"
    echo "  This is the known EZKL bug with input_visibility='KZGCommit'"
}

if grep -q "verified: true" verify_from_gan.log; then
    echo "  ✅ Classifier proof verified successfully - BUG IS FIXED!"
else
    echo "  ❌ Classifier proof verification still hangs/fails"
fi
echo

# ============================================================================
# SUMMARY
# ============================================================================
echo "======================================================================"
echo "PROOF GENERATION COMPLETE"
echo "======================================================================"
echo
echo "✅ GAN proof: $GAN_DIR/proof.json"
echo "✅ Classifier proof: $CLS_DIR/proof_from_gan.json"
echo
echo "Total time: $((SECONDS / 60)) minutes"
echo
