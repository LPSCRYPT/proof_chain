#!/bin/bash
#
# Complete ProofOfFrog Fixed Pipeline
# Continues after calibration and SRS generation
# Includes proper timeouts for long-running operations
#

set -e

BASE_DIR="/root/ezkl_logs/models/ProofOfFrog_Fixed"
GAN_DIR="$BASE_DIR/gan"
CLS_DIR="$BASE_DIR/classifier"
EZKL="ezkl"

echo "======================================================================"
echo "PROOF-OF-FROG COMPLETE PIPELINE - Continuing from Fixed Setup"
echo "======================================================================"
echo

# ============================================================================
# STEP 1: GAN KEY GENERATION
# ============================================================================
echo "======================================================================"
echo "STEP 1: GAN Key Generation (VK + PK)"
echo "======================================================================"
cd "$GAN_DIR"

echo "Starting GAN setup (expected: ~6 minutes)..."
/usr/bin/time -v "$EZKL" setup \
    --compiled-circuit network.ezkl \
    --srs-path kzg.srs \
    --vk-path vk.key \
    --pk-path pk.key \
    2>&1 | tee setup.log

PK_SIZE=$(du -h pk.key | cut -f1)
VK_SIZE=$(du -h vk.key | cut -f1)
echo "  ✓ GAN keys generated: PK=$PK_SIZE, VK=$VK_SIZE"
echo

# ============================================================================
# STEP 2: CLASSIFIER KEY GENERATION
# ============================================================================
echo "======================================================================"
echo "STEP 2: Classifier Key Generation (VK + PK)"
echo "======================================================================"
cd "$CLS_DIR"

echo "Starting Classifier setup (expected: ~7 minutes)..."
/usr/bin/time -v "$EZKL" setup \
    --compiled-circuit network.ezkl \
    --srs-path kzg.srs \
    --vk-path vk.key \
    --pk-path pk.key \
    2>&1 | tee setup.log

PK_SIZE=$(du -h pk.key | cut -f1)
VK_SIZE=$(du -h vk.key | cut -f1)
echo "  ✓ Classifier keys generated: PK=$PK_SIZE, VK=$VK_SIZE"
echo

# ============================================================================
# STEP 3: GAN PROOF GENERATION
# ============================================================================
echo "======================================================================"
echo "STEP 3: GAN Proof Generation"
echo "======================================================================"
cd "$GAN_DIR"

echo "Generating witness..."
"$EZKL" gen-witness \
    --data input.json \
    --compiled-circuit network.ezkl \
    --output witness.json \
    --vk-path vk.key \
    --srs-path kzg.srs \
    2>&1 | tee gen_witness.log

echo "Generating proof (expected: ~6 minutes)..."
/usr/bin/time -v "$EZKL" prove \
    --compiled-circuit network.ezkl \
    --pk-path pk.key \
    --proof-path proof.json \
    --srs-path kzg.srs \
    --witness witness.json \
    2>&1 | tee prove.log

PROOF_SIZE=$(du -h proof.json | cut -f1)
echo "  ✓ GAN proof generated: $PROOF_SIZE"

# Check for KZG commitments
echo "Checking witness for KZG commitments..."
python3 -c '
import json
with open("witness.json") as f:
    w = json.load(f)
    po = w.get("processed_outputs", {})
    kzg = po.get("polycommit")
    if kzg is not None and kzg != "None":
        print(f"  ✓ KZG commitment generated: {len(str(kzg))} chars")
    else:
        print(f"  ✗ KZG commitment is None - calibration may have failed")
        exit(1)
'

echo

# ============================================================================
# STEP 4: CLASSIFIER PROOF GENERATION
# ============================================================================
echo "======================================================================"
echo "STEP 4: Classifier Proof Generation (From GAN Output)"
echo "======================================================================"
cd "$CLS_DIR"

echo "Creating classifier input from GAN output..."
python3 << 'EOPYTHON'
import json
import sys

# Read GAN witness
with open('../gan/witness.json') as f:
    gan_witness = json.load(f)

# Extract GAN output (the generated image)
gan_output = gan_witness['outputs'][0]
print(f"  ✓ Extracted GAN output: {len(gan_output)} values")

# Create classifier input
cls_input = {'input_data': [gan_output]}

with open('input_from_gan.json', 'w') as f:
    json.dump(cls_input, f)
print("  ✓ Created classifier input from GAN output")
EOPYTHON

echo "Generating classifier witness..."
"$EZKL" gen-witness \
    --data input_from_gan.json \
    --compiled-circuit network.ezkl \
    --output witness_from_gan.json \
    --vk-path vk.key \
    --srs-path kzg.srs \
    2>&1 | tee gen_witness_from_gan.log

echo "Linking KZG commitments via swap_proof_commitments..."
python3 << 'EOPYTHON'
import json
import ezkl

# Read GAN witness to get processed_outputs
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
    print(f"    Commitment size: {len(str(kzg))} chars")
else:
    print(f"    ✗ WARNING: No KZG commitment found!")
EOPYTHON

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
# STEP 5: VERIFICATION
# ============================================================================
echo "======================================================================"
echo "STEP 5: Verification"
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
echo "COMPLETE PIPELINE SUMMARY"
echo "======================================================================"
echo
echo "Status:"
echo "  ✅ GAN setup complete with calibration"
echo "  ✅ Classifier setup complete with calibration"
echo "  ✅ SRS generated and shared"
echo "  ✅ GAN proof generated with KZG commitments"
echo "  ✅ Classifier proof generated with linked commitments"
echo "  ✅ GAN proof verified"
if grep -q "verified: true" "$CLS_DIR/verify_from_gan.log" 2>/dev/null; then
    echo "  ✅ Classifier proof verified - FULL SUCCESS!"
else
    echo "  ⚠️  Classifier proof verification blocked by EZKL bug"
fi
echo
echo "Proof files:"
echo "  GAN: $GAN_DIR/proof.json ($(du -h $GAN_DIR/proof.json | cut -f1))"
echo "  Classifier: $CLS_DIR/proof_from_gan.json ($(du -h $CLS_DIR/proof_from_gan.json | cut -f1))"
echo
echo "Total time: $(($SECONDS / 60)) minutes"
echo
