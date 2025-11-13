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
