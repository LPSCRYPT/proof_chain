#!/bin/bash

# Wait for training to complete
echo "Monitoring training log for completion..."
LOG_FILE="/root/cifar_gan_training/training_tiny_gan.log"

while true; do
    if grep -q "ONNX export complete" "$LOG_FILE"; then
        echo "Training completed! Starting EZKL pipeline..."
        break
    fi
    sleep 60
done

# Setup EZKL workspace
WORKSPACE="/root/ezkl_logs/models/TinyConditionalGAN_32x32"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# Copy the ONNX model
cp /root/cifar_gan_training/tiny_conditional_gan_cifar10.onnx ./network.onnx
echo "✓ Copied ONNX model"

# Generate input.json for 'frog' class (index 6)
python3 << 'PYTHON_END'
import torch
import json
import numpy as np

# Create input: 32 latent dims + 10 one-hot class dims
latent_dim = 32
num_classes = 10
frog_class = 6

# Random latent vector
z = torch.randn(1, latent_dim, 1, 1)

# One-hot encoding for frog
one_hot = torch.zeros(1, num_classes, 1, 1)
one_hot[0, frog_class, 0, 0] = 1

# Concatenate
z_with_class = torch.cat([z, one_hot], dim=1)

# Flatten and create JSON
input_array = z_with_class.numpy()
input_data_flat = input_array.flatten().tolist()

input_json = {
    'input_shapes': [[1, 42, 1, 1]],
    'input_data': [input_data_flat]
}

with open('input.json', 'w') as f:
    json.dump(input_json, f, indent=2)

print("✓ Generated input.json for frog class")
PYTHON_END

# Run EZKL pipeline with memory monitoring
echo "Starting EZKL pipeline..."
echo "================================"

echo "Step 1: Generate settings..."
/usr/bin/time -v /root/.ezkl/ezkl gen-settings -M network.onnx --settings-path settings.json 2>&1 | tee -a ezkl_pipeline.log

echo "Step 2: Calibrate settings..."
/usr/bin/time -v /root/.ezkl/ezkl calibrate-settings -M network.onnx -D input.json --settings-path settings.json 2>&1 | tee -a ezkl_pipeline.log

# Extract logrows for SRS generation
LOGROWS=$(python3 -c 'import json; print(json.load(open("settings.json"))["run_args"]["logrows"])')
echo "Detected logrows: $LOGROWS"

echo "Step 3: Compile circuit..."
/usr/bin/time -v /root/.ezkl/ezkl compile-circuit -M network.onnx --compiled-circuit network.ezkl --settings-path settings.json 2>&1 | tee -a ezkl_pipeline.log

echo "Step 4: Generate SRS..."
/usr/bin/time -v /root/.ezkl/ezkl gen-srs --srs-path kzg.srs --logrows $LOGROWS 2>&1 | tee -a ezkl_pipeline.log

echo "Step 5: Setup keys..."
/usr/bin/time -v /root/.ezkl/ezkl setup --compiled-circuit network.ezkl --vk-path vk.key --pk-path pk.key --srs-path kzg.srs 2>&1 | tee -a ezkl_pipeline.log

echo "Step 6: Generate witness..."
python3 << 'WITNESS_END'
import ezkl
result = ezkl.gen_witness("input.json", "network.ezkl", "witness.json")
print(f"Witness generation result: {result}")
WITNESS_END

echo "Step 7: Mock prover..."
/usr/bin/time -v /root/.ezkl/ezkl mock -M network.ezkl -W witness.json 2>&1 | tee -a ezkl_pipeline.log

echo "Step 8: Generate proof..."
/usr/bin/time -v /root/.ezkl/ezkl prove --compiled-circuit network.ezkl --pk-path pk.key --proof-path proof.json --srs-path kzg.srs --witness witness.json 2>&1 | tee -a ezkl_pipeline.log

echo "Step 9: Verify proof..."
/usr/bin/time -v /root/.ezkl/ezkl verify --proof-path proof.json --settings-path settings.json --vk-path vk.key --srs-path kzg.srs 2>&1 | tee -a ezkl_pipeline.log

echo "================================"
echo "EZKL pipeline complete!"
echo "Full log available at: $WORKSPACE/ezkl_pipeline.log"
