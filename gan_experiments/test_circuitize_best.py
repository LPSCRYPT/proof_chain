#!/usr/bin/env python3
import torch
import sys
import os
sys.path.insert(0, '/root/proof_chain')
sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')

from flexible_gan_architectures import create_generator, estimate_einsum_ops

# Test the best quality model (exp_016)
config = {
    'experiment_id': 'exp_016',
    'name': 'conv_quality_focus',
    'architecture': {
        'model_type': 'conv',
        'latent_dim': 100,
        'embed_dim': 50,
        'ngf': 72,
        'num_layers': 4,
        'use_bias': False,
        'activation': 'relu',
        'use_batchnorm': True,
        'normalization': 'batch'
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('='*80)
print('TESTING HIGHEST QUALITY MODEL FOR EZKL COMPILATION')
print('Model: exp_016 - 72 ops, 29.4% accuracy (best quality)')
print('='*80)

# Create and load model
generator = create_generator(config).to(device)
model_path = 'tier3/models/exp_016_generator.pth'

if os.path.exists(model_path):
    generator.load_state_dict(torch.load(model_path, map_location=device))
    print(f'✓ Model loaded from {model_path}')

generator.eval()

# Count ops
ops = estimate_einsum_ops(generator)
params = sum(p.numel() for p in generator.parameters())
print(f'\nModel Statistics:')
print(f'  Einsum ops: {ops}')
print(f'  Parameters: {params:,}')
print(f'  Model size: {os.path.getsize(model_path)/(1024*1024):.2f} MB')

# Export to ONNX
print('\n' + '='*60)
print('STEP 1: EXPORTING TO ONNX')
print('='*60)

onnx_path = 'best_quality_gan.onnx'
dummy_noise = torch.randn(1, 100, 1, 1).to(device)
dummy_labels = torch.tensor([0], dtype=torch.long).to(device)

torch.onnx.export(
    generator,
    (dummy_noise, dummy_labels),
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['noise', 'labels'],
    output_names=['generated_image']
)
print(f'✓ ONNX export successful: {onnx_path}')
print(f'  File size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB')

# Now try to compile to EZKL
print('\n' + '='*60)
print('STEP 2: COMPILING TO EZKL CIRCUIT')
print('='*60)

os.system('/root/.ezkl/ezkl table -M best_quality_gan.onnx')
print('\nAttempting circuit compilation...')
result = os.system('/root/.ezkl/ezkl compile-circuit -M best_quality_gan.onnx -O best_quality_circuit.ezkl --bits 20 2>&1 | tail -10')

if result == 0:
    print('\n✓ Circuit compilation successful!')
    # Get circuit stats
    os.system('/root/.ezkl/ezkl print-circuit-size --circuit best_quality_circuit.ezkl 2>&1 | head -10')
else:
    print('\n✗ Circuit compilation failed - model too complex')
