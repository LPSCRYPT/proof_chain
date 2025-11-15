#!/usr/bin/env python3
import torch
import sys
import os
sys.path.insert(0, '/root/proof_chain')
sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')

from flexible_gan_architectures import create_generator, estimate_einsum_ops

# Define models with different ops counts to test circuit compilation
models_to_test = [
    {
        'name': '34-ops (successfully compiled before)', 
        'ops': 34,
        'config': {
            'experiment_id': 'test_34',
            'architecture': {
                'model_type': 'conv',
                'latent_dim': 100,
                'embed_dim': 50,
                'ngf': 48,
                'num_layers': 3,
                'use_bias': False,
                'use_batchnorm': True
            }
        }
    },
    {
        'name': '72-ops (best quality model)',
        'ops': 72,
        'config': {
            'experiment_id': 'test_72',
            'architecture': {
                'model_type': 'conv',
                'latent_dim': 100,
                'embed_dim': 50,
                'ngf': 72,
                'num_layers': 4,
                'use_bias': False,
                'use_batchnorm': True
            }
        }
    },
    {
        'name': '96-ops (highest ops tested)',
        'ops': 96,
        'config': {
            'experiment_id': 'test_96',
            'architecture': {
                'model_type': 'conv',
                'latent_dim': 100,
                'embed_dim': 50,
                'ngf': 96,
                'num_layers': 4,
                'use_bias': False,
                'use_batchnorm': True,
                'use_skip': True
            }
        }
    }
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('='*80)
print('TESTING CIRCUIT COMPILATION FEASIBILITY')
print('='*80)

results = []

for model_info in models_to_test:
    print(f"\nTesting: {model_info['name']}")
    print('-'*60)
    
    # Create model (don't load weights, just test architecture)
    generator = create_generator(model_info['config']).to(device)
    generator.eval()
    
    # Count actual ops
    actual_ops = estimate_einsum_ops(generator)
    params = sum(p.numel() for p in generator.parameters())
    
    print(f"Expected ops: {model_info['ops']}")
    print(f"Actual ops: {actual_ops}")
    print(f"Parameters: {params:,}")
    
    # Export to ONNX
    onnx_path = f"test_{model_info['ops']}ops.onnx"
    dummy_noise = torch.randn(1, 100, 1, 1).to(device)
    dummy_labels = torch.tensor([0], dtype=torch.long).to(device)
    
    try:
        torch.onnx.export(
            generator,
            (dummy_noise, dummy_labels),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['noise', 'labels'],
            output_names=['generated_image'],
            verbose=False
        )
        onnx_size = os.path.getsize(onnx_path) / (1024*1024)
        print(f"✓ ONNX export successful: {onnx_size:.2f} MB")
        
        # Try to compile to EZKL
        print("Attempting EZKL compilation...")
        circuit_path = f"test_{model_info['ops']}ops.ezkl"
        cmd = f"/root/.ezkl/ezkl compile-circuit -M {onnx_path} -O {circuit_path} --bits 20 2>&1"
        result = os.system(cmd + " | tail -5")
        
        if result == 0 and os.path.exists(circuit_path):
            # Get circuit size
            size_cmd = f"/root/.ezkl/ezkl print-circuit-size --circuit {circuit_path} 2>&1 | grep -E 'constraints|logrows'"
            os.system(size_cmd)
            print(f"✅ CIRCUIT COMPILATION SUCCESSFUL")
            results.append({'ops': actual_ops, 'params': params, 'status': 'SUCCESS'})
        else:
            print(f"❌ CIRCUIT COMPILATION FAILED")
            results.append({'ops': actual_ops, 'params': params, 'status': 'FAILED'})
            
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({'ops': actual_ops, 'params': params, 'status': 'ERROR'})

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for i, result in enumerate(results):
    print(f"{models_to_test[i]['name']}:")
    print(f"  Ops: {result['ops']}, Status: {result['status']}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if any(r['status'] == 'SUCCESS' and r['ops'] >= 72 for r in results):
    print("✅ 72+ op models CAN be circuitized!")
    print("The highest quality models are technically feasible for ZK.")
else:
    print("❌ Only lower-op models can be circuitized.")
    print("72+ op models exceed EZKL compilation limits.")
