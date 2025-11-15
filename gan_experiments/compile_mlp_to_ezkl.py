#!/usr/bin/env python3
"""
Compile MLP GAN model to EZKL and get actual circuit metrics
"""

import torch
import json
import os
import sys
import subprocess

sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')

def compile_to_ezkl():
    print("="*80)
    print("COMPILING MLP GAN (exp_026) TO EZKL CIRCUIT")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MLP configuration
    config = {
        "experiment_id": "exp_026",
        "name": "mlp_only",
        "architecture": {
            "model_type": "mlp",
            "latent_dim": 100,
            "embed_dim": 50,
            "hidden_dims": [512, 1024, 2048, 3072],
            "use_batchnorm": True,
            "use_bias": False,
            "activation": "relu"
        }
    }
    
    from flexible_gan_architectures import create_generator
    
    # Load model
    generator = create_generator(config).to(device)
    model_path = "/root/proof_chain/gan_experiments/tier4/models/exp_026_generator.pth"
    
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ MLP model loaded from {model_path}")
    else:
        print(f"⚠️ Model not found")
        return
    
    generator.eval()
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    onnx_path = "/root/proof_chain/gan_experiments/mlp_gan.onnx"
    
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
    
    print(f"✓ ONNX export complete: {onnx_path}")
    
    # Create input JSON for EZKL
    input_data = {
        "input_shapes": [[1, 100, 1, 1], [1]],
        "input_data": [
            dummy_noise.cpu().numpy().tolist(),
            dummy_labels.cpu().numpy().tolist()
        ]
    }
    
    input_path = "/root/proof_chain/gan_experiments/mlp_input.json"
    with open(input_path, 'w') as f:
        json.dump(input_data, f)
    
    # Compile to EZKL
    print("\n" + "="*80)
    print("COMPILING TO EZKL CIRCUIT")
    print("="*80)
    
    settings = {
        "run_args": {
            "tolerance": {"val": 0.0, "scale": 4},
            "input_scale": 8,
            "param_scale": 8,
            "scale_rebase_multiplier": 10,
            "lookup_range": [-32768, 32767],
            "logrows": 17,
            "num_inner_cols": 2,
            "variables": [["batch_size", 1]],
            "input_visibility": "public",
            "output_visibility": "public",
            "param_visibility": "fixed"
        }
    }
    
    settings_path = "/root/proof_chain/gan_experiments/mlp_settings.json"
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    circuit_path = "/root/proof_chain/gan_experiments/mlp_gan.ezkl"
    
    # Compile
    cmd = f"/root/.ezkl/ezkl compile-circuit --model {onnx_path} --compiled-circuit {circuit_path} --settings-path {settings_path}"
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✓ Circuit compilation successful!")
        
        # Parse output for metrics
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'constraints' in line.lower() or 'einsum' in line.lower() or 'ops' in line.lower():
                print(f"  {line}")
        
        # Get circuit info
        info_cmd = f"/root/.ezkl/ezkl print-circuit-size --compiled-circuit {circuit_path}"
        info_result = subprocess.run(info_cmd, shell=True, capture_output=True, text=True)
        
        if info_result.returncode == 0:
            print("\nCircuit metrics:")
            print(info_result.stdout)
        
        # Save results
        results = {
            "model": "exp_026_mlp",
            "estimated_ops": 20,
            "compilation_output": result.stdout,
            "circuit_info": info_result.stdout if info_result.returncode == 0 else "N/A"
        }
        
        results_path = "/root/proof_chain/gan_experiments/mlp_ezkl_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        
    else:
        print(f"\n❌ Compilation failed:")
        print(result.stderr)

if __name__ == "__main__":
    compile_to_ezkl()
