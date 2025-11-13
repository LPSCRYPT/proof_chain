# Proof-of-Frog: Composite ZK-ML Proof System

A cryptographically-linked GAN→Classifier proof pipeline using EZKL, demonstrating proof composition for privacy-preserving machine learning.

## Overview

This project implements an EZKL proof composition system that:
1. **GAN Circuit**: Generates a 32×32 CIFAR-10 image conditioned on a class label
2. **Classifier Circuit**: Evaluates the generated image
3. **Cryptographic Linking**: Binds the two proofs via KZG commitments

The intermediate image remains private (committed via KZG) while proving the complete pipeline executed correctly.

## Prerequisites

### System Requirements
- Ubuntu/Linux environment
- Python 3.8+
- 150GB+ disk space (for generated keys and artifacts)
- 150GB+ RAM recommended for proof generation

### Install EZKL

**Option 1: Download Pre-built Binary (Recommended)**
```bash
# Create directory for EZKL
mkdir -p ~/.ezkl

# Download latest EZKL binary
curl -L https://github.com/zkonduit/ezkl/releases/download/v23.0.3/ezkl -o ~/.ezkl/ezkl
chmod +x ~/.ezkl/ezkl

# Add to PATH (add to ~/.bashrc to make permanent)
export PATH="$HOME/.ezkl:$PATH"

# Verify installation
ezkl --version
```

**Option 2: Install Python Bindings**
```bash
pip install ezkl
```

Both the binary (`ezkl`) and Python bindings (`import ezkl`) are required for the scripts to function.

### Install Python Dependencies
```bash
pip install torch torchvision numpy onnx
```

## Architecture

```
GAN Circuit (output_visibility="KZGCommit")
    ↓ [KZG commitment of generated image]
Classifier Circuit (input_visibility="KZGCommit")
    ↓
Two cryptographically-linked proofs
```

## Models

- **GAN**: Tiny Conditional GAN (252K params, 32×32 RGB output)
  - Input: 42 values (32 latent + 10 class one-hot)
  - Output: 3072 values (3×32×32 RGB image)
  - Trained on CIFAR-10 (10 classes)

- **Classifier**: Tiny Classifier (620K params)
  - Input: 3072 values (flattened RGB image)
  - Output: 10 class logits
  - Trained on CIFAR-10

## Quick Start

```bash
# 1. Setup and calibrate both circuits
python3 proof_of_frog_fixed.py

# 2. Generate complete proof pipeline
bash complete_fixed_pipeline_WORKING.sh
```

## Key Scripts

### Setup Scripts
- `proof_of_frog_fixed.py` - Complete setup with proper `calibrate_settings()` workflow
- `complete_fixed_pipeline_WORKING.sh` - Main proof generation pipeline

### Training Scripts (in cifar_gan_training/)
- `train_tiny_conditional_gan_32x32.py` - GAN training
- `train_classifier.py` - Classifier training

## EZKL Workflow

The correct EZKL workflow for polycommit composition:

```python
# 1. Generate settings with visibility configuration
run_args = ezkl.PyRunArgs()
run_args.output_visibility = "polycommit"  # or "KZGCommit"
ezkl.gen_settings(model='network.onnx', output='settings.json', py_run_args=run_args)

# 2. Calibrate settings (CRITICAL - enables KZG commitments\!)
ezkl.calibrate_settings(
    data='input.json',
    model='network.onnx',
    settings='settings.json',
    target='resources'
)

# 3. Compile circuit
ezkl.compile_circuit(
    model='network.onnx',
    compiled_circuit='network.ezkl',
    settings_path='settings.json'
)

# 4. Setup keys
ezkl.setup(
    compiled_circuit='network.ezkl',
    srs_path='kzg.srs',
    vk_path='vk.key',
    pk_path='pk.key'
)

# 5. Generate proof
ezkl.prove(
    compiled_circuit='network.ezkl',
    pk_path='pk.key',
    proof_path='proof.json',
    srs_path='kzg.srs',
    witness='witness.json'
)
```

**Key Fix**: The `calibrate_settings()` step was missing from the original implementation, causing KZG commitments to be `None` instead of containing actual commitment data.

## Directory Structure

```
proof_chain/
├── README.md
├── .gitignore
├── proof_of_frog_fixed.py              # Main setup script
├── complete_fixed_pipeline_WORKING.sh  # Proof generation pipeline
├── ezkl_workflow_logger.py             # Logging utilities
├── cifar_gan_training/                 # Model training scripts
│   ├── train_tiny_conditional_gan_32x32.py
│   ├── train_classifier.py
│   └── *.onnx (trained models)
├── docs/                               # Documentation
│   ├── agents.md
│   ├── KZG_COMMITMENT_FIX.md
│   └── PROOF_OF_FROG_FINAL_STATUS.md
├── archive/                            # Old/experimental scripts
└── ezkl_logs/                          # Generated artifacts (gitignored)
    └── models/
        └── ProofOfFrog_Fixed/
            ├── gan/
            │   ├── network.onnx
            │   ├── settings.json
            │   ├── network.ezkl
            │   ├── kzg.srs (2.1GB)
            │   ├── pk.key (81GB)
            │   ├── vk.key (37MB)
            │   ├── witness.json
            │   └── proof.json (27KB)
            └── classifier/
                ├── network.onnx
                ├── settings.json
                ├── network.ezkl
                ├── kzg.srs
                ├── pk.key (72GB)
                ├── vk.key (20MB)
                ├── witness_from_gan.json
                └── proof_from_gan.json (25KB)
```

## Performance Metrics

| Stage | Time | Size | Status |
|-------|------|------|--------|
| GAN Setup (VK + PK) | 6.2 min | PK: 81GB, VK: 37MB | ✅ |
| Classifier Setup | 6.9 min | PK: 72GB, VK: 20MB | ✅ |
| SRS Generation | Shared | 2.1GB | ✅ |
| GAN Proof Generation | 6 min | 27 KB | ✅ |
| Classifier Proof Generation | 12 min | 25 KB | ✅ |
| Commitment Linking | Instant | - | ✅ |
| GAN Proof Verification | 0.5 s | - | ✅ |
| Classifier Proof Verification | HANGS | - | ⚠️ |

## Status

### Working Components
- ✅ GAN proof generation and verification
- ✅ Classifier proof generation
- ✅ Cryptographic linking via `swap_proof_commitments()`
- ✅ Witness-level consistency validation
- ✅ Proper KZG commitment generation (with calibration)

### Known Issues
- ⚠️ Classifier proof verification hangs with `input_visibility="KZGCommit"`
- Investigating whether this is an EZKL framework bug or configuration issue
- All other components work correctly

## References

- [EZKL Documentation](https://docs.ezkl.xyz/)
- [EZKL GitHub Repository](https://github.com/zkonduit/ezkl)
- [EZKL Proof Splitting Example](https://github.com/zkonduit/ezkl/blob/main/examples/notebooks/proof_splitting.ipynb)
- [MNIST GAN Proof Splitting](https://github.com/zkonduit/ezkl/blob/main/examples/notebooks/mnist_gan_proof_splitting.ipynb)

## Citation

If you use this work, please reference:
- EZKL framework: [zkonduit/ezkl](https://github.com/zkonduit/ezkl)
- Proof composition pattern from EZKL examples

## License

Research prototype - check with project maintainers before production use.
