# KZG Commitment Generation Fix

## Problem
KZG commitments in witness files were `None` despite:
- Setting `output_visibility="KZGCommit"` in settings
- Calling `calibrate_settings()` after `gen_settings()`
- Passing `--srs-path` to `gen-witness` command

## Root Cause
The `gen-witness` CLI command requires **BOTH** parameters to generate KZG commitments:
- `--vk-path <VK_PATH>` (verification key)
- `--srs-path <SRS_PATH>` (structured reference string)

From EZKL help output:
```
-V, --vk-path <VK_PATH>
      Path to the verification key file (optional - solely used to generate kzg commits)
-P, --srs-path <SRS_PATH>
      Path to the srs file (optional - solely used to generate kzg commits)
```

## Solution
Complete fix requires THREE components:

### 1. Proper Settings Workflow (Python API)
```python
# 1. Generate settings with visibility config
run_args = ezkl.PyRunArgs()
run_args.output_visibility = "polycommit"  # or "KZGCommit"
ezkl.gen_settings(model=network.onnx, output=settings.json, py_run_args=run_args)

# 2. Calibrate settings (CRITICAL!)
ezkl.calibrate_settings(
    data=input.json,
    model=network.onnx,
    settings=settings.json,
    target=resources
)

# 3. Compile circuit
ezkl.compile_circuit(
    model=network.onnx,
    compiled_circuit=network.ezkl,
    settings_path=settings.json
)
```

### 2. Generate Keys
```bash
ezkl setup \
    --compiled-circuit network.ezkl \
    --srs-path kzg.srs \
    --vk-path vk.key \
    --pk-path pk.key
```

### 3. Generate Witness with BOTH Parameters
```bash
ezkl gen-witness \
    --data input.json \
    --compiled-circuit network.ezkl \
    --output witness.json \
    --vk-path vk.key \    # REQUIRED for KZG commitments
    --srs-path kzg.srs \  # REQUIRED for KZG commitments
```

## Verification
Check that KZG commitments are generated:

```python
import json
with open("witness.json") as f:
    w = json.load(f)
    kzg = w["processed_outputs"]["polycommit"]
    print(f"KZG Commitment: {kzg}")
    # Expected: [[d55928e461329d7ff6f3f8dd9236e9c3bcf30b5a19c8a3d90ef40961e659049b]]
    # NOT: None
```

## Files
- `/root/complete_fixed_pipeline_WORKING.sh` - Fully fixed pipeline script
- `/root/proof_of_frog_fixed.py` - Python setup with calibration
- `/root/ezkl_logs/models/ProofOfFrog_Fixed/` - Working proof directory

## Timeline
- Initial attempt: Only `--srs-path` → commitments still `None`
- Discovery: EZKL help shows both `--vk-path` and `--srs-path` needed
- Fix applied: Added both parameters → commitments generated successfully
- Result: Hex commitment `d55928e461329d7ff6f3f8dd9236e9c3bcf30b5a19c8a3d90ef40961e659049b`

## Key Insight
The Python API (`ezkl.gen_witness(vk_path=...)`) abstracts both parameters into a single `vk_path` argument, but internally uses both VK and SRS for commitment generation. The CLI requires explicit specification of both.
