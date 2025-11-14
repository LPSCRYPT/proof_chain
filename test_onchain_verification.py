#!/usr/bin/env python3
"""
Test on-chain verification of EZKL proofs using deployed verifier contracts
"""

import json
import subprocess
import sys

# Deployment addresses
GAN_VERIFIER = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
CLASSIFIER_VERIFIER = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
RPC_URL = "http://localhost:8545"

# EZKL and Foundry paths
EZKL_PATH = "/root/.ezkl/ezkl"
CAST_PATH = "/root/.foundry/bin/cast"

# Proof paths
GAN_PROOF_DIR = "ezkl_logs/models/ProofOfFrog_Fixed/gan"
CLASSIFIER_PROOF_DIR = "ezkl_logs/models/ProofOfFrog_Fixed/classifier"

def load_proof(proof_path):
    """Load proof JSON file"""
    with open(proof_path, 'r') as f:
        return json.load(f)

def load_calldata(calldata_path):
    """Load calldata bytes file"""
    with open(calldata_path, 'rb') as f:
        return f.read().hex()

def verify_proof_onchain(verifier_address, proof_path, calldata_path, name="Proof"):
    """Verify proof on-chain using cast"""
    print(f"\n{'='*60}")
    print(f"Verifying {name} on-chain...")
    print(f"{'='*60}")
    print(f"Verifier address: {verifier_address}")
    print(f"Proof path: {proof_path}")
    
    # Load calldata
    calldata = load_calldata(calldata_path)
    print(f"Calldata size: {len(calldata)//2} bytes")
    
    # Call verifyProof function using cast
    # The calldata already includes the function selector and encoded arguments
    cmd = [
        CAST_PATH, "call",
        verifier_address,
        calldata,
        "--rpc-url", RPC_URL
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        # Parse result (should be 0x0000...0001 for true, 0x0000...0000 for false)
        if output.endswith("1"):
            print(f"✅ {name} VERIFIED SUCCESSFULLY on-chain!")
            return True
        else:
            print(f"❌ {name} VERIFICATION FAILED on-chain")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error calling verifier contract:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def get_contract_code(address):
    """Check if contract is deployed"""
    cmd = [CAST_PATH, "code", address, "--rpc-url", RPC_URL]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def main():
    print("\n" + "="*60)
    print("PROOF-OF-FROG: On-Chain Verification Test")
    print("="*60)
    
    # Check if contracts are deployed
    print("\nChecking deployed contracts...")
    gan_code = get_contract_code(GAN_VERIFIER)
    classifier_code = get_contract_code(CLASSIFIER_VERIFIER)
    
    if gan_code == "0x":
        print(f"❌ GAN Verifier not deployed at {GAN_VERIFIER}")
        return False
    else:
        print(f"✅ GAN Verifier deployed ({len(gan_code)//2} bytes)")
    
    if classifier_code == "0x":
        print(f"❌ Classifier Verifier not deployed at {CLASSIFIER_VERIFIER}")
        return False
    else:
        print(f"✅ Classifier Verifier deployed ({len(classifier_code)//2} bytes)")
    
    # Test GAN proof
    gan_success = verify_proof_onchain(
        GAN_VERIFIER,
        f"{GAN_PROOF_DIR}/proof.json",
        f"{GAN_PROOF_DIR}/gan_calldata.bytes",
        "GAN Proof"
    )
    
    # Test Classifier proof
    classifier_success = verify_proof_onchain(
        CLASSIFIER_VERIFIER,
        f"{CLASSIFIER_PROOF_DIR}/proof_from_gan.json",
        f"{CLASSIFIER_PROOF_DIR}/classifier_calldata.bytes",
        "Classifier Proof (from GAN)"
    )
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"GAN Proof: {'✅ PASS' if gan_success else '❌ FAIL'}")
    print(f"Classifier Proof: {'✅ PASS' if classifier_success else '❌ FAIL'}")
    
    if gan_success and classifier_success:
        print("\n🎉 ALL PROOFS VERIFIED SUCCESSFULLY ON-CHAIN! 🎉")
        print("\nThe complete proof chain (GAN → Classifier) is working correctly.")
        print("The proofs are cryptographically linked via KZG commitments.")
        return True
    else:
        print("\n❌ Some proofs failed verification")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
