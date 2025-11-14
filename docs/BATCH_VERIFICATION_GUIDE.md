# Batch Verification for Halo2/EZKL Proofs

## Overview

Batch verification allows verifying multiple proofs with ~40-70% gas savings compared to individual verification.

## Current Status with EZKL

**EZKL v23.0.3 does NOT have native batch verification**

However, you can implement it using:
1. **Proof Aggregation** (recursive proofs)
2. **Custom Batch Verifier Contract** (manual implementation)
3. **Third-party aggregation services**

---

## Option 1: Proof Aggregation (Recursive Proofs)

### Concept

Create a "meta-proof" that proves "I verified N proofs correctly":

```
Proof 1 ──┐
Proof 2 ──┤
Proof 3 ──┼──→ Aggregation Circuit ──→ Single Aggregated Proof
Proof 4 ──┤
Proof 5 ──┘
```

### Implementation

```python
# Step 1: Create aggregation circuit (ONNX model that verifies proofs)
# This would be a custom circuit that:
# - Takes N proofs as input
# - Verifies each one
# - Outputs a single boolean

# Step 2: Generate proof of aggregation
ezkl.prove(
    compiled_circuit='aggregator.ezkl',
    witness='aggregated_witness.json',  # Contains all N proofs
    proof_path='aggregated_proof.json',
    pk_path='aggregator_pk.key',
    srs_path='kzg.srs'
)

# Step 3: Verify single aggregated proof on-chain
# Gas cost: ~700K (regardless of N)
```

**Trade-offs:**
- ✅ Constant on-chain cost (~700K gas)
- ✅ Works with existing EZKL infrastructure
- ❌ Requires designing aggregation circuit
- ❌ Off-chain aggregation cost (O(N) proving time)
- ❌ Large aggregation circuit = large verifier

---

## Option 2: Custom Batch Verifier Contract

### Concept

Modify the Solidity verifier to accept multiple proofs and use random linear combination.

### Halo2 Batch Verification Formula

For Halo2 (used by EZKL), batch verification works by:

```solidity
// Instead of:
verify(proof1) && verify(proof2) && verify(proof3)

// Do:
// 1. Generate random challenges r1, r2, r3
// 2. Combine proofs: combined = r1·proof1 + r2·proof2 + r3·proof3
// 3. Single pairing check on combined proof
```

### Implementation Steps

**1. Generate random challenges (Fiat-Shamir):**

```solidity
function batchVerify(
    bytes[] calldata proofs,
    uint256[][] calldata instances
) public returns (bool) {
    require(proofs.length > 0, "No proofs");
    
    // Generate random challenges using proof commitments as entropy
    uint256[] memory challenges = new uint256[](proofs.length);
    bytes32 seed = keccak256(abi.encodePacked(proofs[0]));
    
    for (uint i = 0; i < proofs.length; i++) {
        seed = keccak256(abi.encodePacked(seed, proofs[i]));
        challenges[i] = uint256(seed) % R; // R = curve order
    }
    
    // ... combine proofs using challenges
}
```

**2. Combine proof elements:**

```solidity
// Combine polynomial commitments
Point memory combinedCommitment = Point(0, 0);
for (uint i = 0; i < proofs.length; i++) {
    Point memory commitment = extractCommitment(proofs[i]);
    combinedCommitment = pointAdd(
        combinedCommitment,
        scalarMul(commitment, challenges[i])
    );
}
```

**3. Single pairing check:**

```solidity
// Single pairing instead of N pairings
return pairing(
    combinedCommitment,
    G2Generator,
    combinedEvaluation,
    evaluationPoint
);
```

### Challenges

❌ **EZKL-generated verifiers are very complex** (1500+ lines of assembly)
❌ **Manual modification is error-prone** 
❌ **Need deep understanding of Halo2 internals**
❌ **Must maintain across EZKL updates**

**Verdict:** Not recommended unless you're an expert in Halo2 verification

---

## Option 3: Simple Application-Level Batching

### Concept

Don't modify verification logic, just batch the submissions:

```solidity
contract ProofBatcher {
    struct Batch {
        bytes32[] proofHashes;
        bool verified;
    }
    
    mapping(uint256 => Batch) public batches;
    uint256 public currentBatch;
    
    // Users submit proofs to current batch
    function submitProof(bytes calldata proof) external {
        require(!batches[currentBatch].verified, "Batch closed");
        
        bytes32 proofHash = keccak256(proof);
        batches[currentBatch].proofHashes.push(proofHash);
        
        // Store proof off-chain or in calldata
    }
    
    // Operator verifies all proofs in batch
    function verifyBatch(
        uint256 batchId,
        bytes[] calldata proofs
    ) external {
        Batch storage batch = batches[batchId];
        require(!batch.verified, "Already verified");
        
        // Verify each proof individually
        for (uint i = 0; i < proofs.length; i++) {
            require(
                verifier.verifyProof(proofs[i], instances[i]),
                "Proof invalid"
            );
        }
        
        batch.verified = true;
    }
}
```

**Gas Savings:**
- ✅ Saves on transaction overhead (~21K per tx)
- ✅ Single batch verification call
- ⚠️ Still pays full verification cost per proof
- **Savings:** ~5-10% (minimal)

---

## Option 4: Use Existing Aggregation Services

### Available Services

**1. Nil Foundation's zkLLVM Aggregation**
- Supports arbitrary proof aggregation
- Not specifically for EZKL

**2. Succinct Labs SP1**
- Recursive proof composition
- Would need to wrap EZKL proofs

**3. Custom Aggregation Layer**
- Build your own using Halo2 recursive circuits
- Significant development effort

---

## Recommended Approach for Your Use Case

Given you're using EZKL v23.0.3:

### For Low Volume (<10 proofs/day)
**Use individual verification on L2s**
- Cost: /bin/bash.12 per proof on Arbitrum/Base
- Simple, no aggregation needed
- Total cost: </day

### For Medium Volume (10-100 proofs/day)
**Use application-level batching (Option 3)**
- Batch 10-20 proofs per transaction
- Saves ~10% on transaction overhead
- Cost: ~0/day on L2s

### For High Volume (>100 proofs/day)
**Implement proof aggregation (Option 1)**
- Design aggregation circuit
- Generate single proof for batches of 100
- Cost: ~/bin/bash.70 per 100 proofs on L2s
- Requires significant engineering effort

---

## Example: Application-Level Batching

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "./GANVerifier.sol";

contract BatchedGANVerifier {
    GANVerifier public verifier;
    
    event ProofBatchVerified(uint256 indexed batchId, uint256 proofCount);
    
    constructor(address _verifier) {
        verifier = GANVerifier(_verifier);
    }
    
    function verifyBatch(
        bytes[] calldata proofs,
        uint256[][] calldata instances
    ) external returns (bool) {
        require(proofs.length > 0, "Empty batch");
        require(proofs.length == instances.length, "Length mismatch");
        
        for (uint i = 0; i < proofs.length; i++) {
            bool valid = verifier.verifyProof(proofs[i], instances[i]);
            require(valid, string(abi.encodePacked("Proof ", i, " invalid")));
        }
        
        emit ProofBatchVerified(block.number, proofs.length);
        return true;
    }
    
    function estimateBatchGas(uint256 numProofs) external pure returns (uint256) {
        // 663K per proof + 21K base + 5K per proof overhead
        return 21_000 + (numProofs * (663_129 + 5_000));
    }
}
```

---

## Gas Cost Comparison

| Approach | 10 Proofs | 100 Proofs | Complexity |
|----------|-----------|------------|------------|
| Individual | 6.6M gas | 66M gas | Low |
| App-level Batch | 6.4M gas (-3%) | 64M gas (-3%) | Low |
| True Batch Verify | 2.4M gas (-64%) | 20M gas (-70%) | Very High |
| Proof Aggregation | 700K gas (-89%) | 700K gas (-99%) | High |

---

## Conclusion

**For your Proof-of-Frog system:**

1. **Short term:** Deploy individual verifiers on L2 (/bin/bash.12/proof)
2. **Medium term:** Add simple batching contract (saves tx overhead)
3. **Long term:** Consider aggregation if volume >1000 proofs/day

**Most practical: Just use L2s** - At /bin/bash.12/proof, even 1000 verifications = 20. The engineering effort for aggregation isn't worth it unless you're doing >10K verifications/day.

---

*Last Updated: 2025-11-14*
