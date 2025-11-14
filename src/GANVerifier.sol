// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract GANVerifier {
    uint256 internal constant    DELTA = 4131629893567559867359510883348571134090853742863529169391034518566172092834;
    uint256 internal constant        R = 21888242871839275222246405745257275088548364400416034343698204186575808495617; 

    uint256 internal constant FIRST_QUOTIENT_X_CPTR = 0x0624;
    uint256 internal constant  LAST_QUOTIENT_X_CPTR = 0x06e4;

    uint256 internal constant                VK_MPTR = 0x07a0;
    uint256 internal constant         VK_DIGEST_MPTR = 0x07a0;
    uint256 internal constant     NUM_INSTANCES_MPTR = 0x07c0;
    uint256 internal constant                 K_MPTR = 0x07e0;
    uint256 internal constant             N_INV_MPTR = 0x0800;
    uint256 internal constant             OMEGA_MPTR = 0x0820;
    uint256 internal constant         OMEGA_INV_MPTR = 0x0840;
    uint256 internal constant    OMEGA_INV_TO_L_MPTR = 0x0860;
    uint256 internal constant   HAS_ACCUMULATOR_MPTR = 0x0880;
    uint256 internal constant        ACC_OFFSET_MPTR = 0x08a0;
    uint256 internal constant     NUM_ACC_LIMBS_MPTR = 0x08c0;
    uint256 internal constant NUM_ACC_LIMB_BITS_MPTR = 0x08e0;
    uint256 internal constant              G1_X_MPTR = 0x0900;
    uint256 internal constant              G1_Y_MPTR = 0x0920;
    uint256 internal constant            G2_X_1_MPTR = 0x0940;
    uint256 internal constant            G2_X_2_MPTR = 0x0960;
    uint256 internal constant            G2_Y_1_MPTR = 0x0980;
    uint256 internal constant            G2_Y_2_MPTR = 0x09a0;
    uint256 internal constant      NEG_S_G2_X_1_MPTR = 0x09c0;
    uint256 internal constant      NEG_S_G2_X_2_MPTR = 0x09e0;
    uint256 internal constant      NEG_S_G2_Y_1_MPTR = 0x0a00;
    uint256 internal constant      NEG_S_G2_Y_2_MPTR = 0x0a20;

    uint256 internal constant CHALLENGE_MPTR = 0x1080;

    uint256 internal constant THETA_MPTR = 0x1080;
    uint256 internal constant  BETA_MPTR = 0x10a0;
    uint256 internal constant GAMMA_MPTR = 0x10c0;
    uint256 internal constant     Y_MPTR = 0x10e0;
    uint256 internal constant     X_MPTR = 0x1100;
    uint256 internal constant  ZETA_MPTR = 0x1120;
    uint256 internal constant    NU_MPTR = 0x1140;
    uint256 internal constant    MU_MPTR = 0x1160;

    uint256 internal constant       ACC_LHS_X_MPTR = 0x1180;
    uint256 internal constant       ACC_LHS_Y_MPTR = 0x11a0;
    uint256 internal constant       ACC_RHS_X_MPTR = 0x11c0;
    uint256 internal constant       ACC_RHS_Y_MPTR = 0x11e0;
    uint256 internal constant             X_N_MPTR = 0x1200;
    uint256 internal constant X_N_MINUS_1_INV_MPTR = 0x1220;
    uint256 internal constant          L_LAST_MPTR = 0x1240;
    uint256 internal constant         L_BLIND_MPTR = 0x1260;
    uint256 internal constant             L_0_MPTR = 0x1280;
    uint256 internal constant   INSTANCE_EVAL_MPTR = 0x12a0;
    uint256 internal constant   QUOTIENT_EVAL_MPTR = 0x12c0;
    uint256 internal constant      QUOTIENT_X_MPTR = 0x12e0;
    uint256 internal constant      QUOTIENT_Y_MPTR = 0x1300;
    uint256 internal constant          R_EVAL_MPTR = 0x1320;
    uint256 internal constant   PAIRING_LHS_X_MPTR = 0x1340;
    uint256 internal constant   PAIRING_LHS_Y_MPTR = 0x1360;
    uint256 internal constant   PAIRING_RHS_X_MPTR = 0x1380;
    uint256 internal constant   PAIRING_RHS_Y_MPTR = 0x13a0;

    function verifyProof(
        bytes calldata proof,
        uint256[] calldata instances
    ) public returns (bool) {
        assembly {
            // Read EC point (x, y) at (proof_cptr, proof_cptr + 0x20),
            // and check if the point is on affine plane,
            // and store them in (hash_mptr, hash_mptr + 0x20).
            // Return updated (success, proof_cptr, hash_mptr).
            function read_ec_point(success, proof_cptr, hash_mptr, q) -> ret0, ret1, ret2 {
                let x := calldataload(proof_cptr)
                let y := calldataload(add(proof_cptr, 0x20))
                ret0 := and(success, lt(x, q))
                ret0 := and(ret0, lt(y, q))
                ret0 := and(ret0, eq(mulmod(y, y, q), addmod(mulmod(x, mulmod(x, x, q), q), 3, q)))
                mstore(hash_mptr, x)
                mstore(add(hash_mptr, 0x20), y)
                ret1 := add(proof_cptr, 0x40)
                ret2 := add(hash_mptr, 0x40)
            }

            // Squeeze challenge by keccak256(memory[0..hash_mptr]),
            // and store hash mod r as challenge in challenge_mptr,
            // and push back hash in 0x00 as the first input for next squeeze.
            // Return updated (challenge_mptr, hash_mptr).
            function squeeze_challenge(challenge_mptr, hash_mptr, r) -> ret0, ret1 {
                let hash := keccak256(0x00, hash_mptr)
                mstore(challenge_mptr, mod(hash, r))
                mstore(0x00, hash)
                ret0 := add(challenge_mptr, 0x20)
                ret1 := 0x20
            }

            // Squeeze challenge without absorbing new input from calldata,
            // by putting an extra 0x01 in memory[0x20] and squeeze by keccak256(memory[0..21]),
            // and store hash mod r as challenge in challenge_mptr,
            // and push back hash in 0x00 as the first input for next squeeze.
            // Return updated (challenge_mptr).
            function squeeze_challenge_cont(challenge_mptr, r) -> ret {
                mstore8(0x20, 0x01)
                let hash := keccak256(0x00, 0x21)
                mstore(challenge_mptr, mod(hash, r))
                mstore(0x00, hash)
                ret := add(challenge_mptr, 0x20)
            }

            // Batch invert values in memory[mptr_start..mptr_end] in place.
            // Return updated (success).
            function batch_invert(success, mptr_start, mptr_end) -> ret {
                let gp_mptr := mptr_end
                let gp := mload(mptr_start)
                let mptr := add(mptr_start, 0x20)
                for
                    {}
                    lt(mptr, sub(mptr_end, 0x20))
                    {}
                {
                    gp := mulmod(gp, mload(mptr), R)
                    mstore(gp_mptr, gp)
                    mptr := add(mptr, 0x20)
                    gp_mptr := add(gp_mptr, 0x20)
                }
                gp := mulmod(gp, mload(mptr), R)

                mstore(gp_mptr, 0x20)
                mstore(add(gp_mptr, 0x20), 0x20)
                mstore(add(gp_mptr, 0x40), 0x20)
                mstore(add(gp_mptr, 0x60), gp)
                mstore(add(gp_mptr, 0x80), sub(R, 2))
                mstore(add(gp_mptr, 0xa0), R)
                ret := and(success, staticcall(gas(), 0x05, gp_mptr, 0xc0, gp_mptr, 0x20))
                let all_inv := mload(gp_mptr)

                let first_mptr := mptr_start
                let second_mptr := add(first_mptr, 0x20)
                gp_mptr := sub(gp_mptr, 0x20)
                for
                    {}
                    lt(second_mptr, mptr)
                    {}
                {
                    let inv := mulmod(all_inv, mload(gp_mptr), R)
                    all_inv := mulmod(all_inv, mload(mptr), R)
                    mstore(mptr, inv)
                    mptr := sub(mptr, 0x20)
                    gp_mptr := sub(gp_mptr, 0x20)
                }
                let inv_first := mulmod(all_inv, mload(second_mptr), R)
                let inv_second := mulmod(all_inv, mload(first_mptr), R)
                mstore(first_mptr, inv_first)
                mstore(second_mptr, inv_second)
            }

            // Add (x, y) into point at (0x00, 0x20).
            // Return updated (success).
            function ec_add_acc(success, x, y) -> ret {
                mstore(0x40, x)
                mstore(0x60, y)
                ret := and(success, staticcall(gas(), 0x06, 0x00, 0x80, 0x00, 0x40))
            }

            // Scale point at (0x00, 0x20) by scalar.
            function ec_mul_acc(success, scalar) -> ret {
                mstore(0x40, scalar)
                ret := and(success, staticcall(gas(), 0x07, 0x00, 0x60, 0x00, 0x40))
            }

            // Add (x, y) into point at (0x80, 0xa0).
            // Return updated (success).
            function ec_add_tmp(success, x, y) -> ret {
                mstore(0xc0, x)
                mstore(0xe0, y)
                ret := and(success, staticcall(gas(), 0x06, 0x80, 0x80, 0x80, 0x40))
            }

            // Scale point at (0x80, 0xa0) by scalar.
            // Return updated (success).
            function ec_mul_tmp(success, scalar) -> ret {
                mstore(0xc0, scalar)
                ret := and(success, staticcall(gas(), 0x07, 0x80, 0x60, 0x80, 0x40))
            }

            // Perform pairing check.
            // Return updated (success).
            function ec_pairing(success, lhs_x, lhs_y, rhs_x, rhs_y) -> ret {
                mstore(0x00, lhs_x)
                mstore(0x20, lhs_y)
                mstore(0x40, mload(G2_X_1_MPTR))
                mstore(0x60, mload(G2_X_2_MPTR))
                mstore(0x80, mload(G2_Y_1_MPTR))
                mstore(0xa0, mload(G2_Y_2_MPTR))
                mstore(0xc0, rhs_x)
                mstore(0xe0, rhs_y)
                mstore(0x100, mload(NEG_S_G2_X_1_MPTR))
                mstore(0x120, mload(NEG_S_G2_X_2_MPTR))
                mstore(0x140, mload(NEG_S_G2_Y_1_MPTR))
                mstore(0x160, mload(NEG_S_G2_Y_2_MPTR))
                ret := and(success, staticcall(gas(), 0x08, 0x00, 0x180, 0x00, 0x20))
                ret := and(ret, mload(0x00))
            }

            // Modulus
            let q := 21888242871839275222246405745257275088696311157297823662689037894645226208583 // BN254 base field
            let r := 21888242871839275222246405745257275088548364400416034343698204186575808495617 // BN254 scalar field 

            // Initialize success as true
            let success := true

            {
                // Load vk_digest and num_instances of vk into memory
                mstore(0x07a0, 0x2591410ed4336c8cfcaa7bca75cd124a9593a26245afeba8ccf8cccfc5c4fffb) // vk_digest
                mstore(0x07c0, 0x000000000000000000000000000000000000000000000000000000000000002a) // num_instances

                // Check valid length of proof
                success := and(success, eq(0x0ec0, proof.length))

                // Check valid length of instances
                let num_instances := mload(NUM_INSTANCES_MPTR)
                success := and(success, eq(num_instances, instances.length))

                // Absorb vk diegst
                mstore(0x00, mload(VK_DIGEST_MPTR))

                // Read instances and witness commitments and generate challenges
                let hash_mptr := 0x20
                let instance_cptr := instances.offset
                for
                    { let instance_cptr_end := add(instance_cptr, mul(0x20, num_instances)) }
                    lt(instance_cptr, instance_cptr_end)
                    {}
                {
                    let instance := calldataload(instance_cptr)
                    success := and(success, lt(instance, r))
                    mstore(hash_mptr, instance)
                    instance_cptr := add(instance_cptr, 0x20)
                    hash_mptr := add(hash_mptr, 0x20)
                }

                let proof_cptr := proof.offset
                let challenge_mptr := CHALLENGE_MPTR

                // Phase 1
                for
                    { let proof_cptr_end := add(proof_cptr, 0x01c0) }
                    lt(proof_cptr, proof_cptr_end)
                    {}
                {
                    success, proof_cptr, hash_mptr := read_ec_point(success, proof_cptr, hash_mptr, q)
                }

                challenge_mptr, hash_mptr := squeeze_challenge(challenge_mptr, hash_mptr, r)

                // Phase 2
                for
                    { let proof_cptr_end := add(proof_cptr, 0x0180) }
                    lt(proof_cptr, proof_cptr_end)
                    {}
                {
                    success, proof_cptr, hash_mptr := read_ec_point(success, proof_cptr, hash_mptr, q)
                }

                challenge_mptr, hash_mptr := squeeze_challenge(challenge_mptr, hash_mptr, r)
                challenge_mptr := squeeze_challenge_cont(challenge_mptr, r)

                // Phase 3
                for
                    { let proof_cptr_end := add(proof_cptr, 0x0280) }
                    lt(proof_cptr, proof_cptr_end)
                    {}
                {
                    success, proof_cptr, hash_mptr := read_ec_point(success, proof_cptr, hash_mptr, q)
                }

                challenge_mptr, hash_mptr := squeeze_challenge(challenge_mptr, hash_mptr, r)

                // Phase 4
                for
                    { let proof_cptr_end := add(proof_cptr, 0x0100) }
                    lt(proof_cptr, proof_cptr_end)
                    {}
                {
                    success, proof_cptr, hash_mptr := read_ec_point(success, proof_cptr, hash_mptr, q)
                }

                challenge_mptr, hash_mptr := squeeze_challenge(challenge_mptr, hash_mptr, r)

                // Read evaluations
                for
                    { let proof_cptr_end := add(proof_cptr, 0x0780) }
                    lt(proof_cptr, proof_cptr_end)
                    {}
                {
                    let eval := calldataload(proof_cptr)
                    success := and(success, lt(eval, r))
                    mstore(hash_mptr, eval)
                    proof_cptr := add(proof_cptr, 0x20)
                    hash_mptr := add(hash_mptr, 0x20)
                }

                // Read batch opening proof and generate challenges
                challenge_mptr, hash_mptr := squeeze_challenge(challenge_mptr, hash_mptr, r)       // zeta
                challenge_mptr := squeeze_challenge_cont(challenge_mptr, r)                        // nu

                success, proof_cptr, hash_mptr := read_ec_point(success, proof_cptr, hash_mptr, q) // W

                challenge_mptr, hash_mptr := squeeze_challenge(challenge_mptr, hash_mptr, r)       // mu

                success, proof_cptr, hash_mptr := read_ec_point(success, proof_cptr, hash_mptr, q) // W'

                // Load full vk into memory
                mstore(0x07a0, 0x2591410ed4336c8cfcaa7bca75cd124a9593a26245afeba8ccf8cccfc5c4fffb) // vk_digest
                mstore(0x07c0, 0x000000000000000000000000000000000000000000000000000000000000002a) // num_instances
                mstore(0x07e0, 0x0000000000000000000000000000000000000000000000000000000000000018) // k
                mstore(0x0800, 0x30644e427ce32d4886b01bfe313ba1dba6db8b2045d128178a7164500e0a6c11) // n_inv
                mstore(0x0820, 0x1951441010b2b95a6e47a6075066a50a036f5ba978c050f2821df86636c0facb) // omega
                mstore(0x0840, 0x26501ebfe559ea5826f023d3e76e4b66f170cd940408eb5590a4075c80b498d6) // omega_inv
                mstore(0x0860, 0x20669a4af88d28096b76c1e00c7eb94022b5eeb63400055aa63bd19ae7ede714) // omega_inv_to_l
                mstore(0x0880, 0x0000000000000000000000000000000000000000000000000000000000000000) // has_accumulator
                mstore(0x08a0, 0x0000000000000000000000000000000000000000000000000000000000000000) // acc_offset
                mstore(0x08c0, 0x0000000000000000000000000000000000000000000000000000000000000000) // num_acc_limbs
                mstore(0x08e0, 0x0000000000000000000000000000000000000000000000000000000000000000) // num_acc_limb_bits
                mstore(0x0900, 0x0000000000000000000000000000000000000000000000000000000000000001) // g1_x
                mstore(0x0920, 0x0000000000000000000000000000000000000000000000000000000000000002) // g1_y
                mstore(0x0940, 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2) // g2_x_1
                mstore(0x0960, 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed) // g2_x_2
                mstore(0x0980, 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b) // g2_y_1
                mstore(0x09a0, 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa) // g2_y_2
                mstore(0x09c0, 0x186282957db913abd99f91db59fe69922e95040603ef44c0bd7aa3adeef8f5ac) // neg_s_g2_x_1
                mstore(0x09e0, 0x17944351223333f260ddc3b4af45191b856689eda9eab5cbcddbbe570ce860d2) // neg_s_g2_x_2
                mstore(0x0a00, 0x06d971ff4a7467c3ec596ed6efc674572e32fd6f52b721f97e35b0b3d3546753) // neg_s_g2_y_1
                mstore(0x0a20, 0x06ecdb9f9567f59ed2eee36e1e1d58797fd13cc97fafc2910f5e8a12f202fa9a) // neg_s_g2_y_2
                mstore(0x0a40, 0x1759640ed3b210993b341f5ba77bf873807ccf50aab4bbd344b8fecad2fafdf3) // fixed_comms[0].x
                mstore(0x0a60, 0x2384b2f92ea8b5a66dbe5c7a8b94ec90511aff54fe4b1ca3c7f43b4e4e79e77d) // fixed_comms[0].y
                mstore(0x0a80, 0x0b1b5744aa5fb262504e0c6a32b791c32f5aab01fc28ee3f5422035e653f8768) // fixed_comms[1].x
                mstore(0x0aa0, 0x19eae87867d188aa72205b5e5fd79af11942bc50c68130fa09f97d1f0ac84c8a) // fixed_comms[1].y
                mstore(0x0ac0, 0x1b543c469ba50c1ad1826c847da74854978723659f967bbfb384f04eb12e0bf2) // fixed_comms[2].x
                mstore(0x0ae0, 0x1dbaad175c9f4e9f3de8465308c16c9497228a8ce603c1c9dbd8ff4f0e9d0597) // fixed_comms[2].y
                mstore(0x0b00, 0x2c7a4b7a467d369ee107ebffbda2fa890f602f175a80a83ce64715bf0a96463e) // fixed_comms[3].x
                mstore(0x0b20, 0x07f6cf84b8d9534cec90f54236e13a4e09022f5da259a392b986bc2d6a5bf635) // fixed_comms[3].y
                mstore(0x0b40, 0x1f5886bc0558717e3351d15f2c1b30228f6c35f545b6b407abf090c018f4b9e1) // fixed_comms[4].x
                mstore(0x0b60, 0x2ff6a0891f8a0e34af354e0db56370db4ddba77b212520f4d7a0c0d77116f3f4) // fixed_comms[4].y
                mstore(0x0b80, 0x0077671293c536f685c08fdc5f84e752c2387a080f09821941524c8b818a537e) // fixed_comms[5].x
                mstore(0x0ba0, 0x1967394082020455a658f1f90cf8554a35a42cff8a1520ba1389f04a2201d3d7) // fixed_comms[5].y
                mstore(0x0bc0, 0x0077671293c536f685c08fdc5f84e752c2387a080f09821941524c8b818a537e) // fixed_comms[6].x
                mstore(0x0be0, 0x1967394082020455a658f1f90cf8554a35a42cff8a1520ba1389f04a2201d3d7) // fixed_comms[6].y
                mstore(0x0c00, 0x06feaddae4046d702db8fb9a106b315f7e7dec0c47c0cde4c6ce0e0f325a820b) // fixed_comms[7].x
                mstore(0x0c20, 0x04abbd95d755534ad5bccac7823b7bb5784127ee784edaebb4e57f47e806f936) // fixed_comms[7].y
                mstore(0x0c40, 0x06feaddae4046d702db8fb9a106b315f7e7dec0c47c0cde4c6ce0e0f325a820b) // fixed_comms[8].x
                mstore(0x0c60, 0x04abbd95d755534ad5bccac7823b7bb5784127ee784edaebb4e57f47e806f936) // fixed_comms[8].y
                mstore(0x0c80, 0x0ae4dad709d39d345969617fb0f407061c16d7ade3ebf765b28cb213816ab73f) // fixed_comms[9].x
                mstore(0x0ca0, 0x19dfcdb46cf9b660cadbca98681f3f44b21f873a92bb6b98810efd4898c51fdf) // fixed_comms[9].y
                mstore(0x0cc0, 0x0ae4dad709d39d345969617fb0f407061c16d7ade3ebf765b28cb213816ab73f) // fixed_comms[10].x
                mstore(0x0ce0, 0x19dfcdb46cf9b660cadbca98681f3f44b21f873a92bb6b98810efd4898c51fdf) // fixed_comms[10].y
                mstore(0x0d00, 0x00ef15961bfcd48b53b7a2631c235e01d68719be755107297c3105af70513c10) // fixed_comms[11].x
                mstore(0x0d20, 0x1e20550402df9db9ec1c0cc5582d9425601fb8d9eea83de527aa348a5d78d9b0) // fixed_comms[11].y
                mstore(0x0d40, 0x097f0263a22d8d017edcc3e17add11e8815e9c0018848ce0427f8d6a0aa06bb1) // fixed_comms[12].x
                mstore(0x0d60, 0x0a8bb64729448aeb03e6a7f5b4b8f59b05b4b1ecc156c8f320bc9de0c28ea911) // fixed_comms[12].y
                mstore(0x0d80, 0x1cd985a97efb2e9ad7cd00553c2e3e5ec8ee45e56331ffc76e9e7b4f27910c10) // fixed_comms[13].x
                mstore(0x0da0, 0x169c6c48b7f2f816a56efb6f885303387c2d5b6f90b0ab136f23eaa7a4bda334) // fixed_comms[13].y
                mstore(0x0dc0, 0x0000000000000000000000000000000000000000000000000000000000000000) // fixed_comms[14].x
                mstore(0x0de0, 0x0000000000000000000000000000000000000000000000000000000000000000) // fixed_comms[14].y
                mstore(0x0e00, 0x0000000000000000000000000000000000000000000000000000000000000000) // fixed_comms[15].x
                mstore(0x0e20, 0x0000000000000000000000000000000000000000000000000000000000000000) // fixed_comms[15].y
                mstore(0x0e40, 0x1d67546c2e35cb0e4c9ccfab53d1f4b5a3870aabd62325175c63d56223bc2e6e) // permutation_comms[0].x
                mstore(0x0e60, 0x19b5ccabc0405e0d546b2e35a2cca051d26ca08ed833b755e83957db23d899fd) // permutation_comms[0].y
                mstore(0x0e80, 0x0e3285dead74d892fb141236550534243a55a57233467f221bb675e17cd5542b) // permutation_comms[1].x
                mstore(0x0ea0, 0x2bacc506c020df1a9da85d70dbbb240058a423c652acddec296c738028905ef9) // permutation_comms[1].y
                mstore(0x0ec0, 0x09456700395c3eac4e8272b0055154e8b8a2a7cc86c7906ef50cd25250939e23) // permutation_comms[2].x
                mstore(0x0ee0, 0x0e891abef75b64837d3c8a7fe68a81dcf1fa48513af9db8e7036442454fa3ab7) // permutation_comms[2].y
                mstore(0x0f00, 0x2841b5f116bae5a6f457651dcf5e9ba8adabf5c8094f02ce325ce33a0809f08e) // permutation_comms[3].x
                mstore(0x0f20, 0x286eb49ef2f54e3b8f8ac9e271b7f8359e729f4f4ea7971cd5dacd5c535eebf4) // permutation_comms[3].y
                mstore(0x0f40, 0x22645f168273dc79a1ed1fb1e564835d20ffb66a1684d26dbb819b16c0d46611) // permutation_comms[4].x
                mstore(0x0f60, 0x1effb4ea318c22cdbddb6f7f40dd12888fa9e86f3f17f64d394a1d4fda6d7f18) // permutation_comms[4].y
                mstore(0x0f80, 0x1ee2718ec8886fdab8479b47e611fb94801622b5e5f9acea18f8e2a464f1cfda) // permutation_comms[5].x
                mstore(0x0fa0, 0x03a106651fe02a4e8c1133281073f58cef1bbb38a1e2a3890353f778360986dd) // permutation_comms[5].y
                mstore(0x0fc0, 0x0fdbb60ff734927dcab6aeb1b2afe82833dc20daca251b856e09f1d5966c9b58) // permutation_comms[6].x
                mstore(0x0fe0, 0x176fa6109f9110b7ca6c7517f7a7d01042cf3e2634520ddbed61651e63e11bcf) // permutation_comms[6].y
                mstore(0x1000, 0x2a7ed13b20ca7cd8af5a03bf174016c7ebf0c6d69f6efc33796c7ea5356fd261) // permutation_comms[7].x
                mstore(0x1020, 0x1bf249ae3f38ca94c7ed7f00d65222f217c8fab28e23c682b60083d0f832ab3c) // permutation_comms[7].y
                mstore(0x1040, 0x1a9da1add67b69948f6f12272d490201de6523e657209c35b1e3ea6d26b78fad) // permutation_comms[8].x
                mstore(0x1060, 0x2f8c1949d256ce1b15120b33f785a9409233afa8ee80b03895206fb462106d55) // permutation_comms[8].y

                // Read accumulator from instances
                if mload(HAS_ACCUMULATOR_MPTR) {
                    let num_limbs := mload(NUM_ACC_LIMBS_MPTR)
                    let num_limb_bits := mload(NUM_ACC_LIMB_BITS_MPTR)

                    let cptr := add(instances.offset, mul(mload(ACC_OFFSET_MPTR), 0x20))
                    let lhs_y_off := mul(num_limbs, 0x20)
                    let rhs_x_off := mul(lhs_y_off, 2)
                    let rhs_y_off := mul(lhs_y_off, 3)
                    let lhs_x := calldataload(cptr)
                    let lhs_y := calldataload(add(cptr, lhs_y_off))
                    let rhs_x := calldataload(add(cptr, rhs_x_off))
                    let rhs_y := calldataload(add(cptr, rhs_y_off))
                    for
                        {
                            let cptr_end := add(cptr, mul(0x20, num_limbs))
                            let shift := num_limb_bits
                        }
                        lt(cptr, cptr_end)
                        {}
                    {
                        cptr := add(cptr, 0x20)
                        lhs_x := add(lhs_x, shl(shift, calldataload(cptr)))
                        lhs_y := add(lhs_y, shl(shift, calldataload(add(cptr, lhs_y_off))))
                        rhs_x := add(rhs_x, shl(shift, calldataload(add(cptr, rhs_x_off))))
                        rhs_y := add(rhs_y, shl(shift, calldataload(add(cptr, rhs_y_off))))
                        shift := add(shift, num_limb_bits)
                    }

                    success := and(success, eq(mulmod(lhs_y, lhs_y, q), addmod(mulmod(lhs_x, mulmod(lhs_x, lhs_x, q), q), 3, q)))
                    success := and(success, eq(mulmod(rhs_y, rhs_y, q), addmod(mulmod(rhs_x, mulmod(rhs_x, rhs_x, q), q), 3, q)))

                    mstore(ACC_LHS_X_MPTR, lhs_x)
                    mstore(ACC_LHS_Y_MPTR, lhs_y)
                    mstore(ACC_RHS_X_MPTR, rhs_x)
                    mstore(ACC_RHS_Y_MPTR, rhs_y)
                }

                pop(q)
            }

            // Revert earlier if anything from calldata is invalid
            if iszero(success) {
                revert(0, 0)
            }

            // Compute lagrange evaluations and instance evaluation
            {
                let k := mload(K_MPTR)
                let x := mload(X_MPTR)
                let x_n := x
                for
                    { let idx := 0 }
                    lt(idx, k)
                    { idx := add(idx, 1) }
                {
                    x_n := mulmod(x_n, x_n, r)
                }

                let omega := mload(OMEGA_MPTR)

                let mptr := X_N_MPTR
                let mptr_end := add(mptr, mul(0x20, add(mload(NUM_INSTANCES_MPTR), 6)))
                if iszero(mload(NUM_INSTANCES_MPTR)) {
                    mptr_end := add(mptr_end, 0x20)
                }
                for
                    { let pow_of_omega := mload(OMEGA_INV_TO_L_MPTR) }
                    lt(mptr, mptr_end)
                    { mptr := add(mptr, 0x20) }
                {
                    mstore(mptr, addmod(x, sub(r, pow_of_omega), r))
                    pow_of_omega := mulmod(pow_of_omega, omega, r)
                }
                let x_n_minus_1 := addmod(x_n, sub(r, 1), r)
                mstore(mptr_end, x_n_minus_1)
                success := batch_invert(success, X_N_MPTR, add(mptr_end, 0x20))

                mptr := X_N_MPTR
                let l_i_common := mulmod(x_n_minus_1, mload(N_INV_MPTR), r)
                for
                    { let pow_of_omega := mload(OMEGA_INV_TO_L_MPTR) }
                    lt(mptr, mptr_end)
                    { mptr := add(mptr, 0x20) }
                {
                    mstore(mptr, mulmod(l_i_common, mulmod(mload(mptr), pow_of_omega, r), r))
                    pow_of_omega := mulmod(pow_of_omega, omega, r)
                }

                let l_blind := mload(add(X_N_MPTR, 0x20))
                let l_i_cptr := add(X_N_MPTR, 0x40)
                for
                    { let l_i_cptr_end := add(X_N_MPTR, 0xc0) }
                    lt(l_i_cptr, l_i_cptr_end)
                    { l_i_cptr := add(l_i_cptr, 0x20) }
                {
                    l_blind := addmod(l_blind, mload(l_i_cptr), r)
                }

                let instance_eval := 0
                for
                    {
                        let instance_cptr := instances.offset
                        let instance_cptr_end := add(instance_cptr, mul(0x20, mload(NUM_INSTANCES_MPTR)))
                    }
                    lt(instance_cptr, instance_cptr_end)
                    {
                        instance_cptr := add(instance_cptr, 0x20)
                        l_i_cptr := add(l_i_cptr, 0x20)
                    }
                {
                    instance_eval := addmod(instance_eval, mulmod(mload(l_i_cptr), calldataload(instance_cptr), r), r)
                }

                let x_n_minus_1_inv := mload(mptr_end)
                let l_last := mload(X_N_MPTR)
                let l_0 := mload(add(X_N_MPTR, 0xc0))

                mstore(X_N_MPTR, x_n)
                mstore(X_N_MINUS_1_INV_MPTR, x_n_minus_1_inv)
                mstore(L_LAST_MPTR, l_last)
                mstore(L_BLIND_MPTR, l_blind)
                mstore(L_0_MPTR, l_0)
                mstore(INSTANCE_EVAL_MPTR, instance_eval)
            }

            // Compute quotient evavluation
            {
                let quotient_eval_numer
                let y := mload(Y_MPTR)
                {
                    let f_11 := calldataload(0x0984)
                    let var0 := 0x2
                    let var1 := sub(R, f_11)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_11, var2, R)
                    let var4 := 0x3
                    let var5 := addmod(var4, var1, R)
                    let var6 := mulmod(var3, var5, R)
                    let a_5 := calldataload(0x07c4)
                    let a_1 := calldataload(0x0744)
                    let a_3 := calldataload(0x0784)
                    let var7 := addmod(a_1, a_3, R)
                    let var8 := sub(R, var7)
                    let var9 := addmod(a_5, var8, R)
                    let var10 := mulmod(var6, var9, R)
                    quotient_eval_numer := var10
                }
                {
                    let f_12 := calldataload(0x09a4)
                    let var0 := 0x2
                    let var1 := sub(R, f_12)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_12, var2, R)
                    let var4 := 0x3
                    let var5 := addmod(var4, var1, R)
                    let var6 := mulmod(var3, var5, R)
                    let a_6 := calldataload(0x07e4)
                    let a_2 := calldataload(0x0764)
                    let a_4 := calldataload(0x07a4)
                    let var7 := addmod(a_2, a_4, R)
                    let var8 := sub(R, var7)
                    let var9 := addmod(a_6, var8, R)
                    let var10 := mulmod(var6, var9, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var10, r)
                }
                {
                    let f_11 := calldataload(0x0984)
                    let var0 := 0x1
                    let var1 := sub(R, f_11)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_11, var2, R)
                    let var4 := 0x2
                    let var5 := addmod(var4, var1, R)
                    let var6 := mulmod(var3, var5, R)
                    let a_5 := calldataload(0x07c4)
                    let a_1 := calldataload(0x0744)
                    let a_3 := calldataload(0x0784)
                    let var7 := mulmod(a_1, a_3, R)
                    let var8 := sub(R, var7)
                    let var9 := addmod(a_5, var8, R)
                    let var10 := mulmod(var6, var9, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var10, r)
                }
                {
                    let f_12 := calldataload(0x09a4)
                    let var0 := 0x1
                    let var1 := sub(R, f_12)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_12, var2, R)
                    let var4 := 0x2
                    let var5 := addmod(var4, var1, R)
                    let var6 := mulmod(var3, var5, R)
                    let a_6 := calldataload(0x07e4)
                    let a_2 := calldataload(0x0764)
                    let a_4 := calldataload(0x07a4)
                    let var7 := mulmod(a_2, a_4, R)
                    let var8 := sub(R, var7)
                    let var9 := addmod(a_6, var8, R)
                    let var10 := mulmod(var6, var9, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var10, r)
                }
                {
                    let f_11 := calldataload(0x0984)
                    let var0 := 0x1
                    let var1 := sub(R, f_11)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_11, var2, R)
                    let var4 := 0x3
                    let var5 := addmod(var4, var1, R)
                    let var6 := mulmod(var3, var5, R)
                    let a_5 := calldataload(0x07c4)
                    let a_1 := calldataload(0x0744)
                    let a_3 := calldataload(0x0784)
                    let var7 := sub(R, a_3)
                    let var8 := addmod(a_1, var7, R)
                    let var9 := sub(R, var8)
                    let var10 := addmod(a_5, var9, R)
                    let var11 := mulmod(var6, var10, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var11, r)
                }
                {
                    let f_12 := calldataload(0x09a4)
                    let var0 := 0x1
                    let var1 := sub(R, f_12)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_12, var2, R)
                    let var4 := 0x3
                    let var5 := addmod(var4, var1, R)
                    let var6 := mulmod(var3, var5, R)
                    let a_6 := calldataload(0x07e4)
                    let a_2 := calldataload(0x0764)
                    let a_4 := calldataload(0x07a4)
                    let var7 := sub(R, a_4)
                    let var8 := addmod(a_2, var7, R)
                    let var9 := sub(R, var8)
                    let var10 := addmod(a_6, var9, R)
                    let var11 := mulmod(var6, var10, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var11, r)
                }
                {
                    let f_13 := calldataload(0x09c4)
                    let var0 := 0x1
                    let var1 := sub(R, f_13)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_13, var2, R)
                    let var4 := 0x3
                    let var5 := addmod(var4, var1, R)
                    let var6 := mulmod(var3, var5, R)
                    let a_5 := calldataload(0x07c4)
                    let a_5_prev_1 := calldataload(0x0804)
                    let var7 := 0x0
                    let a_1 := calldataload(0x0744)
                    let a_3 := calldataload(0x0784)
                    let var8 := mulmod(a_1, a_3, R)
                    let var9 := addmod(var7, var8, R)
                    let a_2 := calldataload(0x0764)
                    let a_4 := calldataload(0x07a4)
                    let var10 := mulmod(a_2, a_4, R)
                    let var11 := addmod(var9, var10, R)
                    let var12 := addmod(a_5_prev_1, var11, R)
                    let var13 := sub(R, var12)
                    let var14 := addmod(a_5, var13, R)
                    let var15 := mulmod(var6, var14, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var15, r)
                }
                {
                    let f_13 := calldataload(0x09c4)
                    let var0 := 0x2
                    let var1 := sub(R, f_13)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_13, var2, R)
                    let var4 := 0x3
                    let var5 := addmod(var4, var1, R)
                    let var6 := mulmod(var3, var5, R)
                    let a_5 := calldataload(0x07c4)
                    let var7 := 0x0
                    let a_1 := calldataload(0x0744)
                    let a_3 := calldataload(0x0784)
                    let var8 := mulmod(a_1, a_3, R)
                    let var9 := addmod(var7, var8, R)
                    let a_2 := calldataload(0x0764)
                    let a_4 := calldataload(0x07a4)
                    let var10 := mulmod(a_2, a_4, R)
                    let var11 := addmod(var9, var10, R)
                    let var12 := sub(R, var11)
                    let var13 := addmod(a_5, var12, R)
                    let var14 := mulmod(var6, var13, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var14, r)
                }
                {
                    let f_13 := calldataload(0x09c4)
                    let var0 := 0x1
                    let var1 := sub(R, f_13)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_13, var2, R)
                    let var4 := 0x2
                    let var5 := addmod(var4, var1, R)
                    let var6 := mulmod(var3, var5, R)
                    let a_5 := calldataload(0x07c4)
                    let a_3 := calldataload(0x0784)
                    let var7 := mulmod(var0, a_3, R)
                    let a_4 := calldataload(0x07a4)
                    let var8 := mulmod(var7, a_4, R)
                    let var9 := sub(R, var8)
                    let var10 := addmod(a_5, var9, R)
                    let var11 := mulmod(var6, var10, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var11, r)
                }
                {
                    let f_14 := calldataload(0x09e4)
                    let var0 := 0x2
                    let var1 := sub(R, f_14)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_14, var2, R)
                    let a_5 := calldataload(0x07c4)
                    let a_5_prev_1 := calldataload(0x0804)
                    let var4 := 0x1
                    let a_3 := calldataload(0x0784)
                    let var5 := mulmod(var4, a_3, R)
                    let a_4 := calldataload(0x07a4)
                    let var6 := mulmod(var5, a_4, R)
                    let var7 := mulmod(a_5_prev_1, var6, R)
                    let var8 := sub(R, var7)
                    let var9 := addmod(a_5, var8, R)
                    let var10 := mulmod(var3, var9, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var10, r)
                }
                {
                    let f_15 := calldataload(0x0a04)
                    let a_5 := calldataload(0x07c4)
                    let var0 := 0x0
                    let a_3 := calldataload(0x0784)
                    let var1 := addmod(var0, a_3, R)
                    let a_4 := calldataload(0x07a4)
                    let var2 := addmod(var1, a_4, R)
                    let var3 := sub(R, var2)
                    let var4 := addmod(a_5, var3, R)
                    let var5 := mulmod(f_15, var4, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var5, r)
                }
                {
                    let f_14 := calldataload(0x09e4)
                    let var0 := 0x1
                    let var1 := sub(R, f_14)
                    let var2 := addmod(var0, var1, R)
                    let var3 := mulmod(f_14, var2, R)
                    let a_5 := calldataload(0x07c4)
                    let a_5_prev_1 := calldataload(0x0804)
                    let var4 := 0x0
                    let a_3 := calldataload(0x0784)
                    let var5 := addmod(var4, a_3, R)
                    let a_4 := calldataload(0x07a4)
                    let var6 := addmod(var5, a_4, R)
                    let var7 := addmod(a_5_prev_1, var6, R)
                    let var8 := sub(R, var7)
                    let var9 := addmod(a_5, var8, R)
                    let var10 := mulmod(var3, var9, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var10, r)
                }
                {
                    let f_5 := calldataload(0x08c4)
                    let var0 := 0x0
                    let var1 := mulmod(f_5, var0, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var1, r)
                }
                {
                    let f_6 := calldataload(0x08e4)
                    let var0 := 0x0
                    let var1 := mulmod(f_6, var0, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var1, r)
                }
                {
                    let f_7 := calldataload(0x0904)
                    let var0 := 0x0
                    let var1 := mulmod(f_7, var0, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var1, r)
                }
                {
                    let f_8 := calldataload(0x0924)
                    let var0 := 0x0
                    let var1 := mulmod(f_8, var0, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var1, r)
                }
                {
                    let f_9 := calldataload(0x0944)
                    let var0 := 0x0
                    let var1 := mulmod(f_9, var0, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var1, r)
                }
                {
                    let f_10 := calldataload(0x0964)
                    let var0 := 0x0
                    let var1 := mulmod(f_10, var0, R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), var1, r)
                }
                {
                    let l_0 := mload(L_0_MPTR)
                    let eval := addmod(l_0, sub(R, mulmod(l_0, calldataload(0x0b64), R)), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let perm_z_last := calldataload(0x0c24)
                    let eval := mulmod(mload(L_LAST_MPTR), addmod(mulmod(perm_z_last, perm_z_last, R), sub(R, perm_z_last), R), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let eval := mulmod(mload(L_0_MPTR), addmod(calldataload(0x0bc4), sub(R, calldataload(0x0ba4)), R), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let eval := mulmod(mload(L_0_MPTR), addmod(calldataload(0x0c24), sub(R, calldataload(0x0c04)), R), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let gamma := mload(GAMMA_MPTR)
                    let beta := mload(BETA_MPTR)
                    let lhs := calldataload(0x0b84)
                    let rhs := calldataload(0x0b64)
                    lhs := mulmod(lhs, addmod(addmod(calldataload(0x0724), mulmod(beta, calldataload(0x0a44), R), R), gamma, R), R)
                    lhs := mulmod(lhs, addmod(addmod(calldataload(0x0744), mulmod(beta, calldataload(0x0a64), R), R), gamma, R), R)
                    lhs := mulmod(lhs, addmod(addmod(calldataload(0x0764), mulmod(beta, calldataload(0x0a84), R), R), gamma, R), R)
                    mstore(0x00, mulmod(beta, mload(X_MPTR), R))
                    rhs := mulmod(rhs, addmod(addmod(calldataload(0x0724), mload(0x00), R), gamma, R), R)
                    mstore(0x00, mulmod(mload(0x00), DELTA, R))
                    rhs := mulmod(rhs, addmod(addmod(calldataload(0x0744), mload(0x00), R), gamma, R), R)
                    mstore(0x00, mulmod(mload(0x00), DELTA, R))
                    rhs := mulmod(rhs, addmod(addmod(calldataload(0x0764), mload(0x00), R), gamma, R), R)
                    mstore(0x00, mulmod(mload(0x00), DELTA, R))
                    let left_sub_right := addmod(lhs, sub(R, rhs), R)
                    let eval := addmod(left_sub_right, sub(R, mulmod(left_sub_right, addmod(mload(L_LAST_MPTR), mload(L_BLIND_MPTR), R), R)), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let gamma := mload(GAMMA_MPTR)
                    let beta := mload(BETA_MPTR)
                    let lhs := calldataload(0x0be4)
                    let rhs := calldataload(0x0bc4)
                    lhs := mulmod(lhs, addmod(addmod(calldataload(0x0784), mulmod(beta, calldataload(0x0aa4), R), R), gamma, R), R)
                    lhs := mulmod(lhs, addmod(addmod(calldataload(0x07a4), mulmod(beta, calldataload(0x0ac4), R), R), gamma, R), R)
                    lhs := mulmod(lhs, addmod(addmod(calldataload(0x07c4), mulmod(beta, calldataload(0x0ae4), R), R), gamma, R), R)
                    rhs := mulmod(rhs, addmod(addmod(calldataload(0x0784), mload(0x00), R), gamma, R), R)
                    mstore(0x00, mulmod(mload(0x00), DELTA, R))
                    rhs := mulmod(rhs, addmod(addmod(calldataload(0x07a4), mload(0x00), R), gamma, R), R)
                    mstore(0x00, mulmod(mload(0x00), DELTA, R))
                    rhs := mulmod(rhs, addmod(addmod(calldataload(0x07c4), mload(0x00), R), gamma, R), R)
                    mstore(0x00, mulmod(mload(0x00), DELTA, R))
                    let left_sub_right := addmod(lhs, sub(R, rhs), R)
                    let eval := addmod(left_sub_right, sub(R, mulmod(left_sub_right, addmod(mload(L_LAST_MPTR), mload(L_BLIND_MPTR), R), R)), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let gamma := mload(GAMMA_MPTR)
                    let beta := mload(BETA_MPTR)
                    let lhs := calldataload(0x0c44)
                    let rhs := calldataload(0x0c24)
                    lhs := mulmod(lhs, addmod(addmod(calldataload(0x07e4), mulmod(beta, calldataload(0x0b04), R), R), gamma, R), R)
                    lhs := mulmod(lhs, addmod(addmod(calldataload(0x0824), mulmod(beta, calldataload(0x0b24), R), R), gamma, R), R)
                    lhs := mulmod(lhs, addmod(addmod(mload(INSTANCE_EVAL_MPTR), mulmod(beta, calldataload(0x0b44), R), R), gamma, R), R)
                    rhs := mulmod(rhs, addmod(addmod(calldataload(0x07e4), mload(0x00), R), gamma, R), R)
                    mstore(0x00, mulmod(mload(0x00), DELTA, R))
                    rhs := mulmod(rhs, addmod(addmod(calldataload(0x0824), mload(0x00), R), gamma, R), R)
                    mstore(0x00, mulmod(mload(0x00), DELTA, R))
                    rhs := mulmod(rhs, addmod(addmod(mload(INSTANCE_EVAL_MPTR), mload(0x00), R), gamma, R), R)
                    let left_sub_right := addmod(lhs, sub(R, rhs), R)
                    let eval := addmod(left_sub_right, sub(R, mulmod(left_sub_right, addmod(mload(L_LAST_MPTR), mload(L_BLIND_MPTR), R), R)), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_0 := mload(L_0_MPTR)
                    let eval := mulmod(l_0, calldataload(0x0c64), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_last := mload(L_LAST_MPTR)
                    let eval := mulmod(l_last, calldataload(0x0c64), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let theta := mload(THETA_MPTR)
                    let beta := mload(BETA_MPTR)
                    let table
                    {
                        let f_1 := calldataload(0x0844)
                        let f_2 := calldataload(0x0864)
                        table := f_1
                        table := addmod(mulmod(table, theta, R), f_2, R)
                        table := addmod(table, beta, R)
                    }
                    let input_0
                    {
                        let f_5 := calldataload(0x08c4)
                        let var0 := 0x1
                        let var1 := mulmod(f_5, var0, R)
                        let a_1 := calldataload(0x0744)
                        let var2 := mulmod(var1, a_1, R)
                        let var3 := sub(R, var1)
                        let var4 := addmod(var0, var3, R)
                        let var5 := 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efff9d69
                        let var6 := mulmod(var4, var5, R)
                        let var7 := addmod(var2, var6, R)
                        let a_5 := calldataload(0x07c4)
                        let var8 := mulmod(var1, a_5, R)
                        let var9 := 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffe023
                        let var10 := mulmod(var4, var9, R)
                        let var11 := addmod(var8, var10, R)
                        input_0 := var7
                        input_0 := addmod(mulmod(input_0, theta, R), var11, R)
                        input_0 := addmod(input_0, beta, R)
                    }
                    let lhs
                    let rhs
                    rhs := table
                    {
                        let tmp := input_0
                        rhs := addmod(rhs, sub(R, mulmod(calldataload(0x0ca4), tmp, R)), R)
                        lhs := mulmod(mulmod(table, tmp, R), addmod(calldataload(0x0c84), sub(R, calldataload(0x0c64)), R), R)
                    }
                    let eval := mulmod(addmod(1, sub(R, addmod(mload(L_BLIND_MPTR), mload(L_LAST_MPTR), R)), R), addmod(lhs, sub(R, rhs), R), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_0 := mload(L_0_MPTR)
                    let eval := mulmod(l_0, calldataload(0x0cc4), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_last := mload(L_LAST_MPTR)
                    let eval := mulmod(l_last, calldataload(0x0cc4), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let theta := mload(THETA_MPTR)
                    let beta := mload(BETA_MPTR)
                    let table
                    {
                        let f_1 := calldataload(0x0844)
                        let f_2 := calldataload(0x0864)
                        table := f_1
                        table := addmod(mulmod(table, theta, R), f_2, R)
                        table := addmod(table, beta, R)
                    }
                    let input_0
                    {
                        let f_6 := calldataload(0x08e4)
                        let var0 := 0x1
                        let var1 := mulmod(f_6, var0, R)
                        let a_2 := calldataload(0x0764)
                        let var2 := mulmod(var1, a_2, R)
                        let var3 := sub(R, var1)
                        let var4 := addmod(var0, var3, R)
                        let var5 := 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efff9d69
                        let var6 := mulmod(var4, var5, R)
                        let var7 := addmod(var2, var6, R)
                        let a_6 := calldataload(0x07e4)
                        let var8 := mulmod(var1, a_6, R)
                        let var9 := 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffe023
                        let var10 := mulmod(var4, var9, R)
                        let var11 := addmod(var8, var10, R)
                        input_0 := var7
                        input_0 := addmod(mulmod(input_0, theta, R), var11, R)
                        input_0 := addmod(input_0, beta, R)
                    }
                    let lhs
                    let rhs
                    rhs := table
                    {
                        let tmp := input_0
                        rhs := addmod(rhs, sub(R, mulmod(calldataload(0x0d04), tmp, R)), R)
                        lhs := mulmod(mulmod(table, tmp, R), addmod(calldataload(0x0ce4), sub(R, calldataload(0x0cc4)), R), R)
                    }
                    let eval := mulmod(addmod(1, sub(R, addmod(mload(L_BLIND_MPTR), mload(L_LAST_MPTR), R)), R), addmod(lhs, sub(R, rhs), R), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_0 := mload(L_0_MPTR)
                    let eval := mulmod(l_0, calldataload(0x0d24), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_last := mload(L_LAST_MPTR)
                    let eval := mulmod(l_last, calldataload(0x0d24), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let theta := mload(THETA_MPTR)
                    let beta := mload(BETA_MPTR)
                    let table
                    {
                        let f_3 := calldataload(0x0884)
                        table := f_3
                        table := addmod(table, beta, R)
                    }
                    let input_0
                    {
                        let f_7 := calldataload(0x0904)
                        let var0 := 0x1
                        let var1 := mulmod(f_7, var0, R)
                        let a_1 := calldataload(0x0744)
                        let var2 := mulmod(var1, a_1, R)
                        let var3 := sub(R, var1)
                        let var4 := addmod(var0, var3, R)
                        let var5 := 0x0
                        let var6 := mulmod(var4, var5, R)
                        let var7 := addmod(var2, var6, R)
                        input_0 := var7
                        input_0 := addmod(input_0, beta, R)
                    }
                    let lhs
                    let rhs
                    rhs := table
                    {
                        let tmp := input_0
                        rhs := addmod(rhs, sub(R, mulmod(calldataload(0x0d64), tmp, R)), R)
                        lhs := mulmod(mulmod(table, tmp, R), addmod(calldataload(0x0d44), sub(R, calldataload(0x0d24)), R), R)
                    }
                    let eval := mulmod(addmod(1, sub(R, addmod(mload(L_BLIND_MPTR), mload(L_LAST_MPTR), R)), R), addmod(lhs, sub(R, rhs), R), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_0 := mload(L_0_MPTR)
                    let eval := mulmod(l_0, calldataload(0x0d84), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_last := mload(L_LAST_MPTR)
                    let eval := mulmod(l_last, calldataload(0x0d84), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let theta := mload(THETA_MPTR)
                    let beta := mload(BETA_MPTR)
                    let table
                    {
                        let f_3 := calldataload(0x0884)
                        table := f_3
                        table := addmod(table, beta, R)
                    }
                    let input_0
                    {
                        let f_8 := calldataload(0x0924)
                        let var0 := 0x1
                        let var1 := mulmod(f_8, var0, R)
                        let a_2 := calldataload(0x0764)
                        let var2 := mulmod(var1, a_2, R)
                        let var3 := sub(R, var1)
                        let var4 := addmod(var0, var3, R)
                        let var5 := 0x0
                        let var6 := mulmod(var4, var5, R)
                        let var7 := addmod(var2, var6, R)
                        input_0 := var7
                        input_0 := addmod(input_0, beta, R)
                    }
                    let lhs
                    let rhs
                    rhs := table
                    {
                        let tmp := input_0
                        rhs := addmod(rhs, sub(R, mulmod(calldataload(0x0dc4), tmp, R)), R)
                        lhs := mulmod(mulmod(table, tmp, R), addmod(calldataload(0x0da4), sub(R, calldataload(0x0d84)), R), R)
                    }
                    let eval := mulmod(addmod(1, sub(R, addmod(mload(L_BLIND_MPTR), mload(L_LAST_MPTR), R)), R), addmod(lhs, sub(R, rhs), R), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_0 := mload(L_0_MPTR)
                    let eval := mulmod(l_0, calldataload(0x0de4), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_last := mload(L_LAST_MPTR)
                    let eval := mulmod(l_last, calldataload(0x0de4), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let theta := mload(THETA_MPTR)
                    let beta := mload(BETA_MPTR)
                    let table
                    {
                        let f_4 := calldataload(0x08a4)
                        table := f_4
                        table := addmod(table, beta, R)
                    }
                    let input_0
                    {
                        let f_9 := calldataload(0x0944)
                        let var0 := 0x1
                        let var1 := mulmod(f_9, var0, R)
                        let a_1 := calldataload(0x0744)
                        let var2 := mulmod(var1, a_1, R)
                        let var3 := sub(R, var1)
                        let var4 := addmod(var0, var3, R)
                        let var5 := 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000
                        let var6 := mulmod(var4, var5, R)
                        let var7 := addmod(var2, var6, R)
                        input_0 := var7
                        input_0 := addmod(input_0, beta, R)
                    }
                    let lhs
                    let rhs
                    rhs := table
                    {
                        let tmp := input_0
                        rhs := addmod(rhs, sub(R, mulmod(calldataload(0x0e24), tmp, R)), R)
                        lhs := mulmod(mulmod(table, tmp, R), addmod(calldataload(0x0e04), sub(R, calldataload(0x0de4)), R), R)
                    }
                    let eval := mulmod(addmod(1, sub(R, addmod(mload(L_BLIND_MPTR), mload(L_LAST_MPTR), R)), R), addmod(lhs, sub(R, rhs), R), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_0 := mload(L_0_MPTR)
                    let eval := mulmod(l_0, calldataload(0x0e44), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let l_last := mload(L_LAST_MPTR)
                    let eval := mulmod(l_last, calldataload(0x0e44), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }
                {
                    let theta := mload(THETA_MPTR)
                    let beta := mload(BETA_MPTR)
                    let table
                    {
                        let f_4 := calldataload(0x08a4)
                        table := f_4
                        table := addmod(table, beta, R)
                    }
                    let input_0
                    {
                        let f_10 := calldataload(0x0964)
                        let var0 := 0x1
                        let var1 := mulmod(f_10, var0, R)
                        let a_2 := calldataload(0x0764)
                        let var2 := mulmod(var1, a_2, R)
                        let var3 := sub(R, var1)
                        let var4 := addmod(var0, var3, R)
                        let var5 := 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000
                        let var6 := mulmod(var4, var5, R)
                        let var7 := addmod(var2, var6, R)
                        input_0 := var7
                        input_0 := addmod(input_0, beta, R)
                    }
                    let lhs
                    let rhs
                    rhs := table
                    {
                        let tmp := input_0
                        rhs := addmod(rhs, sub(R, mulmod(calldataload(0x0e84), tmp, R)), R)
                        lhs := mulmod(mulmod(table, tmp, R), addmod(calldataload(0x0e64), sub(R, calldataload(0x0e44)), R), R)
                    }
                    let eval := mulmod(addmod(1, sub(R, addmod(mload(L_BLIND_MPTR), mload(L_LAST_MPTR), R)), R), addmod(lhs, sub(R, rhs), R), R)
                    quotient_eval_numer := addmod(mulmod(quotient_eval_numer, y, r), eval, r)
                }

                pop(y)

                let quotient_eval := mulmod(quotient_eval_numer, mload(X_N_MINUS_1_INV_MPTR), r)
                mstore(QUOTIENT_EVAL_MPTR, quotient_eval)
            }

            // Compute quotient commitment
            {
                mstore(0x00, calldataload(LAST_QUOTIENT_X_CPTR))
                mstore(0x20, calldataload(add(LAST_QUOTIENT_X_CPTR, 0x20)))
                let x_n := mload(X_N_MPTR)
                for
                    {
                        let cptr := sub(LAST_QUOTIENT_X_CPTR, 0x40)
                        let cptr_end := sub(FIRST_QUOTIENT_X_CPTR, 0x40)
                    }
                    lt(cptr_end, cptr)
                    {}
                {
                    success := ec_mul_acc(success, x_n)
                    success := ec_add_acc(success, calldataload(cptr), calldataload(add(cptr, 0x20)))
                    cptr := sub(cptr, 0x40)
                }
                mstore(QUOTIENT_X_MPTR, mload(0x00))
                mstore(QUOTIENT_Y_MPTR, mload(0x20))
            }

            // Compute pairing lhs and rhs
            {
                {
                    let x := mload(X_MPTR)
                    let omega := mload(OMEGA_MPTR)
                    let omega_inv := mload(OMEGA_INV_MPTR)
                    let x_pow_of_omega := mulmod(x, omega, R)
                    mstore(0x0360, x_pow_of_omega)
                    mstore(0x0340, x)
                    x_pow_of_omega := mulmod(x, omega_inv, R)
                    mstore(0x0320, x_pow_of_omega)
                    x_pow_of_omega := mulmod(x_pow_of_omega, omega_inv, R)
                    x_pow_of_omega := mulmod(x_pow_of_omega, omega_inv, R)
                    x_pow_of_omega := mulmod(x_pow_of_omega, omega_inv, R)
                    x_pow_of_omega := mulmod(x_pow_of_omega, omega_inv, R)
                    x_pow_of_omega := mulmod(x_pow_of_omega, omega_inv, R)
                    mstore(0x0300, x_pow_of_omega)
                }
                {
                    let mu := mload(MU_MPTR)
                    for
                        {
                            let mptr := 0x0380
                            let mptr_end := 0x0400
                            let point_mptr := 0x0300
                        }
                        lt(mptr, mptr_end)
                        {
                            mptr := add(mptr, 0x20)
                            point_mptr := add(point_mptr, 0x20)
                        }
                    {
                        mstore(mptr, addmod(mu, sub(R, mload(point_mptr)), R))
                    }
                    let s
                    s := mload(0x03c0)
                    mstore(0x0400, s)
                    let diff
                    diff := mload(0x0380)
                    diff := mulmod(diff, mload(0x03a0), R)
                    diff := mulmod(diff, mload(0x03e0), R)
                    mstore(0x0420, diff)
                    mstore(0x00, diff)
                    diff := mload(0x0380)
                    diff := mulmod(diff, mload(0x03e0), R)
                    mstore(0x0440, diff)
                    diff := mload(0x03a0)
                    mstore(0x0460, diff)
                    diff := mload(0x0380)
                    diff := mulmod(diff, mload(0x03a0), R)
                    mstore(0x0480, diff)
                }
                {
                    let point_2 := mload(0x0340)
                    let coeff
                    coeff := 1
                    coeff := mulmod(coeff, mload(0x03c0), R)
                    mstore(0x20, coeff)
                }
                {
                    let point_1 := mload(0x0320)
                    let point_2 := mload(0x0340)
                    let coeff
                    coeff := addmod(point_1, sub(R, point_2), R)
                    coeff := mulmod(coeff, mload(0x03a0), R)
                    mstore(0x40, coeff)
                    coeff := addmod(point_2, sub(R, point_1), R)
                    coeff := mulmod(coeff, mload(0x03c0), R)
                    mstore(0x60, coeff)
                }
                {
                    let point_0 := mload(0x0300)
                    let point_2 := mload(0x0340)
                    let point_3 := mload(0x0360)
                    let coeff
                    coeff := addmod(point_0, sub(R, point_2), R)
                    coeff := mulmod(coeff, addmod(point_0, sub(R, point_3), R), R)
                    coeff := mulmod(coeff, mload(0x0380), R)
                    mstore(0x80, coeff)
                    coeff := addmod(point_2, sub(R, point_0), R)
                    coeff := mulmod(coeff, addmod(point_2, sub(R, point_3), R), R)
                    coeff := mulmod(coeff, mload(0x03c0), R)
                    mstore(0xa0, coeff)
                    coeff := addmod(point_3, sub(R, point_0), R)
                    coeff := mulmod(coeff, addmod(point_3, sub(R, point_2), R), R)
                    coeff := mulmod(coeff, mload(0x03e0), R)
                    mstore(0xc0, coeff)
                }
                {
                    let point_2 := mload(0x0340)
                    let point_3 := mload(0x0360)
                    let coeff
                    coeff := addmod(point_2, sub(R, point_3), R)
                    coeff := mulmod(coeff, mload(0x03c0), R)
                    mstore(0xe0, coeff)
                    coeff := addmod(point_3, sub(R, point_2), R)
                    coeff := mulmod(coeff, mload(0x03e0), R)
                    mstore(0x0100, coeff)
                }
                {
                    success := batch_invert(success, 0, 0x0120)
                    let diff_0_inv := mload(0x00)
                    mstore(0x0420, diff_0_inv)
                    for
                        {
                            let mptr := 0x0440
                            let mptr_end := 0x04a0
                        }
                        lt(mptr, mptr_end)
                        { mptr := add(mptr, 0x20) }
                    {
                        mstore(mptr, mulmod(mload(mptr), diff_0_inv, R))
                    }
                }
                {
                    let coeff := mload(0x20)
                    let zeta := mload(ZETA_MPTR)
                    let r_eval := 0
                    r_eval := addmod(r_eval, mulmod(coeff, calldataload(0x0a24), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(coeff, mload(QUOTIENT_EVAL_MPTR), R), R)
                    for
                        {
                            let mptr := 0x0b44
                            let mptr_end := 0x0a24
                        }
                        lt(mptr_end, mptr)
                        { mptr := sub(mptr, 0x20) }
                    {
                        r_eval := addmod(mulmod(r_eval, zeta, R), mulmod(coeff, calldataload(mptr), R), R)
                    }
                    for
                        {
                            let mptr := 0x0a04
                            let mptr_end := 0x0804
                        }
                        lt(mptr_end, mptr)
                        { mptr := sub(mptr, 0x20) }
                    {
                        r_eval := addmod(mulmod(r_eval, zeta, R), mulmod(coeff, calldataload(mptr), R), R)
                    }
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(coeff, calldataload(0x0e84), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(coeff, calldataload(0x0e24), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(coeff, calldataload(0x0dc4), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(coeff, calldataload(0x0d64), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(coeff, calldataload(0x0d04), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(coeff, calldataload(0x0ca4), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(coeff, calldataload(0x07e4), R), R)
                    for
                        {
                            let mptr := 0x07a4
                            let mptr_end := 0x0704
                        }
                        lt(mptr_end, mptr)
                        { mptr := sub(mptr, 0x20) }
                    {
                        r_eval := addmod(mulmod(r_eval, zeta, R), mulmod(coeff, calldataload(mptr), R), R)
                    }
                    mstore(0x04a0, r_eval)
                }
                {
                    let zeta := mload(ZETA_MPTR)
                    let r_eval := 0
                    r_eval := addmod(r_eval, mulmod(mload(0x40), calldataload(0x0804), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0x60), calldataload(0x07c4), R), R)
                    r_eval := mulmod(r_eval, mload(0x0440), R)
                    mstore(0x04c0, r_eval)
                }
                {
                    let zeta := mload(ZETA_MPTR)
                    let r_eval := 0
                    r_eval := addmod(r_eval, mulmod(mload(0x80), calldataload(0x0c04), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0xa0), calldataload(0x0bc4), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0xc0), calldataload(0x0be4), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(mload(0x80), calldataload(0x0ba4), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0xa0), calldataload(0x0b64), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0xc0), calldataload(0x0b84), R), R)
                    r_eval := mulmod(r_eval, mload(0x0460), R)
                    mstore(0x04e0, r_eval)
                }
                {
                    let zeta := mload(ZETA_MPTR)
                    let r_eval := 0
                    r_eval := addmod(r_eval, mulmod(mload(0xe0), calldataload(0x0e44), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0x0100), calldataload(0x0e64), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(mload(0xe0), calldataload(0x0de4), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0x0100), calldataload(0x0e04), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(mload(0xe0), calldataload(0x0d84), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0x0100), calldataload(0x0da4), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(mload(0xe0), calldataload(0x0d24), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0x0100), calldataload(0x0d44), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(mload(0xe0), calldataload(0x0cc4), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0x0100), calldataload(0x0ce4), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(mload(0xe0), calldataload(0x0c64), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0x0100), calldataload(0x0c84), R), R)
                    r_eval := mulmod(r_eval, zeta, R)
                    r_eval := addmod(r_eval, mulmod(mload(0xe0), calldataload(0x0c24), R), R)
                    r_eval := addmod(r_eval, mulmod(mload(0x0100), calldataload(0x0c44), R), R)
                    r_eval := mulmod(r_eval, mload(0x0480), R)
                    mstore(0x0500, r_eval)
                }
                {
                    let sum := mload(0x20)
                    mstore(0x0520, sum)
                }
                {
                    let sum := mload(0x40)
                    sum := addmod(sum, mload(0x60), R)
                    mstore(0x0540, sum)
                }
                {
                    let sum := mload(0x80)
                    sum := addmod(sum, mload(0xa0), R)
                    sum := addmod(sum, mload(0xc0), R)
                    mstore(0x0560, sum)
                }
                {
                    let sum := mload(0xe0)
                    sum := addmod(sum, mload(0x0100), R)
                    mstore(0x0580, sum)
                }
                {
                    for
                        {
                            let mptr := 0x00
                            let mptr_end := 0x80
                            let sum_mptr := 0x0520
                        }
                        lt(mptr, mptr_end)
                        {
                            mptr := add(mptr, 0x20)
                            sum_mptr := add(sum_mptr, 0x20)
                        }
                    {
                        mstore(mptr, mload(sum_mptr))
                    }
                    success := batch_invert(success, 0, 0x80)
                    let r_eval := mulmod(mload(0x60), mload(0x0500), R)
                    for
                        {
                            let sum_inv_mptr := 0x40
                            let sum_inv_mptr_end := 0x80
                            let r_eval_mptr := 0x04e0
                        }
                        lt(sum_inv_mptr, sum_inv_mptr_end)
                        {
                            sum_inv_mptr := sub(sum_inv_mptr, 0x20)
                            r_eval_mptr := sub(r_eval_mptr, 0x20)
                        }
                    {
                        r_eval := mulmod(r_eval, mload(NU_MPTR), R)
                        r_eval := addmod(r_eval, mulmod(mload(sum_inv_mptr), mload(r_eval_mptr), R), R)
                    }
                    mstore(R_EVAL_MPTR, r_eval)
                }
                {
                    let nu := mload(NU_MPTR)
                    mstore(0x00, calldataload(0x05e4))
                    mstore(0x20, calldataload(0x0604))
                    success := ec_mul_acc(success, mload(ZETA_MPTR))
                    success := ec_add_acc(success, mload(QUOTIENT_X_MPTR), mload(QUOTIENT_Y_MPTR))
                    for
                        {
                            let mptr := 0x1040
                            let mptr_end := 0x0a00
                        }
                        lt(mptr_end, mptr)
                        { mptr := sub(mptr, 0x40) }
                    {
                        success := ec_mul_acc(success, mload(ZETA_MPTR))
                        success := ec_add_acc(success, mload(mptr), mload(add(mptr, 0x20)))
                    }
                    for
                        {
                            let mptr := 0x0364
                            let mptr_end := 0x01a4
                        }
                        lt(mptr_end, mptr)
                        { mptr := sub(mptr, 0x40) }
                    {
                        success := ec_mul_acc(success, mload(ZETA_MPTR))
                        success := ec_add_acc(success, calldataload(mptr), calldataload(add(mptr, 0x20)))
                    }
                    for
                        {
                            let mptr := 0x0164
                            let mptr_end := 0x24
                        }
                        lt(mptr_end, mptr)
                        { mptr := sub(mptr, 0x40) }
                    {
                        success := ec_mul_acc(success, mload(ZETA_MPTR))
                        success := ec_add_acc(success, calldataload(mptr), calldataload(add(mptr, 0x20)))
                    }
                    mstore(0x80, calldataload(0x01a4))
                    mstore(0xa0, calldataload(0x01c4))
                    success := ec_mul_tmp(success, mulmod(nu, mload(0x0440), R))
                    success := ec_add_acc(success, mload(0x80), mload(0xa0))
                    nu := mulmod(nu, mload(NU_MPTR), R)
                    mstore(0x80, calldataload(0x03e4))
                    mstore(0xa0, calldataload(0x0404))
                    success := ec_mul_tmp(success, mload(ZETA_MPTR))
                    success := ec_add_tmp(success, calldataload(0x03a4), calldataload(0x03c4))
                    success := ec_mul_tmp(success, mulmod(nu, mload(0x0460), R))
                    success := ec_add_acc(success, mload(0x80), mload(0xa0))
                    nu := mulmod(nu, mload(NU_MPTR), R)
                    mstore(0x80, calldataload(0x05a4))
                    mstore(0xa0, calldataload(0x05c4))
                    for
                        {
                            let mptr := 0x0564
                            let mptr_end := 0x03e4
                        }
                        lt(mptr_end, mptr)
                        { mptr := sub(mptr, 0x40) }
                    {
                        success := ec_mul_tmp(success, mload(ZETA_MPTR))
                        success := ec_add_tmp(success, calldataload(mptr), calldataload(add(mptr, 0x20)))
                    }
                    success := ec_mul_tmp(success, mulmod(nu, mload(0x0480), R))
                    success := ec_add_acc(success, mload(0x80), mload(0xa0))
                    mstore(0x80, mload(G1_X_MPTR))
                    mstore(0xa0, mload(G1_Y_MPTR))
                    success := ec_mul_tmp(success, sub(R, mload(R_EVAL_MPTR)))
                    success := ec_add_acc(success, mload(0x80), mload(0xa0))
                    mstore(0x80, calldataload(0x0ea4))
                    mstore(0xa0, calldataload(0x0ec4))
                    success := ec_mul_tmp(success, sub(R, mload(0x0400)))
                    success := ec_add_acc(success, mload(0x80), mload(0xa0))
                    mstore(0x80, calldataload(0x0ee4))
                    mstore(0xa0, calldataload(0x0f04))
                    success := ec_mul_tmp(success, mload(MU_MPTR))
                    success := ec_add_acc(success, mload(0x80), mload(0xa0))
                    mstore(PAIRING_LHS_X_MPTR, mload(0x00))
                    mstore(PAIRING_LHS_Y_MPTR, mload(0x20))
                    mstore(PAIRING_RHS_X_MPTR, calldataload(0x0ee4))
                    mstore(PAIRING_RHS_Y_MPTR, calldataload(0x0f04))
                }
            }

            // Random linear combine with accumulator
            if mload(HAS_ACCUMULATOR_MPTR) {
                mstore(0x00, mload(ACC_LHS_X_MPTR))
                mstore(0x20, mload(ACC_LHS_Y_MPTR))
                mstore(0x40, mload(ACC_RHS_X_MPTR))
                mstore(0x60, mload(ACC_RHS_Y_MPTR))
                mstore(0x80, mload(PAIRING_LHS_X_MPTR))
                mstore(0xa0, mload(PAIRING_LHS_Y_MPTR))
                mstore(0xc0, mload(PAIRING_RHS_X_MPTR))
                mstore(0xe0, mload(PAIRING_RHS_Y_MPTR))
                let challenge := mod(keccak256(0x00, 0x100), r)

                // [pairing_lhs] += challenge * [acc_lhs]
                success := ec_mul_acc(success, challenge)
                success := ec_add_acc(success, mload(PAIRING_LHS_X_MPTR), mload(PAIRING_LHS_Y_MPTR))
                mstore(PAIRING_LHS_X_MPTR, mload(0x00))
                mstore(PAIRING_LHS_Y_MPTR, mload(0x20))

                // [pairing_rhs] += challenge * [acc_rhs]
                mstore(0x00, mload(ACC_RHS_X_MPTR))
                mstore(0x20, mload(ACC_RHS_Y_MPTR))
                success := ec_mul_acc(success, challenge)
                success := ec_add_acc(success, mload(PAIRING_RHS_X_MPTR), mload(PAIRING_RHS_Y_MPTR))
                mstore(PAIRING_RHS_X_MPTR, mload(0x00))
                mstore(PAIRING_RHS_Y_MPTR, mload(0x20))
            }

            // Perform pairing
            success := ec_pairing(
                success,
                mload(PAIRING_LHS_X_MPTR),
                mload(PAIRING_LHS_Y_MPTR),
                mload(PAIRING_RHS_X_MPTR),
                mload(PAIRING_RHS_Y_MPTR)
            )

            // Revert if anything fails
            if iszero(success) {
                revert(0x00, 0x00)
            }

            // Return 1 as result if everything succeeds
            mstore(0x00, 1)
            return(0x00, 0x20)
        }
    }
}