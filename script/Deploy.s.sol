// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "forge-std/Script.sol";
import "../src/GANVerifier.sol";
import "../src/ClassifierVerifier.sol";

contract DeployScript is Script {
    function run() external {
        // Get private key from environment variable or use Anvil's default account
        uint256 deployerPrivateKey;
        
        try vm.envUint("PRIVATE_KEY") returns (uint256 pk) {
            deployerPrivateKey = pk;
        } catch {
            // Default Anvil account #0 private key
            deployerPrivateKey = 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;
        }

        vm.startBroadcast(deployerPrivateKey);

        console.log("Deploying GAN Verifier...");
        GANVerifier ganVerifier = new GANVerifier();
        console.log("GAN Verifier deployed at:", address(ganVerifier));

        console.log("\nDeploying Classifier Verifier...");
        console.log("WARNING: This is a large contract (1.3MB). Deployment may take time.");
        
        // Deploy classifier verifier (1.3MB contract)
        // Note: This will only work on networks without strict contract size limits (e.g., Anvil, some L2s)
        ClassifierVerifier classifierVerifier = new ClassifierVerifier();
        console.log("Classifier Verifier deployed at:", address(classifierVerifier));

        vm.stopBroadcast();

        // Save deployment addresses to a file
        string memory deploymentInfo = string(abi.encodePacked(
            "GAN_VERIFIER=", vm.toString(address(ganVerifier)), "\n",
            "CLASSIFIER_VERIFIER=", vm.toString(address(classifierVerifier)), "\n"
        ));
        
        vm.writeFile("deployments.txt", deploymentInfo);
        console.log("\nDeployment addresses saved to deployments.txt");
    }
}
