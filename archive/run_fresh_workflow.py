#!/usr/bin/env python3
"""
Run a complete fresh EZKL workflow with comprehensive logging
"""

import sys
import os
sys.path.append("/root")
from ezkl_workflow_logger import EZKLWorkflowLogger

def run_complete_workflow():
    # Initialize logger
    logger = EZKLWorkflowLogger("1l_average_fresh")
    
    try:
        print(f"🚀 Starting fresh EZKL workflow: {logger.workflow_id}")
        print(f"📁 Working directory: {logger.workflow_dir}")
        
        # Step 1: Generate settings
        success = logger.run_timed_command(
            ["/root/.ezkl/ezkl", "gen-settings", "-M", "/root/ezkl_logs/models/1l_average_fresh/network.onnx", "--settings-path", "settings.json"],
            "gen_settings",
            "Generate circuit settings from ONNX model"
        )
        if not success:
            logger.finalize_log(False, "Failed at gen-settings step")
            return
        
        # Step 2: Calibrate settings  
        success = logger.run_timed_command(
            ["/root/.ezkl/ezkl", "calibrate-settings", "-M", "/root/ezkl_logs/models/1l_average_fresh/network.onnx", 
             "-D", "/root/ezkl_logs/models/1l_average_fresh/input.json", "--settings-path", "settings.json"],
            "calibrate_settings", 
            "Calibrate settings with input data for optimal accuracy"
        )
        if not success:
            logger.finalize_log(False, "Failed at calibrate-settings step")
            return
            
        # Step 3: Compile circuit
        success = logger.run_timed_command(
            ["/root/.ezkl/ezkl", "compile-circuit", "-M", "/root/ezkl_logs/models/1l_average_fresh/network.onnx",
             "--compiled-circuit", "network.ezkl", "--settings-path", "settings.json"],
            "compile_circuit",
            "Compile ONNX model to EZKL circuit"
        )
        if not success:
            logger.finalize_log(False, "Failed at compile-circuit step")
            return
            
        # Step 4: Generate SRS
        success = logger.run_timed_command(
            ["/root/.ezkl/ezkl", "gen-srs", "--srs-path", "kzg.srs", "--logrows", "15"],
            "gen_srs",
            "Generate structured reference string for proving system"
        )
        if not success:
            logger.finalize_log(False, "Failed at gen-srs step")
            return
            
        # Step 5: Setup proving/verification keys
        success = logger.run_timed_command(
            ["/root/.ezkl/ezkl", "setup", "--compiled-circuit", "network.ezkl", 
             "--vk-path", "vk.key", "--pk-path", "pk.key", "--srs-path", "kzg.srs"],
            "setup_keys",
            "Generate proving and verification keys"
        )
        if not success:
            logger.finalize_log(False, "Failed at setup step")
            return
            
        # Step 6: Generate witness (using Python API since CLI has issues)
        print("\n🔄 Running: gen_witness")
        print("Using Python API for witness generation")
        step_start = time.time()
        
        try:
            import ezkl
            result = ezkl.gen_witness("/root/ezkl_logs/models/1l_average_fresh/input.json", "network.ezkl", "witness.json")
            step_end = time.time()
            
            step_data = {
                "step": "gen_witness",
                "description": "Generate witness data using Python API",
                "command": "ezkl.gen_witness(input.json, network.ezkl, witness.json)",
                "start_time": step_start,
                "end_time": step_end,
                "duration": step_end - step_start,
                "return_code": 0,
                "success": True,
                "api_result": str(result)
            }
            logger.log_data["steps"].append(step_data)
            print(f"✅ gen_witness completed successfully")
            print(f"⏱️  Duration: {step_data['duration']:.2f}s")
            
        except Exception as e:
            print(f"❌ gen_witness failed: {str(e)}")
            logger.finalize_log(False, f"Failed at gen_witness step: {str(e)}")
            return
            
        # Step 7: Mock prove (validation)
        success = logger.run_timed_command(
            ["/root/.ezkl/ezkl", "mock", "-M", "network.ezkl", "-W", "witness.json"],
            "mock_prove",
            "Run mock prover to validate circuit setup"
        )
        if not success:
            logger.finalize_log(False, "Failed at mock prove step")
            return
            
        # Step 8: Generate actual proof
        success = logger.run_timed_command(
            ["/root/.ezkl/ezkl", "prove", "--compiled-circuit", "network.ezkl", 
             "--pk-path", "pk.key", "--proof-path", "proof.json", 
             "--srs-path", "kzg.srs", "--witness", "witness.json"],
            "generate_proof",
            "Generate zero-knowledge proof"
        )
        if not success:
            logger.finalize_log(False, "Failed at proof generation step")
            return
            
        # Step 9: Verify proof
        success = logger.run_timed_command(
            ["/root/.ezkl/ezkl", "verify", "--proof-path", "proof.json", 
             "--settings-path", "settings.json", "--vk-path", "vk.key", "--srs-path", "kzg.srs"],
            "verify_proof",
            "Verify the generated zero-knowledge proof"
        )
        if not success:
            logger.finalize_log(False, "Failed at verification step")
            return
            
        # Success!
        logger.finalize_log(True)
        
        print("\n🎉 Complete EZKL workflow finished successfully!")
        print(f"📊 Results saved to: {logger.workflow_dir}")
        
    except Exception as e:
        print(f"\n❌ Workflow failed with error: {str(e)}")
        logger.finalize_log(False, str(e))

if __name__ == "__main__":
    import time
    run_complete_workflow()
