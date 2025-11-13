#!/usr/bin/env python3
"""
EZKL Workflow Logger
Comprehensive logging system for EZKL zero-knowledge ML workflows
"""

import json
import os
import subprocess
import time
import datetime
from pathlib import Path

class EZKLWorkflowLogger:
    def __init__(self, model_name, base_dir="/root/ezkl_logs"):
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workflow_id = f"{model_name}_{self.timestamp}"

        # Create directories
        self.workflow_dir = self.base_dir / "workflows" / self.workflow_id
        self.workflow_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log data
        self.log_data = {
            "workflow_id": self.workflow_id,
            "model_name": model_name,
            "timestamp": self.timestamp,
            "start_time": time.time(),
            "steps": [],
            "total_memory_peak": 0,
            "total_time": 0,
            "success": False,
            "error_details": None
        }

        # System info
        self.get_system_info()

    def get_system_info(self):
        """Collect system information"""
        try:
            # Get system memory info
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()

            # Get CPU info
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()

            self.log_data["system_info"] = {
                "memory_total": self.extract_memory_value(meminfo, "MemTotal"),
                "memory_available": self.extract_memory_value(meminfo, "MemAvailable"),
                "cpu_count": len([line for line in cpuinfo.split('\n') if line.startswith('processor')]),
                "gpu_info": self.get_gpu_info()
            }
        except Exception as e:
            self.log_data["system_info"] = {"error": str(e)}

    def extract_memory_value(self, meminfo, key):
        """Extract memory value from /proc/meminfo"""
        for line in meminfo.split('\n'):
            if line.startswith(key):
                return line.split()[1] + " kB"
        return "unknown"

    def get_gpu_info(self):
        """Get GPU information"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except:
            pass
        return "GPU info unavailable"

    def run_timed_command(self, cmd, step_name, description=""):
        """Run a command with detailed timing and memory monitoring"""
        print(f"\n🔄 Running: {step_name}")
        print(f"Command: {' '.join(cmd)}")

        step_start = time.time()

        # Use GNU time for detailed resource monitoring
        time_cmd = ['/usr/bin/time', '-v'] + cmd

        try:
            result = subprocess.run(time_cmd,
                                  capture_output=True,
                                  text=True,
                                  cwd=str(self.workflow_dir))

            step_end = time.time()

            # Parse time output for memory statistics
            time_output = result.stderr
            memory_stats = self.parse_time_output(time_output)

            step_data = {
                "step": step_name,
                "description": description,
                "command": ' '.join(cmd),
                "start_time": step_start,
                "end_time": step_end,
                "duration": step_end - step_start,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "memory_stats": memory_stats,
                "success": result.returncode == 0
            }

            # Update peak memory
            if memory_stats.get("max_memory_kb", 0) > self.log_data["total_memory_peak"]:
                self.log_data["total_memory_peak"] = memory_stats.get("max_memory_kb", 0)

            self.log_data["steps"].append(step_data)

            # Print results
            if result.returncode == 0:
                print(f"✅ {step_name} completed successfully")
                print(f"⏱️  Duration: {step_data['duration']:.2f}s")
                print(f"🧠 Memory: {memory_stats.get('max_memory_mb', 0):.1f} MB")
            else:
                print(f"❌ {step_name} failed with code {result.returncode}")
                print(f"Error: {result.stderr}")

            return result.returncode == 0

        except Exception as e:
            print(f"❌ Error running {step_name}: {str(e)}")
            step_data = {
                "step": step_name,
                "error": str(e),
                "success": False,
                "start_time": step_start,
                "end_time": time.time()
            }
            self.log_data["steps"].append(step_data)
            return False

    def parse_time_output(self, time_output):
        """Parse GNU time output for memory and timing statistics"""
        stats = {}
        for line in time_output.split('\n'):
            line = line.strip()
            if 'Maximum resident set size (kbytes):' in line:
                kb = int(line.split(':')[-1].strip())
                stats['max_memory_kb'] = kb
                stats['max_memory_mb'] = kb / 1024
            elif 'User time (seconds):' in line:
                stats['user_time'] = float(line.split(':')[-1].strip())
            elif 'System time (seconds):' in line:
                stats['system_time'] = float(line.split(':')[-1].strip())
            elif 'Percent of CPU this job got:' in line:
                stats['cpu_percent'] = line.split(':')[-1].strip()
            elif 'Elapsed (wall clock) time' in line:
                stats['wall_time'] = line.split('):')[-1].strip()
            elif 'Minor (reclaiming a frame) page faults:' in line:
                stats['minor_page_faults'] = int(line.split(':')[-1].strip())
            elif 'Major (requiring I/O) page faults:' in line:
                stats['major_page_faults'] = int(line.split(':')[-1].strip())
        return stats

    def finalize_log(self, success=True, error_details=None):
        """Finalize and save the complete log"""
        self.log_data["end_time"] = time.time()
        self.log_data["total_time"] = self.log_data["end_time"] - self.log_data["start_time"]
        self.log_data["success"] = success
        self.log_data["error_details"] = error_details

        # Convert peak memory to MB
        self.log_data["total_memory_peak_mb"] = self.log_data["total_memory_peak"] / 1024

        # Save detailed JSON log
        log_file = self.workflow_dir / "workflow_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

        # Create human-readable summary
        self.create_summary_report()

        print(f"\n📊 Workflow completed!")
        print(f"Log saved to: {log_file}")
        print(f"Total time: {self.log_data['total_time']:.2f}s")
        print(f"Peak memory: {self.log_data['total_memory_peak_mb']:.1f} MB")

    def create_summary_report(self):
        """Create human-readable summary report"""
        report_file = self.workflow_dir / "summary_report.md"

        with open(report_file, 'w') as f:
            f.write(f"# EZKL Workflow Report: {self.workflow_id}\n\n")
            f.write(f"**Model**: {self.model_name}\n")
            f.write(f"**Timestamp**: {self.timestamp}\n")
            f.write(f"**Total Duration**: {self.log_data['total_time']:.2f} seconds\n")
            f.write(f"**Peak Memory**: {self.log_data['total_memory_peak_mb']:.1f} MB\n")
            f.write(f"**Success**: {'✅' if self.log_data['success'] else '❌'}\n\n")

            f.write("## System Information\n")
            sys_info = self.log_data.get("system_info", {})
            f.write(f"- **Total Memory**: {sys_info.get('memory_total', 'unknown')}\n")
            f.write(f"- **Available Memory**: {sys_info.get('memory_available', 'unknown')}\n")
            f.write(f"- **CPU Cores**: {sys_info.get('cpu_count', 'unknown')}\n")
            f.write(f"- **GPU Info**: {sys_info.get('gpu_info', 'unknown')}\n\n")

            f.write("## Step-by-Step Results\n\n")
            for i, step in enumerate(self.log_data["steps"], 1):
                status = "✅" if step.get("success", False) else "❌"
                f.write(f"### {i}. {step['step']} {status}\n")
                f.write(f"- **Duration**: {step.get('duration', 0):.2f}s\n")
                if 'memory_stats' in step:
                    mem_mb = step['memory_stats'].get('max_memory_mb', 0)
                    f.write(f"- **Memory**: {mem_mb:.1f} MB\n")
                    cpu_pct = step['memory_stats'].get('cpu_percent', 'unknown')
                    f.write(f"- **CPU Usage**: {cpu_pct}\n")
                f.write(f"- **Command**: `{step.get('command', 'unknown')}`\n\n")

if __name__ == "__main__":
    # Example usage will be added when we run the workflow
    pass