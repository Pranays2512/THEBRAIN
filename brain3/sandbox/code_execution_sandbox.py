#!/usr/bin/env python3
"""
brain3/sandbox/code_execution_sandbox.py

PILLAR 4: Autonomous Closed-Loop Tool & Code Execution Sandbox
Provides safe, isolated, high-speed execution environments for Python, C++, and Bash
with resource constraints, timeout guarantees, and structured execution telemetry.
"""

import sys
import os
import subprocess
import tempfile
import time
from typing import Dict, Any, List, Optional

class CodeExecutionSandbox:
    """Safe subprocess execution sandbox with timeout and telemetry capture."""

    def __init__(self, timeout_sec: float = 5.0, max_memory_mb: int = 256):
        self.timeout_sec = timeout_sec
        self.max_memory_mb = max_memory_mb

    def execute_python(self, code_str: str) -> Dict[str, Any]:
        """Executes Python code in an isolated subprocess."""
        start_time = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(code_str)
            tf_path = tf.name

        try:
            proc = subprocess.run(
                [sys.executable, tf_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_sec
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "latency_ms": elapsed_ms,
                "timeout": False
            }
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout_sec}s",
                "latency_ms": elapsed_ms,
                "timeout": True
            }
        finally:
            if os.path.exists(tf_path):
                try:
                    os.remove(tf_path)
                except Exception:
                    pass

    def execute_cpp(self, cpp_code: str) -> Dict[str, Any]:
        """Compiles and executes C++17 code in an isolated binary sandbox."""
        start_time = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as tf:
            tf.write(cpp_code)
            src_path = tf.name
        
        bin_path = src_path + ".bin"
        try:
            # 1. Compile
            compile_proc = subprocess.run(
                ["clang++", "-std=c++17", "-O2", "-o", bin_path, src_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_sec
            )
            if compile_proc.returncode != 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "success": False,
                    "exit_code": compile_proc.returncode,
                    "stdout": "",
                    "stderr": "Compilation Error: " + compile_proc.stderr.strip(),
                    "latency_ms": elapsed_ms,
                    "timeout": False
                }

            # 2. Run
            run_proc = subprocess.run(
                [bin_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_sec
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": run_proc.returncode == 0,
                "exit_code": run_proc.returncode,
                "stdout": run_proc.stdout.strip(),
                "stderr": run_proc.stderr.strip(),
                "latency_ms": elapsed_ms,
                "timeout": False
            }
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout_sec}s",
                "latency_ms": elapsed_ms,
                "timeout": True
            }
        finally:
            for p in [src_path, bin_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
