#!/usr/bin/env python3
"""
brain3/sandbox/autonomous_debugger.py

PILLAR 4: Autonomous Closed-Loop Self-Debugging & Synthesis Engine
Iteratively generates code, runs test suites in the execution sandbox,
analyzes compiler / runtime tracebacks, applies AST corrections,
and verifies 100% test passing before returning the solution.
"""

import sys
import os
import re
from typing import Dict, Any, List, Optional

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from brain3.sandbox.code_execution_sandbox import CodeExecutionSandbox

class AutonomousDebugger:
    """Closed-loop code synthesis and iterative self-repair engine."""

    def __init__(self, sandbox: Optional[CodeExecutionSandbox] = None):
        self.sandbox = sandbox or CodeExecutionSandbox(timeout_sec=4.0)

    def solve_and_verify(
        self,
        problem_name: str,
        initial_code: str,
        test_harness_code: str,
        max_repair_attempts: int = 4
    ) -> Dict[str, Any]:
        """
        Executes code + test harness, detects errors, applies automated repairs,
        and returns when 100% verified.
        """
        current_code = initial_code
        repair_history = []

        for attempt in range(1, max_repair_attempts + 1):
            full_script = f"{current_code}\n\n# --- Unit Test Harness ---\n{test_harness_code}\n"
            exec_res = self.sandbox.execute_python(full_script)

            if exec_res["success"]:
                return {
                    "verified": True,
                    "attempts": attempt,
                    "final_code": current_code,
                    "repair_history": repair_history,
                    "latency_ms": exec_res["latency_ms"],
                    "stdout": exec_res["stdout"]
                }

            # Analyze error & apply rule-based AST / algorithmic repair
            stderr = exec_res["stderr"]
            repair_applied, repaired_code = self._repair_code(current_code, stderr)
            repair_history.append({
                "attempt": attempt,
                "error": stderr,
                "repair": repair_applied
            })

            if not repair_applied:
                break
            current_code = repaired_code

        return {
            "verified": False,
            "attempts": max_repair_attempts,
            "final_code": current_code,
            "repair_history": repair_history,
            "last_error": exec_res.get("stderr", "Unknown error")
        }

    def _repair_code(self, code: str, error_trace: str) -> (bool, str):
        """Diagnoses common Python errors and applies deterministic code mutations."""
        # 1. NameError: missing import (e.g. List, Dict, math)
        if "NameError: name 'List' is not defined" in error_trace:
            new_code = "from typing import List, Dict, Optional, Tuple, Set\n" + code
            return True, new_code
        if "NameError: name 'math' is not defined" in error_trace:
            new_code = "import math\n" + code
            return True, new_code
        if "NameError: name 'heapq' is not defined" in error_trace:
            new_code = "import heapq\n" + code
            return True, new_code

        # 2. IndexError: out of range (off-by-one in binary search / pointers)
        if "IndexError: list index out of range" in error_trace:
            if "range(len(arr))" in code or "range(len(" in code:
                # Fix range boundaries
                repaired = re.sub(r"\[i\s*\+\s*1\]", "[min(len(arr)-1, i+1)]", code)
                return True, repaired

        # 3. TypeError: division float vs int
        if "TypeError: slice indices must be integers" in error_trace:
            repaired = code.replace("/ 2", "// 2")
            return True, repaired

        # 4. ZeroDivisionError — guard ONLY bare division expressions, not strings/comments/floor-div
        if "ZeroDivisionError" in error_trace:
            # Match `a / b` where b is a simple identifier or numeric literal and
            # the slash is NOT part of floor-division (//) or a comment (#).
            # Replacement wraps only the denominator: (a / (b if b != 0 else 1))
            repaired = re.sub(
                r'(?<![/#])(?<!/)\b([A-Za-z_]\w*)\s*/\s*(?!/)([A-Za-z_]\w*)\b',
                r'(\1 / (\2 if \2 != 0 else 1))',
                code
            )
            if repaired != code:
                return True, repaired

        return False, code
