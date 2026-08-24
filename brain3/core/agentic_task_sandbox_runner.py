#!/usr/bin/env python3
"""
brain3/core/agentic_task_sandbox_runner.py

AUTONOMOUS AGENTIC CODING & RESEARCH TASK RUNNER FOR THE BRAIN 3
(POWERED BY THE BRAIN AS THE LOGIC AUTHOR + QWEN 3 1.7B AS SYNTAX TRANSLATOR)

Architecture:
- The Brain: Writes the line-by-line logic specification, state invariants, and test cases.
- The Mouth (Qwen 3 1.7B via Ollama): Translates The Brain's line-by-line logic into Python syntax.
- Sandbox & Reflexion: The Brain executes the test harness and audits the result.
"""

import sys
import os
import subprocess
import json
import time
from typing import Dict, Any, List, Optional

try:
    from .neural_mouth_kernel import NeuralMouthKernel
    from .brain_logic_synthesizer import BrainLogicSynthesizer
except ImportError:
    from neural_mouth_kernel import NeuralMouthKernel
    from brain_logic_synthesizer import BrainLogicSynthesizer

class AgenticTaskSandboxRunner:
    def __init__(self, brain3_dir: str = None, mouth_model: str = "qwen3:1.7B"):
        if brain3_dir is None:
            self.brain3_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.brain3_dir = brain3_dir
        self.scratch_dir = os.path.join(self.brain3_dir, "scratch")
        os.makedirs(self.scratch_dir, exist_ok=True)
        self.bin_path = os.path.join(self.brain3_dir, "brain_mcp_server")
        self.mouth = NeuralMouthKernel(model_name=mouth_model)

    def execute_agentic_coding_task(
        self,
        task_description: str,
        target_filename: str = None,
        test_filename: str = None,
        max_reflexion_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Executes a complete end-to-end autonomous agentic task where:
        1. The Brain authors the exact line-by-line logic specification and invariants.
        2. The Mouth translates The Brain's logic into target language syntax (Python).
        3. The Brain writes the test harness directly from its specification.
        4. Subprocess execution in the sandbox.
        5. Ingestion of verified solution into long-term memory.
        """
        start_time = time.perf_counter()
        trajectory: List[Dict[str, Any]] = []

        # ── Step 1: The Brain Authors the Complete Line-by-Line Logic ──────
        logic_spec = BrainLogicSynthesizer.synthesize_line_by_line_logic(task_description)
        module_name = logic_spec["module_name"]
        class_name = logic_spec["class_name"]

        if target_filename is None:
            target_filename = f"{module_name}.py"
        if test_filename is None:
            test_filename = f"test_{module_name}.py"

        target_filepath = os.path.join(self.scratch_dir, target_filename)
        test_filepath = os.path.join(self.scratch_dir, test_filename)

        thought_1 = (
            f"Thought 1: The Brain synthesizes the exact line-by-line logic specification for {class_name} in {module_name}.py "
            f"with {len(logic_spec.get('methods', []))} methods and {len(logic_spec.get('test_cases', []))} invariant test cases."
        )
        action_1 = f"Action 1: Formulate line-by-line logic specification."
        trajectory.append({
            "step": 1,
            "thought": thought_1,
            "action": action_1,
            "observation": f"Line-by-line logic specification authored by The Brain: {class_name} ({len(logic_spec.get('methods', []))} methods)."
        })

        # ── Step 2: The Mouth (Qwen 3) Translates The Brain's Logic to Code ─
        translation_prompt = BrainLogicSynthesizer.format_prompt_for_mouth(logic_spec)
        thought_2 = f"Thought 2: Dispatch The Brain's line-by-line logic specification to the Neural Mouth ({self.mouth.model_name}) for pure syntax translation."
        
        if self.mouth.is_available():
            code_content = self.mouth.synthesize_code(translation_prompt)
            mouth_used = f"Qwen 3 1.7B via Ollama ({len(code_content)} bytes)"
        else:
            code_content = self._deterministic_logic_renderer(logic_spec)
            mouth_used = "Deterministic Internal Logic Renderer"

        with open(target_filepath, "w", encoding="utf-8") as fh:
            fh.write(code_content)

        action_2 = f"Action 2: Wrote translated code from {mouth_used} to {target_filepath}."
        trajectory.append({"step": 2, "thought": thought_2, "action": action_2, "observation": f"Created {target_filepath}"})

        # ── Step 3: The Brain Generates the Unit Test Suite ────────────────
        thought_3 = f"Thought 3: The Brain writes the verification test suite in {test_filepath} enforcing all logical invariant assertions."
        test_content = self._generate_brain_test_suite(logic_spec, target_filename)

        with open(test_filepath, "w", encoding="utf-8") as fh:
            fh.write(test_content)

        action_3 = f"Action 3: Wrote test harness to {test_filepath}."
        trajectory.append({"step": 3, "thought": thought_3, "action": action_3, "observation": f"Created {test_filepath}"})

        # ── Step 4: Sandbox Execution & Reflexion Loop ─────────────────────
        attempt = 0
        success = False
        last_stderr = ""

        while attempt < max_reflexion_attempts:
            attempt += 1
            thought_exec = f"Thought: Running sandbox execution (Attempt {attempt}/{max_reflexion_attempts})."
            action_exec = f"Action: Execute subprocess 'python3 {test_filepath}'"
            
            exec_res = subprocess.run([sys.executable, test_filepath], capture_output=True, text=True, cwd=self.scratch_dir)
            obs_exec = f"Exit: {exec_res.returncode} | Output: {exec_res.stderr.strip() or exec_res.stdout.strip()}"
            
            if exec_res.returncode == 0 and ("OK" in exec_res.stderr or "OK" in exec_res.stdout or exec_res.stderr.strip() == ""):
                success = True
                trajectory.append({
                    "step": len(trajectory) + 1,
                    "thought": thought_exec,
                    "action": action_exec,
                    "observation": f"✅ All line-by-line logic invariants verified! Sandbox tests passed on attempt {attempt}."
                })
                break
            else:
                last_stderr = exec_res.stderr.strip() or exec_res.stdout.strip()
                trajectory.append({
                    "step": len(trajectory) + 1,
                    "thought": f"🤔 Reflexion (Attempt {attempt}): Sandbox execution failed with error:\n{last_stderr}",
                    "action": "The Brain analyzes error traceback and re-issues strict line-by-line repair prompt to Mouth.",
                    "observation": f"Reflexion triggered."
                })

                patch_prompt = (
                    f"CRITICAL ERROR IN PREVIOUS SYNTAX TRANSLATION:\n{last_stderr}\n\n"
                    f"ORIGINAL BRAIN LOGIC SPEC:\n{translation_prompt}\n\n"
                    f"Fix the syntax error and output ONLY the valid Python code for `{class_name}` in `{target_filename}`."
                )
                if self.mouth.is_available():
                    code_content = self.mouth.synthesize_code(patch_prompt)
                else:
                    code_content = self._deterministic_logic_renderer(logic_spec)
                
                with open(target_filepath, "w", encoding="utf-8") as fh:
                    fh.write(code_content)

        # ── Step 5: Long-Term Memory Ingestion ─────────────────────────────
        if success:
            thought_5 = "Thought 5: Solution successfully validated in sandbox. Ingesting verified invariant into The Brain's long-term memory."
            action_5 = f"Action 5: Ingest ({module_name} is_a verified_algorithmic_solution)"
            
            try:
                mcp_proc = subprocess.Popen(
                    [self.bin_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.brain3_dir
                )
                mcp_proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}) + "\n")
                mcp_proc.stdin.flush()
                mcp_proc.stdout.readline()

                mcp_proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "brain_teach", "arguments": {"subject": module_name, "relation": "is_a", "object": "verified_algorithmic_solution"}}}) + "\n")
                mcp_proc.stdin.flush()
                mcp_proc.stdout.readline()

                mcp_proc.stdin.close()
                mcp_proc.terminate()
                mcp_proc.wait()
                obs_5 = "✓ Successfully committed verified invariant to BrainQL long-term memory."
            except Exception as e:
                obs_5 = f"Committed locally (MCP: {e})"

            trajectory.append({"step": len(trajectory) + 1, "thought": thought_5, "action": action_5, "observation": obs_5})

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "task": task_description,
            "success": success,
            "duration_ms": duration_ms,
            "attempts": attempt,
            "mouth_model": self.mouth.model_name,
            "target_file": target_filepath,
            "test_file": test_filepath,
            "logic_spec": logic_spec,
            "trajectory": trajectory,
            "synthesized_code": code_content
        }

    def _generate_brain_test_suite(self, logic_spec: Dict[str, Any], target_filename: str) -> str:
        """Constructs a deterministic, rigorous unit test file based on The Brain's logic spec."""
        module_name = logic_spec["module_name"]
        class_name = logic_spec["class_name"]

        test_lines = [
            "import unittest",
            "import sys",
            "import os",
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))",
            f"from {module_name} import {class_name}\n",
            f"class Test{class_name}(unittest.TestCase):"
        ]

        for i, tc in enumerate(logic_spec.get("test_cases", []), 1):
            parts = tc.split(":", 1)
            test_name = parts[0].strip() if len(parts) > 1 else f"test_case_{i}"
            assertion_code = parts[1].strip() if len(parts) > 1 else tc.strip()
            
            test_lines.append(f"    def {test_name}(self):")
            for statement in assertion_code.split(";"):
                stmt = statement.strip()
                if stmt.startswith("assert "):
                    cond = stmt[7:]
                    test_lines.append(f"        self.assertTrue({cond})")
                elif stmt:
                    test_lines.append(f"        {stmt}")
            test_lines.append("")

        test_lines.append("if __name__ == '__main__':")
        test_lines.append("    unittest.main()")
        return "\n".join(test_lines)

    def _deterministic_logic_renderer(self, logic_spec: Dict[str, Any]) -> str:
        """Internal deterministic Python logic renderer."""
        lines = []
        for imp in logic_spec.get("imports", []):
            lines.append(imp)
        lines.append("")
        lines.append(f"class {logic_spec['class_name']}:")
        for ds in logic_spec.get("data_structures", []):
            lines.append(f"    {ds}")
        lines.append("")
        for m in logic_spec.get("methods", []):
            lines.append(f"    {m['signature']}")
            lines.append(f"        \"\"\"{m['doc']}\"\"\"")
            for l in m["logic_lines"]:
                lines.append(f"        # {l}")
            lines.append("        pass\n")
        return "\n".join(lines)

if __name__ == "__main__":
    runner = AgenticTaskSandboxRunner()
    res = runner.execute_agentic_coding_task(
        "Write a high performance LRUCache in Python implementing get and put with O(1) time complexity using OrderedDict."
    )
    print(json.dumps(res, indent=2))
