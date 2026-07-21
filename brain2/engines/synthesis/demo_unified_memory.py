#!/usr/bin/env python3
"""
demo_unified_memory.py

Demonstrates the Unified Proposer reading Code Optimizations from the Brain's memory
(knowledge graph/facts) and training the CodeEngine on the fly.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engines.synthesis.unified_proposer import UnifiedProposer


# ── The Brain's Memories (extracted from textbooks / Graphify) ───────────────
memories = [
    {
        "type": "code_optimization",
        "lang": "cpp",
        "name": "HashJoin",
        "pattern_lambda": "lambda ops: len(ops) >= 2 and ops[0][0] == 'EnumProduct' and 'sum==target' in ops[1][1]",
        "render_fn_code": '''
def render_fn(ops):
    code = "bool solve(std::vector<int> input) {\\n"
    code += "    // ⚡ LEARNED FROM MEMORY: Hash Join (Two Sum)\\n"
    code += "    // Compiled O(N^2) Cartesian Product down to O(N) HashMap lookup\\n"
    code += "    std::unordered_map<int, int> numMap;\\n"
    code += "    for (int i = 0; i < input.size(); i++) {\\n"
    code += "        int complement = target - input[i];\\n"
    code += "        if (numMap.count(complement)) {\\n"
    code += "            return {numMap[complement], i};\\n"
    code += "        }\\n"
    code += "        numMap[input[i]] = i;\\n"
    code += "    }\\n"
    code += "    return {};\\n"
    code += "}\\n"
    return code.replace("bool solve", "std::vector<int> solve")
'''
    },
    {
        "type": "code_optimization",
        "lang": "cpp",
        "name": "SetLookup",
        "pattern_lambda": "lambda ops: ops and ops[0][0] == 'HasDuplicate'",
        "render_fn_code": '''
def render_fn(ops):
    code = "bool solve(std::vector<int> input) {\\n"
    code += "    // ⚡ LEARNED FROM MEMORY: Set Operations\\n"
    code += "    std::unordered_set<int> seen;\\n"
    code += "    for (int x : input) {\\n"
    code += "        if (seen.count(x)) return true;\\n"
    code += "        seen.insert(x);\\n"
    code += "    }\\n"
    code += "    return false;\\n"
    code += "}\\n"
    return code
'''
    }
]


def _demo():
    print("=" * 70)
    print("  UnifiedProposer — Memory-Based Code Engine Training")
    print("=" * 70)
    
    print("[1] Initializing UnifiedProposer with Brain Memories...")
    proposer = UnifiedProposer(knowledge_facts=memories)
    
    print("[2] Issuing Two Sum problem to the Proposer...")
    problem = {
        "type": "synthesize",
        "target_val": 9,
        "data": [
            ([2, 7, 11, 15], [0, 1]),
            ([3, 2, 4, 7], [1, 3])
        ]
    }
    
    result = proposer.solve(problem)
    
    if result:
        print(f"\n  [Policy Selected]: {result['policy']}")
        print(f"  [Discovered IR Tree]:\n    {result['tree']}")
        print(f"\n  [Rendered C++ Code]:\n{result['code_cpp']}")
    else:
        print("  FAILED to synthesize.")

    print("\n[3] Issuing Contains Duplicate problem to the Proposer...")
    problem2 = {
        "type": "synthesize",
        "data": [
            ([1, 2, 3, 1], True),
            ([1, 2, 3, 4], False)
        ]
    }
    result2 = proposer.solve(problem2)
    
    if result2:
        print(f"\n  [Policy Selected]: {result2['policy']}")
        print(f"  [Discovered IR Tree]:\n    {result2['tree']}")
        print(f"\n  [Rendered C++ Code]:\n{result2['code_cpp']}")


if __name__ == "__main__":
    _demo()
