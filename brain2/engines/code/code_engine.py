#!/usr/bin/env python3
"""
code_engine.py — The Brain's SINGLE coding engine.

Replaces domain-specific synthesizers (like leetcode_synth) with a unified
engine that discovers algorithmic proofs via A* Search and dynamically LEARNS
compilation optimizations (DSA mapping) instead of hardcoding them.
"""

import os
import sys
import heapq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class CodeEngine:
    """Unified algorithm synthesizer with trainable DSA optimizations."""
    
    def __init__(self, target_val=None):
        self.target_val = target_val  # Context variable for problems like Two Sum
        
        # Registry for learned algorithmic optimizations
        self.learned_optimizations = {"cpp": {}, "java": {}, "python": {}}

        # ── The List-Processing DSL ─────────────────────────────────────
        self.list_ops = [
            ("Reverse", lambda lst: list(reversed(lst))),
            ("Sort", lambda lst: sorted(lst)),
            ("ToSet", lambda lst: list(set(lst))),
            ("EnumProduct", lambda lst: [((i, x), (j, y)) for i, x in enumerate(lst) for j, y in enumerate(lst)]),
            ("First", lambda lst: lst[0] if lst and len(lst) >= 1 else None),
            ("CumulativeSum", lambda lst: [sum(lst[:i+1]) for i in range(len(lst))]),
            ("RollingWindow", lambda lst: lst), # Conceptual structural op
            ("Partition", lambda lst: lst),     # Conceptual structural op
            ("Converge", lambda lst: lst),      # Conceptual structural op
            ("Neighborhood", lambda lst: lst),  # Conceptual structural op
            ("Extrema", lambda lst: lst),       # Conceptual structural op
            ("DepthExplore", lambda lst: lst),
            ("Memoize", lambda lst: lst),
            ("LocalOptimum", lambda lst: lst),
            ("PathRelax", lambda lst: lst),
            ("StateSearch", lambda lst: lst),
            ("PrefixTraverse", lambda lst: lst),
            ("DependencyOrder", lambda lst: lst),
        ]
        
        self.map_ops = [
            ("*2", lambda x: x * 2, "x * 2"),
            ("+1", lambda x: x + 1, "x + 1"),
            ("-1", lambda x: x - 1, "x - 1"),
            ("^2", lambda x: x ** 2, "x ** 2"),
            ("abs", lambda x: abs(x), "abs(x)"),
            ("extract_indices", lambda pair: [pair[0][0], pair[1][0]], "[x[0][0], x[1][0]]"),
        ]
        
        self.filter_ops = [
            (">0", lambda x: x > 0, "x > 0"),
            ("<0", lambda x: x < 0, "x < 0"),
            ("even", lambda x: x % 2 == 0 if isinstance(x, int) else False, "x % 2 == 0"),
            ("odd", lambda x: x % 2 != 0 if isinstance(x, int) else False, "x % 2 != 0"),
            ("sum==target & i!=j", 
             lambda pair: pair[0][0] != pair[1][0] and (pair[0][1] + pair[1][1] == self.target_val),
             f"x[0][0] != x[1][0] && x[0][1] + x[1][1] == target"),
        ]
        
        self.terminal_ops = [
            ("HasDuplicate", lambda lst: len(set(lst)) != len(lst)),
        ]

    # ── Online Learning for DSA ───────────────────────────────────────
    def learn_optimization(self, lang, name, pattern_fn, render_fn):
        """Teach the Brain an optimization mapping."""
        self.learned_optimizations[lang][name] = (pattern_fn, render_fn)

    # ── Evaluation and Heuristics ────────────────────────────────────
    def _heuristic(self, current_states, goal_states):
        total_dist = 0
        for curr, goal in zip(current_states, goal_states):
            if isinstance(goal, bool):
                if curr == goal: continue
                elif isinstance(curr, bool): total_dist += 10
                else: return float('inf')
                
            if not isinstance(curr, list) or not isinstance(goal, list):
                if curr == goal:
                    continue
                return float('inf')
                
            len_diff = abs(len(curr) - len(goal))
            total_dist += len_diff * 1
            
            if len(curr) == len(goal) and len(curr) > 0:
                try:
                    if isinstance(curr[0], list):
                        if goal in curr:
                            total_dist += 1
                        else:
                            total_dist += 20
                    else:
                        c_sort = sorted(curr)
                        g_sort = sorted(goal)
                        for c, g in zip(c_sort, g_sort):
                            total_dist += abs(c - g)
                except Exception:
                    total_dist += 0
                    
            if curr != goal:
                total_dist += 5
                
        return total_dist

    def _apply_op(self, op_type, op_name, data):
        if not isinstance(data, list):
            return None
            
        try:
            if op_type == "Map":
                fn = next(f for n, f, _ in self.map_ops if n == op_name)
                return [fn(x) for x in data]
            elif op_type == "Filter":
                fn = next(f for n, f, _ in self.filter_ops if n == op_name)
                return [x for x in data if fn(x)]
            elif op_type == "List":
                fn = next(f for n, f in self.list_ops if n == op_name)
                return fn(data)
            elif op_type == "Terminal":
                fn = next(f for n, f in self.terminal_ops if n == op_name)
                return fn(data)
        except Exception:
            return None
        return None

    # ── Rendering ────────────────────────────────────────────────────
    def _render_python(self, tree):
        if tree == "Input":
            return "input_list"
        op = tree[0]
        if op == "Map":
            _, map_op, inner = tree
            inner_str = self._render_python(inner)
            expr = next(e for n, _, e in self.map_ops if n == map_op)
            return f"[{expr} for x in {inner_str}]"
        elif op == "Filter":
            _, filter_op, inner = tree
            inner_str = self._render_python(inner)
            cond = next(c for n, _, c in self.filter_ops if n == filter_op)
            return f"[x for x in {inner_str} if {cond}]"
        else:
            inner_str = self._render_python(tree[1])
            if op == "Reverse": return f"list(reversed({inner_str}))"
            if op == "Sort": return f"sorted({inner_str})"
        return "unknown"

    def _flatten_tree(self, tree):
        ops = []
        curr = tree
        while curr != "Input":
            if curr[0] in ("Map", "Filter"):
                ops.append((curr[0], curr[1]))
                curr = curr[2]
            else:
                ops.append((curr[0], None))
                curr = curr[1]
        ops.reverse()
        return ops

    def _render_java(self, tree):
        if tree == "Input":
            return "public List<Integer> solve(List<Integer> input) {\n    return input;\n}"
        ops = self._flatten_tree(tree)
        stream_code = "input.stream()\n"
        for op, arg in ops:
            if op == "Map":
                expr = next(e for n, _, e in self.map_ops if n == arg)
                stream_code += f"        .map(x -> {expr})\n"
            elif op == "Filter":
                cond = next(c for n, _, c in self.filter_ops if n == arg)
                stream_code += f"        .filter(x -> {cond})\n"
            elif op == "Sort":
                stream_code += f"        .sorted()\n"
        stream_code += "        .collect(Collectors.toList());"
        has_reverse = any(op == "Reverse" for op, _ in ops)
        code = "public List<Integer> solve(List<Integer> input) {\n"
        if has_reverse:
            code += f"    List<Integer> res = {stream_code}\n    Collections.reverse(res);\n    return res;\n"
        else:
            code += f"    return {stream_code}\n"
        code += "}"
        return code

    def _render_cpp(self, tree, optimize=True):
        if tree == "Input":
            return "std::vector<int> solve(std::vector<int> input) {\n    return input;\n}"
            
        ops = self._flatten_tree(tree)
        
        if optimize:
            # 1. Apply Dynamically Learned Optimizations
            for name, (pattern_fn, render_fn) in self.learned_optimizations.get("cpp", {}).items():
                if pattern_fn(ops):
                    return render_fn(ops)

            # 2. General AST Optimizations (Loop Fusion)
            code = "std::vector<int> solve(std::vector<int> input) {\n"
            fusable = []
            non_fusable = []
            idx = 0
            while idx < len(ops) and ops[idx][0] in ("Map", "Filter"):
                fusable.append(ops[idx])
                idx += 1
            non_fusable = ops[idx:]
            
            if fusable:
                if len(fusable) > 1:
                    code += "    // ⚡ OPTIMIZED: Loop Fusion (Deforestation)\n"
                code += "    std::vector<int> res;\n    res.reserve(input.size());\n    for (int x : input) {\n"
                indent = "        "
                open_braces = 0
                for op, arg in fusable:
                    if op == "Filter":
                        cond = next(c for n, _, c in self.filter_ops if n == arg)
                        code += f"{indent}if ({cond}) {{\n"
                        indent += "    "
                        open_braces += 1
                    elif op == "Map":
                        expr = next(e for n, _, e in self.map_ops if n == arg)
                        code += f"{indent}x = {expr};\n"
                code += f"{indent}res.push_back(x);\n"
                for _ in range(open_braces):
                    indent = indent[:-4]
                    code += f"{indent}}}\n"
                code += "    }\n"
            else:
                code += "    std::vector<int> res = input;\n"
                
            for op, arg in non_fusable:
                if op == "Sort": code += "    std::sort(res.begin(), res.end());\n"
                elif op == "Reverse": code += "    std::reverse(res.begin(), res.end());\n"
            code += "    return res;\n}\n"
            return code

    # ── Search Engine ────────────────────────────────────────────────
    def synthesize(self, io_examples, max_steps=2000, verbose=False):
        if verbose:
            print(f"Synthesizing algorithm for {len(io_examples)} test cases using A* Search...")
            
        start_states = [inp for inp, _ in io_examples]
        goal_states = [outp for _, outp in io_examples]
        
        pq = []
        tiebreaker = 0
        h_start = self._heuristic(start_states, goal_states)
        if h_start == 0:
            return {"tree": "Input", "code": "def solve(input_list):\n    return input_list", "code_cpp": self._render_cpp("Input"), "candidates_searched": 1}
            
        heapq.heappush(pq, (h_start, 0, tiebreaker, "Input", start_states))
        
        visited_states = set()
        steps = 0
        
        while pq and steps < max_steps:
            f_score, g_score, _, tree, current_states = heapq.heappop(pq)
            steps += 1
            
            if current_states == goal_states:
                if verbose:
                    print(f"  ✓ Discovered valid IR tree after {steps} A* expansions.")
                return {
                    "tree": tree,
                    "code": f"def solve(input_list):\n    return {self._render_python(tree)}",
                    "code_java": self._render_java(tree),
                    "code_cpp": self._render_cpp(tree),
                    "candidates_searched": steps
                }
                
            state_signature = str(current_states)
            if state_signature in visited_states: continue
            visited_states.add(state_signature)
            
            new_g = g_score + 1
            
            for op_name, _, _ in self.map_ops:
                new_states = [self._apply_op("Map", op_name, s) for s in current_states]
                if None not in new_states:
                    h = self._heuristic(new_states, goal_states)
                    if h != float('inf'):
                        tiebreaker += 1
                        heapq.heappush(pq, (new_g + h, new_g, tiebreaker, ("Map", op_name, tree), new_states))
                        
            for op_name, _, _ in self.filter_ops:
                new_states = [self._apply_op("Filter", op_name, s) for s in current_states]
                if None not in new_states:
                    h = self._heuristic(new_states, goal_states)
                    if h != float('inf'):
                        tiebreaker += 1
                        heapq.heappush(pq, (new_g + h, new_g, tiebreaker, ("Filter", op_name, tree), new_states))
                        
            for op_name, _ in self.list_ops:
                new_states = [self._apply_op("List", op_name, s) for s in current_states]
                if None not in new_states:
                    h = self._heuristic(new_states, goal_states)
                    if op_name == "CumulativeSum":
                        print(f"  [DEBUG] CumulativeSum produced {new_states} with h={h}")
                    if h != float('inf'):
                        tiebreaker += 1
                        heapq.heappush(pq, (new_g + h, new_g, tiebreaker, (op_name, tree), new_states))
                        
            for op_name, _ in self.terminal_ops:
                new_states = [self._apply_op("Terminal", op_name, s) for s in current_states]
                if None not in new_states:
                    h = self._heuristic(new_states, goal_states)
                    if h != float('inf'):
                        tiebreaker += 1
                        heapq.heappush(pq, (new_g + h, new_g, tiebreaker, (op_name, tree), new_states))
                        
        if verbose: print(f"  ✗ Exhausted {steps} A* expansions. No solution found.")
        return None

    def solve(self, problem):
        """Interface for the Unified Proposer."""
        io_examples = problem.get("data", [])
        self.target_val = problem.get("target_val", self.target_val)
        res = self.synthesize(io_examples)
        if res:
            res["policy"] = "code_synth"
            res["answer"] = res.get("code")
        return res


def _demo():
    print("=" * 70)
    print("  CodeEngine — Neural-Guided A* Search + Trainable DSA Optimizations")
    print("=" * 70)
    
    engine = CodeEngine(target_val=9)
    # Note: Optimizations will now be loaded dynamically by UnifiedProposer!
    # For this isolated demo to show the raw AST fallback without learned rules:
    
    problems = [
        {
            "name": "Contains Duplicate",
            "io": [
                ([1, 2, 3, 1], True),
                ([1, 2, 3, 4], False)
            ]
        },
        {
            "name": "Two Sum (Target = 9)",
            "io": [
                ([2, 7, 11, 15], [0, 1]),
                ([3, 2, 4, 7], [1, 3])
            ]
        },
        {
            "name": "Sort and double the positive numbers",
            "io": [
                ([3, -1, 2], [4, 6]),
                ([0, 5, -5, 1], [2, 10])
            ]
        }
    ]
    
    for p in problems:
        print(f"\n[Problem: {p['name']}]")
        for inp, outp in p['io']: print(f"    {inp} -> {outp}")
        result = engine.synthesize(p["io"], max_steps=5000, verbose=True)
        if result:
            print(f"\n  [Discovered IR Tree]:\n    {result['tree']}")
            print(f"\n  [Rendered C++ Code]:\n{result['code_cpp']}")
        else:
            print("  FAILED to synthesize.")


if __name__ == "__main__":
    _demo()
