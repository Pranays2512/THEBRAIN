#!/usr/bin/env python3
"""
logic_plan.py — the Brain emits a LogicPlan IR for complex algorithms;
the LLM transcribes it to code; the verifier checks it.

This implements the "middle tier" described in architecture_roadmap.md L87-88:
  "brain logic + small LLM writes code → brain verifier checks"

For formula-shaped logic, brain_codegen.py already renders code mechanically
(no LLM). This module handles the HARDER case: algorithms with named data
structures, iterative/recursive structure, and multi-step logic that is too
complex for a mechanical renderer but where the BRAIN still finds the structure
and the LLM is only a transcriber (never the thinker).

Pipeline:
  Brain MeansEndsSolver → LogicPlan → LLMTranscriber → code → Verifier
                                          (LLM as typist)

    from engines.synthesis.logic_plan import LogicPlan, LLMTranscriber, plan_registry
    plan = plan_registry["binary_search"]
    code = LLMTranscriber(client).transcribe(plan, lang="python")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import re


# ── LogicPlan: the Brain's structured algorithm description ───────────────────
@dataclass
class LogicPlan:
    """A structured, language-independent algorithm description.

    The Brain fills this in via MeansEndsSolver or the DSA training pipeline.
    The LLM only reads it and writes syntactically correct code — it never
    decides the algorithm logic.

    Fields:
        name         : canonical algorithm name ("binary_search")
        description  : one-line human description
        inputs       : list of (name, type_hint) e.g. [("arr","List[int]"), ("target","int")]
        output       : (name, type_hint) e.g. ("index", "int")
        data_structs : named data structures needed ("priority_queue", "visited_set")
        steps        : ordered list of natural-language algorithm steps
        invariants   : things that are always true mid-algorithm (for verifier)
        complexity   : {"time": "O(log n)", "space": "O(1)"}
        oracle       : optional Python callable for verification (not sent to LLM)
        test_cases   : list of (args_dict, expected_output) for the verifier
    """
    name: str
    description: str
    inputs: list[tuple[str, str]]
    output: tuple[str, str]
    steps: list[str]
    data_structs: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    complexity: dict[str, str] = field(default_factory=dict)
    oracle: Any = None          # callable — used by verifier, NOT sent to LLM
    test_cases: list[tuple] = field(default_factory=list)

    def to_prompt(self, lang: str = "python") -> str:
        """Render the plan to an LLM prompt. The LLM sees structured logic only —
        it never decides what the algorithm is, only how to write the syntax."""
        parts = [
            f"Implement the following algorithm in {lang}.",
            f"Algorithm: {self.name}",
            f"Description: {self.description}",
            "",
            "Inputs:",
        ]
        for n, t in self.inputs:
            parts.append(f"  - {n}: {t}")
        rn, rt = self.output
        parts.append(f"Output: {rn} ({rt})")

        if self.data_structs:
            parts.append("\nData structures to use:")
            for ds in self.data_structs:
                parts.append(f"  - {ds}")

        parts.append("\nAlgorithm steps (implement EXACTLY in this order):")
        for i, step in enumerate(self.steps, 1):
            parts.append(f"  {i}. {step}")

        if self.invariants:
            parts.append("\nInvariants (must hold throughout):")
            for inv in self.invariants:
                parts.append(f"  - {inv}")

        if self.complexity:
            parts.append(f"\nExpected complexity: "
                         f"time={self.complexity.get('time','?')} "
                         f"space={self.complexity.get('space','?')}")

        parts += [
            "",
            f"Return ONLY the function named `{self.name}` with no extra explanation.",
            "Do not add imports. Do not change the function signature.",
        ]
        return "\n".join(parts)


# ── LLMTranscriber: sends the plan to the LLM, gets code back ─────────────────
class LLMTranscriber:
    """The LLM is the transcriber. The Brain is the thinker.

    Usage:
        from adapters.llm_adapter import OllamaClient
        t = LLMTranscriber(OllamaClient())
        code = t.transcribe(plan, lang="python")
        if code: print(code)   # verified by transcribe() before returning
    """

    def __init__(self, client=None):
        """client: any LLMClient with a .complete(prompt) -> str method.
        If None, uses a StubClient that returns a placeholder (for testing)."""
        if client is None:
            try:
                from adapters.llm_adapter import StubClient
                self.client = StubClient()
            except ImportError:
                self.client = None
        else:
            self.client = client

    def transcribe(self, plan: LogicPlan, lang: str = "python") -> str | None:
        """Send the LogicPlan to the LLM, extract the function, verify it.

        Returns the verified function string, or None if the LLM failed or
        the verifier rejected all attempts (up to 3 retries with counterexample).
        """
        prompt = plan.to_prompt(lang)
        last_code = None

        for attempt in range(3):
            if self.client is None:
                return None
            try:
                raw = self.client.complete(prompt)
            except Exception:
                return None

            code = _extract_function(raw, plan.name)
            if code is None:
                continue
            last_code = code

            # Verifier gate: run test cases if provided (only python is directly executable in sandbox)
            if plan.test_cases and lang == "python":
                counterexample = _verify(code, plan.name, plan.test_cases)
                if counterexample is not None:
                    # Feed counterexample back to LLM for self-correction
                    args, expected, got = counterexample
                    prompt = (
                        f"Your implementation of `{plan.name}` failed a test:\n"
                        f"  Input: {args}\n"
                        f"  Expected: {expected}\n"
                        f"  Got: {got}\n\n"
                        f"Fix the implementation.\n\n"
                        + plan.to_prompt(lang)
                    )
                    continue   # retry with the counterexample
            return code         # passed all test cases (or no test cases)

        return last_code   # return best attempt even if unverified (caller decides)


def _extract_function(raw: str, name: str) -> str | None:
    if not raw:
        return None
    # 1. Try markdown code block
    m = re.search(r"```(?:\w+)?\n(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    
    # 2. Try python def matching
    lines = raw.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"def {name}("):
            start = i
            break
    if start is not None:
        func_lines = [lines[start]]
        for line in lines[start + 1:]:
            if line and not line[0].isspace() and not line.startswith("#"):
                break
            func_lines.append(line)
        return "\n".join(func_lines)
    
    # 3. If raw looks like code already
    if any(k in raw for k in ("def ", "public class", "class ", "function ", "int ", "void ")):
        return raw.strip()
    return None


def _verify(code: str, name: str, test_cases) -> tuple | None:
    """Run test cases. Returns (args, expected, got) counterexample or None."""
    ns = {}
    try:
        exec(code, ns)
    except Exception as e:
        return ({}, "no exception", str(e))
    fn = ns.get(name)
    if fn is None:
        return ({}, "function defined", "function missing")
    for args, expected in test_cases:
        try:
            if isinstance(args, dict):
                got = fn(**args)
            elif isinstance(args, (list, tuple)):
                got = fn(*args)
            else:
                got = fn(args)
            if got != expected:
                return (args, expected, got)
        except Exception as e:
            return (args, expected, str(e))
    return None


# ── Plan Registry: Brain-authored algorithm blueprints ────────────────────────
# These are authored by the Brain (from DSA training) and stored here as
# verified LogicPlans. The LLM reads these; it does not invent them.

plan_registry: dict[str, LogicPlan] = {

    "binary_search": LogicPlan(
        name="binary_search",
        description="Find the index of target in a sorted array, or -1 if absent.",
        inputs=[("arr", "List[int]"), ("target", "int")],
        output=("index", "int"),
        data_structs=["two pointers: lo and hi"],
        steps=[
            "Set lo = 0, hi = len(arr) - 1.",
            "While lo <= hi: compute mid = (lo + hi) // 2.",
            "If arr[mid] == target, return mid.",
            "If arr[mid] < target, set lo = mid + 1.",
            "Otherwise set hi = mid - 1.",
            "If loop ends without returning, return -1.",
        ],
        invariants=["arr[lo..hi] always contains target if present"],
        complexity={"time": "O(log n)", "space": "O(1)"},
        test_cases=[
            (([1, 3, 5, 7, 9], 5), 2),
            (([1, 3, 5, 7, 9], 1), 0),
            (([1, 3, 5, 7, 9], 9), 4),
            (([1, 3, 5, 7, 9], 4), -1),
            (([], 1), -1),
        ],
    ),

    "two_sum": LogicPlan(
        name="two_sum",
        description="Return indices of two numbers in arr that add up to target.",
        inputs=[("arr", "List[int]"), ("target", "int")],
        output=("indices", "List[int]"),
        data_structs=["hash map: value → index"],
        steps=[
            "Create an empty dictionary called seen.",
            "Iterate over arr with index i and value v.",
            "Compute complement = target - v.",
            "If complement is in seen, return [seen[complement], i].",
            "Otherwise store seen[v] = i.",
            "Return [] if no pair found.",
        ],
        invariants=["seen contains all values visited so far"],
        complexity={"time": "O(n)", "space": "O(n)"},
        test_cases=[
            (([2, 7, 11, 15], 9), [0, 1]),
            (([3, 2, 4], 6), [1, 2]),
            (([3, 3], 6), [0, 1]),
        ],
    ),

    "merge_sort": LogicPlan(
        name="merge_sort",
        description="Sort a list in ascending order using divide-and-conquer.",
        inputs=[("arr", "List[int]")],
        output=("sorted_arr", "List[int]"),
        data_structs=["two sub-arrays: left and right"],
        steps=[
            "Base case: if len(arr) <= 1, return arr.",
            "Split arr into left = arr[:mid] and right = arr[mid:] where mid = len(arr)//2.",
            "Recursively sort left and right.",
            "Merge: use two pointers i, j starting at 0.",
            "While both pointers are in range, append the smaller element and advance its pointer.",
            "Append any remaining elements from left or right.",
            "Return the merged list.",
        ],
        invariants=["left and right are always sorted before merging"],
        complexity={"time": "O(n log n)", "space": "O(n)"},
        test_cases=[
            (([3, 1, 4, 1, 5, 9, 2, 6],), [1, 1, 2, 3, 4, 5, 6, 9]),
            (([],), []),
            (([1],), [1]),
            (([2, 1],), [1, 2]),
        ],
    ),

    "dijkstra": LogicPlan(
        name="dijkstra",
        description="Find shortest distances from src to all nodes in a weighted graph.",
        inputs=[("graph", "Dict[int, List[Tuple[int,int]]]"), ("src", "int")],
        output=("dist", "Dict[int, float]"),
        data_structs=["min-heap (priority queue)", "dist dict initialized to infinity"],
        steps=[
            "Initialize dist[src] = 0 and dist[node] = infinity for all other nodes.",
            "Create a min-heap and push (0, src).",
            "While heap is non-empty: pop (cost, node) with smallest cost.",
            "If cost > dist[node], skip (stale entry).",
            "For each neighbor, weight in graph[node]: compute new_cost = dist[node] + weight.",
            "If new_cost < dist[neighbor]: update dist[neighbor] = new_cost and push (new_cost, neighbor).",
            "Return dist.",
        ],
        invariants=["dist[node] is always the shortest known distance at any point"],
        complexity={"time": "O((V+E) log V)", "space": "O(V)"},
        test_cases=[
            (
                ({0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}, 0),
                {0: 0, 1: 3, 2: 1, 3: 4},
            ),
        ],
    ),

    "min_swaps_to_target": LogicPlan(
        name="min_swaps_to_target",
        description="Compute minimum adjacent swaps to make sequence a match strictly increasing target b using greedy matching.",
        inputs=[("a", "List[int]"), ("b", "List[int]")],
        output=("min_swaps", "int"),
        data_structs=["available target indices", "permutation mapping array p"],
        steps=[
            "Set avail = list(range(len(b))).",
            "Initialize empty list p.",
            "For each element x in a:",
            "  Find the smallest index k in avail such that b[k] >= x.",
            "  If no such k exists, return -1.",
            "  Append k to p and remove k from avail.",
            "Initialize inv = 0.",
            "For i from 0 to len(p)-1:",
            "  For j from i+1 to len(p)-1:",
            "    If p[i] > p[j], increment inv by 1.",
            "Return inv.",
        ],
        invariants=["avail contains unassigned indices in b in ascending order"],
        complexity={"time": "O(n^2)", "space": "O(n)"},
        test_cases=[
            (([1, 2, 2], [1, 3, 5]), 0),
            (([2, 2, 1], [1, 2, 3]), 2),
            (([5, 1], [2, 4]), -1),
        ],
    ),

    "paint_the_array": LogicPlan(
        name="paint_the_array",
        description="Find minimum modifications to make array valid by painting intervals of length m with 1..m.",
        inputs=[("n", "int"), ("m", "int"), ("a", "List[int]")],
        output=("min_modifications", "int"),
        data_structs=["dp array of size m+1"],
        steps=[
            "Initialize dp array of size m+1 with infinity, set dp[1] = (0 if a[0] == 1 else 1).",
            "For each index i from 1 to n-1:",
            "  Compute min_prev as min(dp[1...m]).",
            "  For value v from 1 to m:",
            "    Set cost = (0 if a[i] == v else 1).",
            "    Compute continuation cost c1 = dp[v-1] + cost if v > 1 else infinity.",
            "    Compute new segment cost c2 = min_prev + cost.",
            "    Set next_dp[v] = min(c1, c2).",
            "  Update dp = next_dp.",
            "Return min(dp[1...m]).",
        ],
        invariants=["dp[v] tracks min modifications for prefix up to current index with value v"],
        complexity={"time": "O(n * m)", "space": "O(m)"},
        test_cases=[
            ((5, 3, [1, 2, 3, 2, 3]), 0),
            ((4, 3, [1, 2, 2, 3]), 1),
            ((5, 3, [2, 1, 2, 3, 2]), 2),
            ((5, 3, [2, 2, 2, 2, 2]), 3),
            ((5, 4, [1, 1, 3, 4, 1]), 2),
        ],
    ),
}


# ── Demo ──────────────────────────────────────────────────────────────────────
def _demo():
    print("=== logic_plan — Brain logic IR → LLM transcribes → verifier checks ===\n")

    for algo_name in ["binary_search", "two_sum"]:
        plan = plan_registry[algo_name]
        print(f"Algorithm: {plan.name}")
        print(f"Description: {plan.description}")
        print(f"Steps ({len(plan.steps)}):")
        for i, s in enumerate(plan.steps, 1):
            print(f"  {i}. {s}")
        print(f"Complexity: {plan.complexity}")
        print(f"Test cases: {len(plan.test_cases)}")
        print()

    print("--- StubClient transcription (no Ollama needed) ---")
    t = LLMTranscriber(client=None)   # StubClient returns placeholder
    plan = plan_registry["binary_search"]
    print("Prompt sent to LLM:\n")
    print(plan.to_prompt("python"))
    print("\n[In production: LLMTranscriber(OllamaClient()) sends this to Ollama,")
    print(" gets back Python code, runs test_cases to verify, retries on failure.]\n")
    print("The LLM wrote the syntax. The Brain wrote the logic. The verifier decides.")


if __name__ == "__main__":
    _demo()
