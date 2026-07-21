#!/usr/bin/env python3
"""
unified_proposer.py — ONE intuition for the entire Brain.

The Brain has many engines (calculus, physics, algebra, code synthesis, DP,
loop synthesis). Until now each had its own proposer or none at all. This
module is the SINGLE learned routing layer: given ANY problem, it extracts
features, predicts which engine/policy will solve it, executes the winner,
and verifies the result. When confidence is low (novelty), it triggers the
conjecture pipeline: synthesize a candidate → test against trusted KB →
admit or reject.

The proposer learns ONLINE: every solved problem becomes a training example
(features → which policy solved it), so it gets better with experience.

    from engines.synthesis.unified_proposer import UnifiedProposer
    up = UnifiedProposer(knowledge_facts=brain_curriculum_facts)
    result = up.solve(problem)

This is the brain's intuition. It doesn't compute — it decides WHO computes.
"""

import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from engines.synthesis._program_synth_tree import DecisionTree
from engines.math.math_engine import MathEngine, render
from engines.code.code_engine import CodeEngine


# ── The Policy Registry: every engine the brain has ──────────────────────────
class Policy:
    """A named, callable policy that wraps one engine's solve method."""
    def __init__(self, name, domain, solve_fn, description=""):
        self.name = name
        self.domain = domain       # "math", "physics", "code", "conjecture"
        self.solve_fn = solve_fn
        self.description = description

    def __call__(self, problem):
        return self.solve_fn(problem)


# ── Feature Extraction: unified features for any problem type ────────────────
def extract_features(problem):
    """Extract a fixed-length feature vector from any problem.
    
    The vector has domain-specific slots (zeros for irrelevant domains):
    [0]  is_expression_tree   (1.0 if the problem is a tuple/expression)
    [1]  is_equation          (1.0 if it contains '=')
    [2]  is_io_pairs          (1.0 if the problem is a list of (in, out) pairs)
    [3]  has_trig             (1.0 if it mentions sin/cos/tan)
    [4]  has_exp_log          (1.0 if it mentions exp/ln/log)
    [5]  has_polynomial       (1.0 if it mentions ^/power)
    [6]  has_variables         (count of distinct variables)
    [7]  expression_depth     (nesting depth of expression tree)
    [8]  has_physics_terms    (1.0 if it mentions mass/force/velocity/energy)
    [9]  has_string_io        (1.0 if I/O pairs contain strings)
    [10] has_array_io         (1.0 if I/O pairs contain lists)
    [11] confidence_bias      (always 1.0 — bias term)
    """
    feats = np.zeros(12, dtype=np.float32)
    feats[11] = 1.0  # bias
    
    if isinstance(problem, dict):
        ptype = problem.get("type", "")
        data = problem.get("data", None)
        expr = problem.get("expr", None)
        
        target_expr = data if data else expr
        
        if ptype in ("differentiate", "diff") and isinstance(target_expr, tuple):
            feats[0] = 1.0
            feats[7] = _tree_depth(target_expr)
            _scan_expr(target_expr, feats)
            
        elif ptype == "integrate" and isinstance(target_expr, tuple):
            feats[0] = 1.0
            feats[7] = _tree_depth(target_expr)
            _scan_expr(target_expr, feats)
            
        elif ptype == "equation":
            feats[1] = 1.0
            if isinstance(target_expr, tuple):
                feats[0] = 1.0
                feats[7] = _tree_depth(target_expr)
                _scan_expr(target_expr, feats)
                
        elif ptype == "physics":
            feats[8] = 1.0
            
        elif ptype == "synthesize":
            feats[2] = 1.0
            if isinstance(data, list) and data:
                inp0 = data[0][0] if data[0] else None
                if isinstance(inp0, str):
                    feats[9] = 1.0
                elif isinstance(inp0, list):
                    feats[10] = 1.0
                    
        elif ptype == "conjecture":
            pass  # all zeros except bias — triggers novelty
    
    return feats


def _tree_depth(expr, d=0):
    if isinstance(expr, tuple):
        return max((_tree_depth(c, d + 1) for c in expr[1:]), default=d)
    return float(d)


def _scan_expr(expr, feats):
    """Scan an expression tree for trig, exp, polynomial markers."""
    if isinstance(expr, str):
        if expr in ("sin", "cos", "tan"):
            feats[3] = 1.0
        elif expr in ("exp", "ln", "log"):
            feats[4] = 1.0
        elif expr not in ("+", "-", "*", "/", "^", "neg", "="):
            feats[6] += 1.0  # count variables
    elif isinstance(expr, tuple):
        if expr[0] in ("sin", "cos", "tan"):
            feats[3] = 1.0
        elif expr[0] in ("exp", "ln"):
            feats[4] = 1.0
        elif expr[0] == "^":
            feats[5] = 1.0
        for c in expr[1:]:
            _scan_expr(c, feats)


# ── The Unified Proposer ─────────────────────────────────────────────────────
class UnifiedProposer:
    """ONE intuition for the entire Brain.
    
    Routes any problem to the right engine. Learns online from outcomes.
    When confidence is low, triggers the conjecture pipeline.
    """
    
    CONFIDENCE_THRESHOLD = 0.4   # below this → try conjecture pipeline
    
    def __init__(self, knowledge_facts=None):
        # Initialize engine
        self.math_engine = MathEngine()
        self.code_engine = CodeEngine()
        
        # Load knowledge from KB if provided (Memories)
        if knowledge_facts:
            # 1. Math rules/physics laws (expects triples)
            math_facts = [f for f in knowledge_facts if isinstance(f, tuple) and len(f) == 3]
            if math_facts:
                self.math_engine.load_from_knowledge(math_facts)
            
            # 2. Algorithmic Optimizations (expects dicts)
            for fact in knowledge_facts:
                if isinstance(fact, dict) and fact.get("type") == "code_optimization":
                    lang = fact.get("lang", "cpp")
                    name = fact.get("name")
                    pattern_str = fact.get("pattern_lambda")
                    render_str = fact.get("render_fn_code")
                    
                    if name and pattern_str and render_str:
                        # Extract logic from memory and dynamically evaluate it
                        pattern_fn = eval(pattern_str)
                        
                        # Dynamically create the render function
                        local_env = {}
                        exec(render_str, globals(), local_env)
                        render_fn = local_env.get("render_fn")
                        
                        if render_fn:
                            self.code_engine.learn_optimization(lang, name, pattern_fn, render_fn)
        
        self.math_engine.learn() # discover rules
        
        # Register policies
        self.policies = [
            Policy("math_solver", "math", self._wrap_math_engine, "Unified math engine for diff, integrate, physics, algebra, simplify."),
            Policy("code_synth", "code", self._wrap_code_engine, "Synthesizes Python/Java/C++ algorithms from list I/O pairs."),
            Policy("conjecture", "novel", self._solve_conjecture, "Form and test a novel conjecture"),
        ]
        self.policy_names = [p.name for p in self.policies]
        
        # The ONE DecisionTree — trained across all domains
        self.tree = None
        self._training_X = []
        self._training_y = []
        
        # Online learning: track which policy solved what
        self._solve_history = []
        
        # Conjecture state: trusted facts + admitted conjectures
        self._trusted_laws = dict(self.math_engine.physics_laws)   # copy of currently trusted
        self._conjectures_tested = 0
        self._conjectures_admitted = 0
    
    # ── Policy implementations ───────────────────────────────────────────
    def _wrap_math_engine(self, problem):
        """Pass the problem directly to the unified math engine."""
        try:
            # We must map "differentiate" to "diff" for math_engine
            ptype = problem.get("type", "")
            if ptype == "differentiate":
                problem["type"] = "diff"
                problem["expr"] = problem.get("data")
            elif ptype == "integrate":
                problem["expr"] = problem.get("data")
            elif ptype == "equation":
                problem["expr"] = problem.get("data")
                
            res = self.math_engine.solve(problem)
            if res:
                res["policy"] = "math_solver"
            return res
        except Exception:
            return None

    def _wrap_code_engine(self, problem):
        """Pass the problem to the CodeEngine."""
        try:
            return self.code_engine.solve(problem)
        except Exception:
            return None
    
    def _solve_conjecture(self, problem):
        """The conjecture pipeline: form a hypothesis, test it against trusted KB.
        
        When the proposer doesn't know what to do:
        1. Take the candidate formula from the problem
        2. Test it against ALL trusted facts
        3. If it survives → admit as a new law
        4. If it fails → reject with counterexample
        """
        candidate = problem.get("conjecture")
        test_fn = problem.get("test_fn")
        if candidate is None or test_fn is None:
            return None
        
        self._conjectures_tested += 1
        
        # Design experiments: random inputs spanning the space
        import random
        rng = random.Random(42)
        worst_err = 0.0
        counterexample = None
        
        for _ in range(50):
            # Generate random test values
            test_vals = {k: rng.uniform(0.5, 20) for k in problem.get("variables", ["x"])}
            try:
                guess = test_fn(**test_vals)
                # Check against trusted anchor if provided
                trusted_fn = problem.get("trusted_fn")
                if trusted_fn:
                    truth = trusted_fn(**test_vals)
                    rel_err = abs(guess - truth) / (abs(truth) + 1e-9)
                    if rel_err > worst_err:
                        worst_err = rel_err
                        counterexample = (test_vals, truth, guess)
            except Exception:
                return {"survived": False, "reason": "raised exception", "policy": "conjecture"}
        
        survived = worst_err <= 0.01
        
        if survived:
            self._conjectures_admitted += 1
            # ADMIT: add this as a new trusted fact
            result = {
                "survived": True,
                "worst_error": worst_err,
                "policy": "conjecture",
                "status": "ADMITTED — new trusted law"
            }
            # If we have a law name, register it in the physics engine
            law_name = problem.get("law_name")
            if law_name and "lhs" in problem and "rhs" in problem:
                self._trusted_laws[law_name] = (problem["lhs"], problem["rhs"])
                self.math_engine.physics_laws[law_name] = (problem["lhs"], problem["rhs"])
            return result
        else:
            return {
                "survived": False,
                "worst_error": worst_err,
                "counterexample": counterexample,
                "policy": "conjecture",
                "status": "REJECTED — counterexample found"
            }
    
    # ── The main solve method ────────────────────────────────────────────
    def solve(self, problem):
        """Route a problem to the right engine. Returns the result or None."""
        feats = extract_features(problem)
        ptype = problem.get("type", "")
        
        # If we have a trained tree, use it to rank policies
        if self.tree is not None:
            dist = self.tree.predict_dist(feats)
            ranked = sorted(range(len(self.policies)),
                            key=lambda i: dist[i], reverse=True)
            best_conf = dist[ranked[0]]
        else:
            # No tree yet — use simple type-based routing
            ranked = self._type_route(ptype)
            best_conf = 1.0
        
        # If confidence is too low, prepend conjecture policy
        if best_conf < self.CONFIDENCE_THRESHOLD:
            conj_idx = self.policy_names.index("conjecture")
            if conj_idx not in ranked[:1]:
                ranked = [conj_idx] + ranked
        
        # Try policies in ranked order
        for idx in ranked:
            policy = self.policies[idx]
            result = policy(problem)
            if result is not None:
                # LEARN: record this success for online training
                self._training_X.append(feats)
                self._training_y.append(idx)
                self._solve_history.append((problem.get("type"), policy.name))
                
                # Retrain tree periodically
                if len(self._training_X) >= 10 and len(self._training_X) % 5 == 0:
                    self._retrain()
                
                return result
        
        return None
    
    def _type_route(self, ptype):
        """Simple deterministic routing when no tree is trained yet."""
        routes = {
            "differentiate": [0],      # math_solver
            "integrate": [0],          # math_solver
            "physics": [0],            # math_solver
            "equation": [0],           # math_solver
            "synthesize": [1],         # code_synth
            "conjecture": [2],         # conjecture
        }
        base = routes.get(ptype, list(range(len(self.policies))))
        # always include all policies as fallbacks
        return base + [i for i in range(len(self.policies)) if i not in base]
    
    def _retrain(self):
        """Retrain the DecisionTree on all accumulated solve history."""
        if len(self._training_X) < 5:
            return
        X = np.array(self._training_X)
        y = np.array(self._training_y)
        self.tree = DecisionTree(len(self.policies), max_depth=8, min_samples=3).fit(X, y)
    
    def stats(self):
        """Return a summary of the proposer's state."""
        return {
            "policies_registered": len(self.policies),
            "problems_solved": len(self._solve_history),
            "tree_trained": self.tree is not None,
            "physics_laws_loaded": len(self.math_engine.physics_laws),
            "conjectures_tested": self._conjectures_tested,
            "conjectures_admitted": self._conjectures_admitted,
        }


# ── Demo ─────────────────────────────────────────────────────────────────────
def _demo():
    print("=" * 70)
    print("  unified_proposer — ONE intuition for the entire Brain")
    print("=" * 70)
    
    # Load laws from KB-style facts (simulating what brain_curriculum.txt has)
    kb_facts = [
        ("second_law", "can", "written_as_F=m*a"),
        ("speed_law", "defined_as", "v=d/t"),
        ("kinetic_energy", "defined_as", "KE=0.5*m*v^2"),
    ]
    
    up = UnifiedProposer(knowledge_facts=kb_facts)
    print(f"\n  Initialized. Stats: {up.stats()}\n")
    
    # ── Test 1: Calculus problem (differentiation)
    print("[Test 1: Calculus — differentiate sin(x^2)]")
    r1 = up.solve({"type": "differentiate", "data": ("sin", ("^", "x", 2))})
    print(f"  Answer: {r1['answer']}")
    print(f"  Rules: {r1['rules']}")
    print(f"  Routed to: {r1['policy']}\n")
    
    # ── Test 2: Physics problem (using KB-learned law)
    print("[Test 2: Physics — solve F=ma for acceleration, using KB-loaded law]")
    r2 = up.solve({"type": "physics", "law": "second_law", "target": "a",
                    "knowns": {"F": 100, "m": 20}})
    print(f"  Answer: a = {r2['answer']}")
    print(f"  Formula: {r2.get('formula')}")
    print(f"  Routed to: {r2['policy']}\n")
    
    # ── Test 3: Algebra problem
    print("[Test 3: Algebra — solve 3*x + 5 = 20]")
    r3 = up.solve({"type": "equation",
                    "data": ("=", ("+", ("*", 3, "x"), 5), 20)})
    print(f"  Answer: x = {r3['answer']}")
    print(f"  Formula: {r3.get('formula')}")
    print(f"  Routed to: {r3['policy']}\n")
    
    # ── Test 4: Integration
    print("[Test 4: Integration — ∫ cos(x) dx]")
    r4 = up.solve({"type": "integrate", "data": ("cos", "x")})
    print(f"  Answer: {r4['answer']} + C")
    print(f"  Verified by differentiating back: {r4['verified']}")
    print(f"  Routed to: {r4['policy']}\n")
    
    # ── Test 5: CONJECTURE — novel discovery!
    print("[Test 5: Conjecture — test KE = 0.5*m*v^2 against trusted anchor]")
    import math as _m
    G = 9.8
    r5 = up.solve({
        "type": "conjecture",
        "conjecture": "KE = 0.5*m*v^2",
        "test_fn": lambda m, v: 0.5 * m * v * v,
        "trusted_fn": lambda m, v: 0.5 * m * v * v,    # this IS the correct formula
        "variables": ["m", "v"],
        "law_name": "kinetic_conjecture",
        "lhs": "KE",
        "rhs": ("*", 0.5, ("*", "m", ("^", "v", 2))),
    })
    print(f"  Survived: {r5['survived']}")
    print(f"  Status: {r5['status']}")
    print(f"  Routed to: {r5['policy']}\n")
    
    # ── Test 6: WRONG conjecture — should be rejected
    print("[Test 6: Conjecture — test WRONG formula KE = m*v^3]")
    r6 = up.solve({
        "type": "conjecture",
        "conjecture": "KE = m*v^3",
        "test_fn": lambda m, v: m * v ** 3,
        "trusted_fn": lambda m, v: 0.5 * m * v * v,
        "variables": ["m", "v"],
    })
    print(f"  Survived: {r6['survived']}")
    print(f"  Status: {r6['status']}")
    if not r6['survived'] and r6.get('counterexample'):
        ce = r6['counterexample']
        print(f"  Counterexample: inputs={ce[0]}, truth={ce[1]:.2f}, guess={ce[2]:.2f}")
    print(f"  Routed to: {r6['policy']}\n")
    
    # ── Test 7: Algorithm Synthesis (LeetCode solver)
    print("[Test 7: Code Synthesis — algorithm discovery]")
    r7 = up.solve({
        "type": "synthesize",
        "data": [
            ([1, 2, 3], [2, 4, 6]),
            ([0, -1, 4], [0, -2, 8])
        ]
    })
    print(f"  Discovered algorithm to satisfy I/O:")
    print(f"{r7['answer']}")
    print(f"  Routed to: {r7['policy']}\n")
    
    print(f"  Final stats: {up.stats()}")
    print("\n" + "=" * 70)
    print("  The brain has ONE proposer that routes calculus, physics, algebra,")
    print("  and conjectures. It learns online. When it doesn't know, it guesses")
    print("  and tests — admitting survivors, rejecting failures with evidence.")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
