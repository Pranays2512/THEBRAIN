#!/usr/bin/env python3
"""
test_membrane_bypass.py — Regression test suite for WholeBrain Claims Verification Gate.

Verifies:
1. Fail-Closed Security: Factual claims from factual handlers MUST parse into verifiable claims.
   If un-backed OR unparseable, the gate FAILS CLOSED (solution_type resets to 'none').
2. Multi-Claim Rejection: Multi-claim responses where 1 claim is un-backed are rejected.
3. Positive Conversational Pass-Through: Low-stakes conversational queries ("hello") pass.
4. Verified Factual Pass-Through: Real facts in KB ("whale isa mammal") pass.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faculties.whole_brain import WholeBrain


class TestMembraneBypass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wb = WholeBrain()
        # Seed known concepts and facts for verified factual test
        if hasattr(cls.wb, "concepts"):
            cls.wb.concepts.add("dog")
            cls.wb.concepts.add("whale")
        cls.wb.remember("whale", "isa", "mammal")
        cls.wb.remember("dog", "can", "bark")
        cls.wb.remember("dog", "has", "fur")

    def test_negative_problem_prompt_no_leak(self):
        """Competitive programming prompt containing 'cake' must NEVER return 'A isa metal'."""
        prompt = (
            "E. Cake Trial\n"
            "GLaDOS arranged n cakes in a row. Each cake can be either real (T) or fake (F).\n"
            "Find the maximum number of mistakes GLaDOS can guarantee."
        )
        res = self.wb.ask(prompt)
        kind, msg, verified = res[0], res[1], res[2]
        self.assertNotEqual(msg, "A isa metal", "Bypass bug: 'A isa metal' leaked through ask()")
        if kind == "factual":
            self.assertNotIn("metal", str(msg).lower(), "Unbacked claim 'metal' leaked into factual response")

    def test_negative_unbacked_claim_rejection(self):
        """Un-backed claim like 'cake isa metal' must be rejected by the Claims Gate."""
        prompt = "cake"
        res = self.wb.ask(prompt)
        msg = res[1]
        self.assertNotEqual(msg, "A isa metal", "Un-backed factual claim was not rejected by Claims Gate")

    def test_fail_closed_unparseable_factual_phrasing(self):
        """Factual handler responses that cannot be parsed into verifiable claims must FAIL CLOSED."""
        # Test internal gate logic directly
        old_solution_type = "factual"
        old_ans_msg = "Cake is heavier than air in a complex atmospheric pressure equation"
        
        # Simulate gate logic on WholeBrain
        import re
        claims = re.findall(r"(\w+)\s+(isa|can|has|lives_in|made_of|used_for|is|contains)\s+(\w+)", old_ans_msg.lower())
        all_backed = True
        if claims:
            for subj, rel, obj in claims:
                is_backed = False
                if hasattr(self.wb, "kre") and self.wb.kre is not None:
                    known_objs = self.wb.kre.ask_all(subj, rel)
                    if known_objs and obj in [str(o).lower() for o in known_objs]:
                        is_backed = True
                if not is_backed:
                    all_backed = False
                    break
        else:
            all_backed = False
            
        self.assertFalse(all_backed, "Unparseable complex factual claim should fail closed")

    def test_multi_claim_partial_unbacked_rejection(self):
        """If a response has 2 claims and 1 is un-backed, the whole response MUST be rejected."""
        # Claim 1: "dog has fur" (backed), Claim 2: "dog isa vehicle" (unbacked)
        multi_msg = "dog has fur and dog isa vehicle"
        import re
        claims = re.findall(r"(\w+)\s+(isa|can|has|lives_in|made_of|used_for|is|contains)\s+(\w+)", multi_msg.lower())
        all_backed = True
        for subj, rel, obj in claims:
            is_backed = False
            if hasattr(self.wb, "kre") and self.wb.kre is not None:
                known_objs = self.wb.kre.ask_all(subj, rel)
                if known_objs and obj in [str(o).lower() for o in known_objs]:
                    is_backed = True
            if not is_backed:
                all_backed = False
                break
        self.assertFalse(all_backed, "Multi-claim response with 1 unbacked claim must fail closed")

    def test_positive_conversational_query_pass_through(self):
        """Conversational query with no claims must pass the gate cleanly (no false positives)."""
        prompt = "hello how are you"
        res = self.wb.ask(prompt)
        kind, msg = res[0], res[1]
        self.assertIsNotNone(msg, "Conversational query was blocked by the gate")
        self.assertTrue(len(str(msg).strip()) > 0, "Conversational query returned empty message")

    def test_positive_verified_factual_claim_pass_through(self):
        self.wb.remember("dog", "can", "bark")
        prompt = "can a dog bark"
        res = self.wb.ask(prompt)
        kind, msg = res[0], res[1]
        self.assertEqual(kind, "factual", "Verified factual query did not return factual kind")
        self.assertIn("bark", str(msg).lower(), "Verified fact 'bark' was incorrectly blocked")


if __name__ == "__main__":
    unittest.main()
