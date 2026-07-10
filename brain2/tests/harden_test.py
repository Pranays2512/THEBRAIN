#!/usr/bin/env python3
"""
harden_test.py — surface fragility in the core modules before porting/integration (step 1).

Runs each proven module against EDGE CASES (empty, single, malformed, extreme inputs). A
robust module either handles them or fails cleanly; a crash is a hardening bug to fix. This
is the find-the-fragility pass — fixes follow.
"""

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, "OK", ""))
    except Exception as e:
        RESULTS.append((name, "CRASH", "%s: %s" % (type(e).__name__, e)))


def run():
    from core.synthesis import invariant_miner as IM
    from core.synthesis import refuter as RF
    from core.math import factorizer as FZ
    from core.synthesis import synth_engine as SE
    from core.grounding import context_embed as CE

    # invariant_miner edge cases
    check("IM.mine empty", lambda: IM.mine([]))
    check("IM.mine single", lambda: IM.mine([(0, 1)]))
    check("IM.check empty fn", lambda: IM.check(lambda n: n, [], set()))
    check("IM.mine_functional bad", lambda: IM.mine_functional(lambda a, b: a / b, 2))  # div by zero probe

    # refuter edge cases
    check("RF.refute always-raise", lambda: RF.refute(lambda n: 1 / 0, abs, "int1", n=10))
    check("RF.refute correct", lambda: RF.refute(lambda n: n, lambda n: n, "int1", n=10))

    # factorizer edge cases
    check("FZ.factor empty", lambda: FZ.factor([]))
    check("FZ.factor single", lambda: FZ.factor([("a", ("*", "x", "y"))]))
    check("FZ.factor_au empty", lambda: FZ.factor_au([]))
    check("FZ.eval_tree leaf", lambda: FZ.eval_tree("x", {"x": 3}, {}))
    check("FZ.eval_tree div0", lambda: FZ.eval_tree(("/", "x", "y"), {"x": 1, "y": 0}, {}))

    # synth_engine edge cases
    check("SE.solve empty", lambda: SE.solve([], "int1"))
    check("SE.solve impossible", lambda: SE.solve([((0,), 1), ((1,), 999)], "int1"))
    check("SE.stress no oracle pts", lambda: SE.stress("def f(n):\n return n\n", lambda n: 1 / 0, "int1", n=5))

    # context_embed edge cases
    check("CE.build empty", lambda: CE.build([]))
    check("CE.nearest unknown", lambda: CE.nearest("zzz", ["speed"], CE.build(["the speed of light"])))

    # parsers — garbage / empty / no entities
    from core.reasoning import structural_parser as SP
    from core.reasoning import nested_parser as NP
    from core.reasoning import deeper_grammar as DG
    fkb, mem = SP.build_brain()
    P = SP.StructuralParser({"rocket"}, {"mass"}, {})
    check("SP.parse empty", lambda: P.parse(""))
    check("SP.parse garbage", lambda: P.parse("!@#$ %^&*"))
    check("SP.parse no-tokens", lambda: P.parse("the of a"))
    check("NP.answer empty", lambda: NP.NestedParser(fkb, mem).answer(""))
    check("NP.answer half-if", lambda: NP.NestedParser(fkb, mem).answer("if then"))
    check("DG.answer no-if", lambda: DG.DeeperParser(fkb, mem).answer("hello world"))
    check("DG.answer empty-cond", lambda: DG.DeeperParser(fkb, mem).answer("if then what"))

    # sandbox / irregularity — degenerate data
    from core.synthesis import conjecture_sandbox as CSB
    from core.synthesis import irregularity_detector as ID
    check("CSB.test raises", lambda: CSB.design_and_test(lambda m, v: 1 / 0, n=5))
    check("ID.assess empty", lambda: ID.assess([], []))
    check("ID.assess single", lambda: ID.assess([(1, 1)], [(2, 2)]))

    # proposers — empty / single
    from core.synthesis import online_proposer as OP
    from faculties import feature_learner as FL
    check("OP.solve empty", lambda: OP.OnlineProposer().solve([], "int1"))
    check("FL.features empty", lambda: FL.gen_features([], "int1"))
    check("FL.solve empty", lambda: FL.LearnedProposer().solve([], "list"))

    # LMs — empty / tiny corpus
    from core.math import prob_compute as PC
    check("PC.train empty", lambda: PC.ProbLM().train([]))
    check("PC.generate untrained", lambda: PC.ProbLM().train(["a b c"]).generate())
    check("PC.dist unknown-ctx", lambda: PC.ProbLM().train(["a b c"]).dist(["zzz"]))

    # analogy / semantic — empty / malformed
    from core.events import analogy_struct as AS
    from core.knowledge import semantic_depth as SD
    check("AS.align empty", lambda: AS.align([], []))
    check("AS.align_greedy empty", lambda: AS.align_greedy([], []))
    fkb2, mem2 = SP.build_brain()
    check("SD.learn malformed", lambda: SD.ConceptLearner(fkb2, mem2, {"mass"}).learn_definition("blah"))

    print("=== harden_test — fragility scan of core modules ===\n")
    crashes = 0
    for name, status, detail in RESULTS:
        mark = "  ok  " if status == "OK" else " CRASH"
        print("  [%s] %-26s %s" % (mark, name, detail))
        crashes += status == "CRASH"
    print("\n  %d/%d clean, %d crash(es) to harden." % (len(RESULTS) - crashes, len(RESULTS), crashes))


if __name__ == "__main__":
    run()
