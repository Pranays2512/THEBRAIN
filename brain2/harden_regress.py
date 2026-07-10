#!/usr/bin/env python3
"""
harden_regress.py — correctness regression: lock in the verified OUTPUTS of the core modules.

harden_test checks nothing crashes. This checks the answers are still RIGHT — the production
behaviour we verified, asserted, so a refactor or a C++ port can be checked against known-good
results. Run before every port; a failure means behaviour drifted.
"""

import math


def main():
    import invariant_miner as IM
    import refuter as RF
    from core.math import factorizer as FZ
    import synth_engine as SE
    import conjecture_sandbox as CSB
    import open_world as OW

    ok = []

    def expect(name, cond):
        ok.append((name, bool(cond)))

    # invariant_miner — production set is clean (no arbitrary-magnitude bound)
    expect("IM no spurious in production", "out_le_1000" not in IM.PREDICATES)
    fac = [(0, 1), (1, 1), (4, 24), (5, 120), (6, 720)]
    mined = IM.validate(IM.mine(fac), [(7, 5040), (8, 40320)])
    expect("IM mines factorial invariants", {"out_positive", "out_ge_first"} <= mined)
    expect("IM rejects n*n (no oracle)", not IM.check(lambda n: n * n, [0, 1, 5], mined)[0])
    expect("IM accepts factorial", IM.check(math.factorial, [0, 1, 5], mined)[0])
    expect("IM functional: gcd commutative", "commutative" in IM.mine_functional(math.gcd, 2))

    # refuter — finds the classic overfit break
    rep = RF.refute(lambda a: max(a + [0]), max, "list")   # init=0 overfit
    expect("RF flags max-init0 overfit", not rep["robust"])

    # factorizer — discovers the shared multiply
    _, _, disc = FZ.factor([("m", ("*", "a", "b")), ("w", ("*", "c", "d")),
                            ("x", ("*", "e", "f"))], min_count=3, min_size=3)
    expect("FZ discovers a*b primitive", disc is not None and disc[2] == 2)

    # synth_engine — solves a known task, verified
    ex = SE._ex("int1", math.factorial, [0, 1, 4, 5, 6])
    name, code = SE.solve(ex, "int1")
    expect("SE solves factorial", code is not None and SE.stress(code, math.factorial, "int1")[0])

    # conjecture_sandbox — admits the right physics, rejects the wrong
    expect("CSB admits 0.5mv^2", CSB.design_and_test(lambda m, v: 0.5 * m * v * v)[0])
    expect("CSB rejects mv", not CSB.design_and_test(lambda m, v: m * v)[0])

    # open_world — rediscovers Kepler ~1.5 from real data
    ps = [math.log(t) / math.log(a) for _, a, t in OW.TRAIN if a != 1.0]
    expect("OW rediscovers Kepler p~1.5", abs(sum(ps) / len(ps) - 1.5) < 0.05)

    # language — structural parse + verified compute, incl. compare + abstain
    from core.reasoning import structural_parser as SP
    from core.grounding import context_embed as CE
    fkb, mem = SP.build_brain()
    PP = SP.StructuralParser({"rocket", "sample"}, {"force", "mass", "speed", "accel", "volume"}, {})
    expect("SP single computes force", "1.2e+04" in (SP.answer(PP.parse("force of the rocket"), fkb, mem) or ""))
    expect("SP compare works", "more" in (SP.answer(PP.parse("which has more mass rocket or sample"), fkb, mem) or ""))
    expect("SP abstains on garbage", SP.answer(PP.parse("the vibe of the rocket"), fkb, mem) is None)
    vecs = CE.build(["the rocket has high speed", "the rocket has high velocity"])
    expect("CE maps velocity->speed", CE.nearest("velocity", ["speed", "mass"], vecs)[0] == "speed")

    # analogy — solar->atom structure map (no shared vocab)
    from core.events import analogy_struct as AS
    solar = [("sun", "heavier", "planet"), ("planet", "revolves", "sun"), ("sun", "pulls", "planet")]
    atom = [("nucleus", "massive", "electron"), ("electron", "circles", "nucleus")]
    emap, _, score = AS.align(solar, atom)
    expect("AS maps sun->nucleus", emap and emap.get("sun") == "nucleus" and score >= 2)

    # semantic_depth — learns a new concept from a definition, verified
    from core.knowledge import semantic_depth as SD
    fkb2, mem2 = SP.build_brain()
    L = SD.ConceptLearner(fkb2, mem2, {"force", "mass", "speed", "accel", "volume"})
    tgt, _ = L.learn_definition("momentum is mass times speed")
    expect("SD learns momentum=m*v", tgt == "momentum" and abs(L.query("rocket", "momentum") - 300000) < 1)

    # C++ ports match Python (run under venv2 where brain2 imports)
    try:
        import brain2
        c = [n * n for n in range(31)]; o = [n * (n + 1) // 2 for n in range(31)]
        cpp = brain2.refute_int1(c, o, 0)
        py = RF.refute(lambda n: n * n, lambda n: n * (n + 1) // 2, "int1")
        expect("C++ refuter matches Python", cpp["scope"] == py["scope"] and cpp["robust"] == py["robust"])
        ce = [([float(a[0])], float(y)) for a, y in [((0,), 1), ((1,), 1), ((4,), 24), ((5,), 120)]]
        expect("C++ inv_mine matches Python", set(brain2.inv_mine(ce)) == (IM.mine([(0, 1), (1, 1), (4, 24), (5, 120)]) - {"monotonic_increasing"}))
        tree = ("+", ("*", "m", "v"), ("*", 0.5, "k"))
        env = {"m": 3.0, "v": 4.0, "k": 8.0}
        expect("C++ eval_sexpr matches Python",
               abs(brain2.eval_sexpr(FZ.to_sexpr(tree), env) - FZ.eval_tree(tree, env, {})) < 1e-9)
        import irregularity_detector as ID
        tr = [(0.387, 0.241), (0.723, 0.615), (1.0, 1.0), (1.524, 1.881), (5.203, 11.862)]
        ho = [(9.537, 29.457), (19.191, 84.02)]
        # compare against the PYTHON REFERENCE (_law_error now prefers the C++ port, so
        # comparing to _law_error would be C++ vs C++ and lock nothing)
        expect("C++ law_error matches Python",
               abs(brain2.law_error(tr, ho) - ID._law_error_py(tr, ho)) < 1e-9)
        import feature_learner as FL
        lp = FL.LearnedProposer()
        lp.proto = {"list": {"out_is_list": 1.0, "out_exceeds_max": 0.0, "neg_in": 0.5},
                    "dp":   {"out_is_list": 0.0, "out_exceeds_max": 1.0, "neg_in": 0.5}}
        pyw = lp._disc_weights()
        cppw = brain2.disc_weights([lp.proto["list"], lp.proto["dp"]])
        fe = {"out_is_list": 1.0, "out_exceeds_max": 0.0, "neg_in": 0.0}
        expect("C++ proposer scoring matches Python",
               abs(lp._sim(fe, lp.proto["list"], pyw) - brain2.feat_sim(fe, lp.proto["list"], cppw)) < 1e-9)
        from core.grounding import context_embed as CE2
        va, vb = {"x": 2.0, "y": 1.0}, {"x": 1.0, "z": 3.0}
        expect("C++ cosine matches Python", abs(brain2.cosine_map(va, vb) - CE2.cosine(va, vb)) < 1e-9)
        from core.events import analogy_struct as AS2
        src = [("sun", "heavier", "planet"), ("planet", "revolves", "sun"), ("sun", "pulls", "planet")]
        tgt = [("nucleus", "massive", "electron"), ("electron", "circles", "nucleus")]
        em = {"sun": "nucleus", "planet": "electron"}
        pysc, _ = AS2._score(src, tgt, em)
        expect("C++ analogy_score matches Python", brain2.analogy_score(src, tgt, em) == pysc)
    except ImportError:
        pass  # brain2 not available under this interpreter; skip the C++ match checks

    print("=== harden_regress — correctness lock on core modules ===\n")
    fails = 0
    for name, good in ok:
        print("  [%s] %s" % (" ok " if good else "FAIL", name))
        fails += not good
    print("\n  %d/%d correct.%s" % (len(ok) - fails, len(ok),
                                    "  ALL LOCKED." if not fails else "  REGRESSION!"))
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
