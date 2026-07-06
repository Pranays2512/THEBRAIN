#!/usr/bin/env python3
"""train_all.py — ONE pass that trains EVERYTHING together, on the whole corpus.

The other trainers do a slice: train_from_data does one file, train_pipeline does the LM
distillation. This is the unified driver — every grade-1-9 file, every faculty, in one
process, membrane intact, then the self-improving loop runs and it all persists so the next
session starts where this one ended.

  Stage 1  SYMBOLIC   facts + VERIFIED laws (multi-hop admission) + type closure
  Stage 2  MEMBRANE   pooled verb constraints (frac=0.5) + event predictor + questions
  Stage 3  MORPH      learned morphology -> the mouth
  Stage 4  DIMENSIONS unit/dimension verifier over the laws
  Stage 5  FUZZY      C++ SOM grounding over the raw text            (--brain)
  Stage 6  PROBABILISTIC  owned neural LM over parse pairs           (--lm, heavy/torch)
  Stage 7  FACULTIES  cross-domain / induce / blend + curiosity (standing loop)
  Stage 8  PERSIST    policies + facts + concepts + semantic -> brain_store (survives restart)

    /opt/homebrew/bin/python3.13 train_all.py                 # symbolic + faculties + persist
    KMP_DUPLICATE_LIB_OK=TRUE venv2/bin/python3 train_all.py --brain --lm   # + SOM + owned LM
"""
import os
import re
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from brain_data import BrainData
from type_oracle import TypeOracle
import knowledge_distill as KD
from means_ends import PolicyMemory

FILES = (["data/taxonomy_core.txt"]
         + [f"data/math{i}.txt" for i in range(1, 10)]
         + [f"data/english{i}.txt" for i in range(1, 10)]
         + [f"data/science{i}.txt" for i in range(3, 10)]
         + [f"data/ssc{i}.txt" for i in range(6, 10)])


def _laws_of(d):
    out = []
    for law in d.laws:
        body = law[4:].strip() if law.startswith("LAW:") else law
        m = re.match(r"([a-zA-Z_]\w*)\s*=\s*(.+)", body)
        if not m:
            continue
        t = KD.infix_to_tree(m.group(2))
        if t is not None and any(isinstance(x, str) for x in KD._vars(t)):
            out.append((m.group(1).lower(), t))
    return out


def train(do_lm=False, do_brain=False, files=FILES):
    rep, t0 = {}, time.time()
    datas = [BrainData.from_file(f) for f in files if os.path.exists(f)]
    for d in datas:
        d.load_morph()
    print(f"loaded {len(datas)} files", flush=True)

    # ── Stage 1: SYMBOLIC — facts + verified laws (multi-hop admission) ──────────
    fkb, mem = KD.SimpleKB(), PolicyMemory()
    all_isa, all_ents, all_verbs = [], set(), set()
    facts, laws, parse_pairs, morph, texts = [], [], [], {}, []
    for d in datas:
        for e, r, v in d.facts:
            fkb.learn(e, r, v)
            facts.append((e, r, v))
        laws += _laws_of(d)
        all_isa += [(c, "isa", p) for c, p in d.isa]
        all_ents |= d.entities()
        all_verbs |= d.verbs()
        parse_pairs += d.parse_pairs
        morph.update(d.morph)
        texts += [s for s, _ in d.events] + d.sequences
    # multi-hop: laws whose inputs are DERIVED (other laws) admit to a fixpoint
    admitted, rejected = KD.teach(fkb, mem, facts, laws)
    oracle = TypeOracle(triples=all_isa)
    rep["symbolic"] = {"facts": len(facts), "laws_admitted": len(admitted),
                       "laws_rejected": len(rejected), "types": len(oracle.closure),
                       "entities": len(all_ents), "verbs": len(all_verbs)}

    # ── Stage 2: MEMBRANE — pooled verb constraints + predictor + questions ──────
    constraints = BrainData.learn_verb_constraints_pooled(datas, oracle, frac=0.5)
    from event_predict import EventPredictor
    predictor = EventPredictor()
    for d in datas:
        d.train_predictor(predictor)
    npred = sum(len(d.events) for d in datas)
    nq = 0
    for d in datas:
        _, q = d.learn_questions(d.entities())
        nq += q or 0
    rep["membrane"] = {"verb_constraints": len(constraints),
                       "predictor_transitions": npred, "questions": nq}

    # ── Stage 3 + 4: MORPH + DIMENSIONS ─────────────────────────────────────────
    rep["morph"] = len(morph)
    dim = {"consistent": 0, "violation": 0, "unknown": 0}
    for d in datas:
        r = d.dim_report()
        if r:
            for k in dim:
                dim[k] += r.get(k, 0)
    rep["dimensional"] = dim

    # ── Stage 5: FUZZY — C++ SOM grounding over the raw text ─────────────────────
    if do_brain:
        try:
            import brain2
            b = brain2.Brain(som_rows=24, som_cols=24, n_dims=32)
            for _ in range(2):
                for t in texts:
                    b.perceive_text(t, brain2.ErrorMode.FULL)
            rep["fuzzy_som"] = {"perceived": len(texts), "oov": b.oov_count,
                                "conflicts": b.crisp_conflicts}
        except Exception as e:
            rep["fuzzy_som"] = f"skipped ({type(e).__name__})"

    # ── Stage 6: PROBABILISTIC — owned neural LM over parse pairs ────────────────
    if do_lm:
        try:
            from neural_lm_torch import NeuralLMTorch
            lm = NeuralLMTorch(dim=256, layers=4, ctx=16, epochs=40).train(parse_pairs)
            os.makedirs("trained", exist_ok=True)
            lm.save("trained/owned_lm_data.pt")
            rep["probabilistic_lm"] = {"device": str(lm.device), "params": lm.param_count(),
                                       "vocab": len(lm.w2i)}
        except Exception as e:
            rep["probabilistic_lm"] = f"skipped ({type(e).__name__}: {e})"

    # ── Stage 7 + 8: FACULTIES (standing loop) + PERSIST ────────────────────────
    # feed the trained corpus policies into the front, run the self-improving loop, persist
    try:
        from whole_brain import WholeBrain
        wb = WholeBrain()
        for tgt, p in mem.by_target.items():           # corpus-learned laws -> faculty substrate
            wb.mem.add(p)
        for e, r, v in facts[:2000]:                   # seed associative memory
            if isinstance(v, str) and not v.replace('.', '').isdigit():
                wb.remember(e, r, v)
        loop = wb.run_loop(ticks=3)
        rep["faculties"] = {"loop_ticks": loop["ticks_run"], "persisted": loop["final"]}
    except Exception as e:
        rep["faculties"] = f"skipped ({type(e).__name__}: {e})"

    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def main():
    do_lm = "--lm" in sys.argv
    do_brain = "--brain" in sys.argv
    print("=" * 66)
    print("  train_all — the whole brain, one pass, everything together")
    print("=" * 66)
    rep = train(do_lm=do_lm, do_brain=do_brain)
    print("\n  RESULT")
    for k, v in rep.items():
        print(f"    {k:16s} {v}")
    print("\n  everything trained + persisted. next session loads it "
          "(whole_brain reads brain_store).")


if __name__ == "__main__":
    main()
