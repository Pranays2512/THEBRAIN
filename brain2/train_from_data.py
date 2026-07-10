#!/usr/bin/env python3
"""train_from_data.py — train every faculty from ONE tagged corpus (brain_data format).

Pure-python faculties run anywhere (types, morphology, symbolic knowledge, predictor). The
student LM (torch/MPS) trains under venv2 with --lm. The C++ Brain perceives the raw text
with --brain. One file -> the whole brain, membrane intact (laws verified, not assumed).

    python3 train_from_data.py data/kimi_data.txt            # symbolic + predictor + mouth
    KMP_DUPLICATE_LIB_OK=TRUE venv2/bin/python3 train_from_data.py data/kimi_data.txt --lm --brain
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from brain_data import BrainData


def train(path, do_lm=False, do_brain=False):
    d = BrainData.from_file(path)
    rep = {"corpus": d.report()}

    # 1. GROUND TYPES (fuzzy pillar's crisp base)
    oracle = d.type_oracle()
    rep["types"] = len(oracle.closure)

    # 2. MORPHOLOGY -> the mouth (child-grade -> fluent, by learning)
    rep["morph_loaded"] = d.load_morph()

    # 3. SYMBOLIC KNOWLEDGE (facts taught, laws VERIFIED before admit)
    import knowledge_distill as KD
    from means_ends import PolicyMemory
    fkb, mem = KD.SimpleKB(), PolicyMemory()
    rep["knowledge"] = d.teach_knowledge(fkb, mem)

    # 4. PREDICTOR (predictive processing: verb transitions over the event stream)
    from event_predict import EventPredictor
    predictor = EventPredictor()
    rep["predictor"] = d.train_predictor(predictor)

    # 4b. VERB CONSTRAINTS learned from data (trains the event membrane — no hand-set rules)
    vc = d.learn_verb_constraints(oracle)
    rep["verb_constraints"] = "%d verbs (e.g. %s)" % (
        len(vc), {k: v for k, v in list(vc.items())[:1]})

    # 4c. QUESTION understanding (trains QA: question templates, conjecture->verify->admit)
    _, qn = d.learn_questions(d.entities())
    rep["questions_learned"] = qn

    # 4d. DIMENSIONAL verifier over the laws (domain_features, fed by UNIT lines)
    rep["dimensional"] = d.dim_report()

    # 5. STUDENT LM (owned neural net; learns the PARSING, text -> structure) — heavy
    if do_lm:
        try:
            from core.neural.neural_lm_torch import NeuralLMTorch
            lm = NeuralLMTorch(dim=256, layers=4, ctx=16, epochs=40).train(d.parse_pairs)
            os.makedirs("trained", exist_ok=True)
            lm.save("trained/owned_lm_data.pt")
            rep["lm"] = {"backend": "%s" % lm.device, "params": lm.param_count(),
                         "vocab": len(lm.w2i), "sample": " ".join(lm.generate(seed=0))}
        except Exception as e:
            rep["lm"] = "skipped (%s: %s)" % (type(e).__name__, e)

    # 6. C++ BRAIN perceives the raw text (SOM/episodic/emotion over sentences + stories)
    if do_brain:
        try:
            import brain2
            b = brain2.Brain(som_rows=24, som_cols=24, n_dims=32)
            texts = [s for s, _ in d.events] + d.sequences
            for _ in range(2):
                for t in texts:
                    b.perceive_text(t, brain2.ErrorMode.FULL)
            rep["brain"] = {"perceived": len(texts), "oov": b.oov_count}
        except Exception as e:
            rep["brain"] = "skipped (%s)" % type(e).__name__

    return d, oracle, fkb, mem, predictor, rep


def _demo(rep, fkb, oracle, predictor):
    from mouth import say_event
    from core.events.event_parse import parse_event
    from core.events.event_form import Event, POS
    print("=== trained from data ===")
    for k, v in rep.items():
        print("  %-11s %s" % (k, v))
    # the mouth, now fluent from LEARNED morphology (irregulars from the data)
    print("\n  mouth (learned morphology — irregulars from the data):")
    for verb, agent, patient, tense in [("throw", "boy", "ball", "past"), ("drink", "girl", "milk", "past"),
                                        ("hold", "boy", "book", "past"), ("eat", "dog", "bread", "present")]:
        print("    ", say_event(Event(verb, agent, patient, tense, POS)))
    # a verified derived quantity (only admitted laws computed)
    for obj in ("box", "cart", "ball"):
        for q in ("momentum", "area", "force"):
            val, _ = fkb.ask(obj, q)
            if val is not None:
                print("    verified: %s.%s = %.4g" % (obj, q, val))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "data/kimi_data.txt"
    d, oracle, fkb, mem, predictor, rep = train(
        path, do_lm="--lm" in sys.argv, do_brain="--brain" in sys.argv)
    _demo(rep, fkb, oracle, predictor)
