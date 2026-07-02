#!/usr/bin/env python3
"""
train_pipeline.py — PHASE 3 scaffold: the one unified neural training pipeline.

The insight (from the roadmap): the LM distillation, the SOM grounding, and the proposer
training are ONE pipeline — same corpus, same verified-solution signal. This is that driver.
It runs the three stages in a single pass:

  1. DISTILL  — corpus (optionally expanded by a qwen-coder teacher; falls back to given text)
  2. LM       — train the owned neural LM on the corpus (the probabilistic pillar)
  3. GROUND   — self-organize the SOM on data vectors + ground concepts (the fuzzy pillar)
  4. PROPOSER — learn synthesis-space priors from verified outcomes (the search guide)

It runs TINY here to prove the wiring end to end. The TRAINING PHASE is the same driver with
your resources: use_teacher=True (qwen-coder up), a real corpus, more epochs, a GPU. Nothing
else changes — the pipeline is already connected; only scale differs.

    python3 train_pipeline.py                    # tiny wiring proof (offline)
    venv2/bin/python3 train_pipeline.py --real   # + qwen-coder teacher + grounding on the SOM
"""

import os
# torch and brain2 both link LLVM libomp; allow the duplicate AND pin OMP to 1 thread so the
# grounding/brain stages (brain2/OpenMP) and the LM stage (torch) coexist in one process on
# macOS without a segfault. (Must be set before torch/brain2 import.)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys

import context_embed as CE
import feature_learner as FL
import synth_engine as SE
from neural_lm import NeuralLM


import re

TOPICS = ["speed", "mass", "force", "energy", "gravity", "momentum", "acceleration", "heat",
          "temperature", "light", "sound", "electricity", "magnetism", "atoms", "molecules",
          "water", "metals", "motion", "friction", "pressure", "density", "volume", "waves",
          "orbits", "planets", "stars", "electrons", "chemistry", "velocity", "power"]


def _clean_sentences(text):
    out = []
    for ln in text.split("\n"):
        ln = re.sub(r"^\s*[\d\-\*\.\)]+\s*", "", ln).strip()      # strip leading numbering/bullets
        for s in re.split(r"(?<=[.!?])\s+", ln):
            s = s.strip()
            if len(s.split()) >= 4:
                out.append(s)
    return out


class UnifiedTrainer:
    def __init__(self, corpus=None, use_teacher=False, seeds=None,
                 teacher_model="qwen3-coder:480b-cloud", lm_epochs=120,
                 lm_dim=128, lm_layers=2, lm_ctx=32):
        self.corpus = list(corpus or CE.CORPUS)
        self.use_teacher = use_teacher
        self.seeds = seeds or ("speed", "mass", "force", "energy")
        self.teacher_model = teacher_model
        self.lm_epochs, self.lm_dim, self.lm_layers, self.lm_ctx = lm_epochs, lm_dim, lm_layers, lm_ctx
        self.report = {}
        import knowledge_distill as KD
        from means_ends import PolicyMemory
        self.fkb = KD.SimpleKB()          # symbolic brain: exact taught facts
        self.mem = PolicyMemory()         # symbolic brain: policies/laws
        try:
            import corpus_scale as CS
            self.corpus = list(CS.LARGE)
        except Exception:
            pass

    # 1b. TEACH — the teacher PARSES each topic into sentence=>structure PAIRS. The symbolic
    #     brain learns the STRUCTURE (facts + VERIFIED laws); the student LM learns the PARSING
    #     itself (map a sentence to its structured form). The student's whole job is parsing —
    #     text -> the structure the symbolic core verifies — so it trains on parse pairs only.
    def stage_teach_knowledge(self):
        if not self.use_teacher:
            return
        from llm_adapter import OllamaClient, SafeClient
        import knowledge_distill as KD
        teacher = SafeClient(OllamaClient(self.teacher_model))
        prompt = ("For the topic '%s', output ONLY lines of the form  <sentence> => <structure>  where\n"
                  "structure is  FACT: object | property | number   or   LAW: quantity = expression\n"
                  "(expression uses the properties and + - * / and numbers). Use one object.\n"
                  "Example:  the box has mass 5 => FACT: box | mass | 5\n"
                  "Give 6-8 such lines, no prose.")
        parse_pairs, struct_lines = [], []
        for topic in self.seeds:
            raw = teacher.complete(prompt % topic)
            for ln in raw.splitlines():
                if "=>" not in ln:
                    continue
                left, right = ln.split("=>", 1)
                left, right = left.strip(), right.strip()
                if len(left.split()) >= 3 and (right.startswith("FACT:") or right.startswith("LAW:")):
                    parse_pairs.append("%s => %s" % (left.lower(), right))  # STUDENT: the parsing
                    struct_lines.append(right)                              # BRAIN: the structure
        # BRAIN learns the structure (facts taught, laws verified before admit)
        f, l, _ = KD.parse_teacher("\n".join(struct_lines))
        adm, rej = KD.teach(self.fkb, self.mem, f, l)
        # STUDENT learns the PARSING ONLY — its corpus is the sentence=>structure pairs
        self.corpus = parse_pairs
        self.report["parse_pairs"] = len(parse_pairs)
        self.report["facts_learned"] = len(f)
        self.report["laws_admitted"] = len(adm)
        self.report["laws_rejected"] = len(rej)
        self._taught = adm
        self._entities = {e for e, _, _ in f}

    # 1. DISTILL — expand the corpus via the teacher (qwen-coder), cleaned into sentences
    def stage_distill(self):
        if self.use_teacher:
            from llm_adapter import OllamaClient, SafeClient
            teacher = SafeClient(OllamaClient(self.teacher_model))
            for s in self.seeds:
                out = teacher.complete("Write four short simple factual sentences about %s. "
                                       "Plain sentences, no numbering." % s)
                self.corpus += _clean_sentences(out)
        self.report["corpus_sentences"] = len(self.corpus)

    # 2. LM — train the owned probabilistic pillar on the corpus. Prefer the PyTorch-MPS
    #    Transformer (real, Mac-GPU) when torch is installed; else the numpy proof model.
    def stage_lm(self):
        try:
            from neural_lm_torch import NeuralLMTorch
            self.lm = NeuralLMTorch(dim=self.lm_dim, layers=self.lm_layers,
                                    ctx=self.lm_ctx, epochs=self.lm_epochs).train(self.corpus)
            self.report["lm_backend"] = "torch/%s (%d params)" % (self.lm.device, self.lm.param_count())
        except ImportError:
            self.lm = NeuralLM(epochs=self.lm_epochs).train(self.corpus)
            self.report["lm_backend"] = "numpy (install torch for Mac-GPU + scale)"
        self.report["lm_vocab"] = len(self.lm.w2i)
        self.report["lm_sample"] = " ".join(self.lm.generate(seed=0))
        # PERSIST the trained student so the run produces a reusable artifact (not in-memory only)
        if hasattr(self.lm, "save"):
            os.makedirs("trained", exist_ok=True)
            self.lm.save("trained/owned_lm.pt")
            self.report["lm_saved"] = "trained/owned_lm.pt"

    # 2b. BRAIN — train the C++ Brain's own neural (SOM + predictor) on the corpus via the
    #     perceive loop. This is the BRAIN half; stage_lm trains the STUDENT (owned LM). Both
    #     learn from the same corpus in the one run.
    def stage_train_brain(self, epochs=3):
        try:
            import brain2
            b = brain2.Brain(som_rows=16, som_cols=16, n_dims=32)
            for _ in range(epochs):
                for line in self.corpus:
                    b.perceive_text(line)
            self.brain = b
            self.report["brain_trained"] = "SOM+predictor, %d lines x%d epochs" % (len(self.corpus), epochs)
            self.report["brain_oov"] = b.oov_count
        except Exception as e:
            self.report["brain_trained"] = "skipped (%s)" % type(e).__name__

    # 3. GROUND — SOM self-organizes on data vectors + grounds concepts (fuzzy pillar)
    def stage_ground(self):
        try:
            import grounding as G
            import brain2
            train, test = G.make_data()
            som = brain2.SOM(G.ROWS, G.COLS, G.D, init_lr=0.3)
            for _ in range(6):
                for v, _ in train:
                    som.update(v, som.find_bmu(v), 1.0)
            cents = G.ground(som, [(v, k) for v, k in train][:G.K * 5])
            acc = sum(G.recognize(som, cents, v) == k for v, k in test) / len(test)
            self.report["grounding_acc"] = round(acc, 3)
        except Exception as e:
            self.report["grounding_acc"] = "skipped (%s)" % type(e).__name__

    # 4. PROPOSER — learn space priors from verified synthesis outcomes (search guide)
    def stage_proposer(self, tasks):
        prop = FL.LearnedProposer()
        att = 0
        for _name, kind, oracle, ins in tasks:
            ex = SE._ex(kind, oracle, ins)
            _, _, a = prop.solve(ex, kind, oracle)
            att += a
        self.report["proposer_attempts"] = att
        self.report["proposer_learned"] = len(prop.proto)

    def stage_verify_learned(self):
        """Prove the brain learned NEW verified knowledge: read back a taught derived quantity
        (teach stored the verified value directly — no recursive solve needed)."""
        shown = []
        for target in getattr(self, "_taught", [])[:3]:
            for e in getattr(self, "_entities", set()):
                v, _ = self.fkb.ask(e, target)
                if v is not None:
                    shown.append("%s.%s=%.4g" % (e, target, v)); break
        self.report["verified_answers"] = shown or ["(none computed)"]

    def run(self, tasks):
        if self.use_teacher:
            self.stage_teach_knowledge()   # teacher parses -> brain(structure) + student(parsing)
        else:
            self.stage_distill()
        self.report["corpus_sentences"] = len(self.corpus)
        self.stage_lm()            # STUDENT: learns the PARSING (text -> structure)
        self.stage_train_brain()   # BRAIN: the C++ SOM + predictor
        self.stage_ground()
        self.stage_proposer(tasks)
        self.stage_verify_learned()
        return self.report


def _demo(real=False):
    import math
    msub = lambda a: max(sum(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a)))
    tasks = [("factorial", "int1", math.factorial, [0, 1, 4, 5, 6]),
             ("gcd", "int2", math.gcd, [(12, 8), (48, 36), (7, 5)]),
             ("subarray", "list", msub, [[1, -2, 3], [-1, -2], [2, 3, 4]])]
    print("=== train_pipeline — %s run (brain + student, one pass) ===\n"
          % ("REAL (qwen-coder teacher, scaled)" if real else "tiny wiring proof"))
    if real:
        t = UnifiedTrainer(use_teacher=True, seeds=TOPICS,
                           lm_epochs=250, lm_dim=256, lm_layers=4, lm_ctx=48)
    else:
        t = UnifiedTrainer(use_teacher=False)
    rep = t.run(tasks)
    for k in ["parse_pairs", "facts_learned", "laws_admitted", "laws_rejected",
              "verified_answers", "corpus_sentences", "lm_backend", "lm_vocab", "lm_sample",
              "brain_trained", "grounding_acc", "proposer_attempts"]:
        if k in rep:
            print("  %-18s %s" % (k, rep.get(k)))
    print("\n  STUDENT (owned LM) + BRAIN (SOM/predictor) + grounding + proposer all trained from")
    print("  ONE corpus + verified signal in ONE pass. This is the")
    print("  wiring proof. TRAINING PHASE = same driver, your resources: --real (qwen-coder),")
    print("  a large corpus, more epochs, a GPU. The pipeline is connected; only scale changes.")


if __name__ == "__main__":
    _demo(real="--real" in sys.argv)
