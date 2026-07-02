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

import sys

import context_embed as CE
import feature_learner as FL
import synth_engine as SE
from neural_lm import NeuralLM


class UnifiedTrainer:
    def __init__(self, corpus=None, use_teacher=False,
                 teacher_model="qwen3-coder:480b-cloud", lm_epochs=120):
        self.corpus = list(corpus or CE.CORPUS)
        self.use_teacher = use_teacher
        self.teacher_model = teacher_model
        self.lm_epochs = lm_epochs
        self.report = {}
        try:
            import corpus_scale as CS
            self.corpus = list(CS.LARGE)
        except Exception:
            pass

    # 1. DISTILL — expand corpus via the teacher if available (scale hook)
    def stage_distill(self, seeds=("speed", "mass", "force", "energy")):
        if self.use_teacher:
            from llm_adapter import OllamaClient, SafeClient
            teacher = SafeClient(OllamaClient(self.teacher_model))
            for s in seeds:
                out = teacher.complete("Write three short factual sentences about %s in physics." % s)
                self.corpus += [ln.strip() for ln in out.splitlines() if ln.strip()]
        self.report["corpus_sentences"] = len(self.corpus)

    # 2. LM — train the owned probabilistic pillar on the corpus
    def stage_lm(self):
        self.lm = NeuralLM(epochs=self.lm_epochs).train(self.corpus)
        self.report["lm_vocab"] = len(self.lm.w2i)
        self.report["lm_sample"] = " ".join(self.lm.generate(seed=0))

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

    def run(self, tasks):
        self.stage_distill()
        self.stage_lm()
        self.stage_ground()
        self.stage_proposer(tasks)
        return self.report


def _demo(real=False):
    import math
    msub = lambda a: max(sum(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a)))
    tasks = [("factorial", "int1", math.factorial, [0, 1, 4, 5, 6]),
             ("gcd", "int2", math.gcd, [(12, 8), (48, 36), (7, 5)]),
             ("subarray", "list", msub, [[1, -2, 3], [-1, -2], [2, 3, 4]])]
    print("=== train_pipeline — PHASE 3 scaffold (one run: LM + grounding + proposer) ===\n")
    t = UnifiedTrainer(use_teacher=real)
    rep = t.run(tasks)
    for k in ["corpus_sentences", "lm_vocab", "lm_sample", "grounding_acc",
              "proposer_attempts", "proposer_learned"]:
        print("  %-18s %s" % (k, rep.get(k)))
    print("\n  All three trainings ran from ONE corpus + verified signal in ONE pass. This is the")
    print("  wiring proof. TRAINING PHASE = same driver, your resources: --real (qwen-coder),")
    print("  a large corpus, more epochs, a GPU. The pipeline is connected; only scale changes.")


if __name__ == "__main__":
    _demo(real="--real" in sys.argv)
