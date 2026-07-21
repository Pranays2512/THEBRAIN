#!/usr/bin/env python3
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.grounding import context_embed as CE
from faculties import feature_learner as FL
from engines.synthesis import synth_engine as SE
from engines.neural.neural_lm import NeuralLM

import json
import re

TOPICS = [
    "Arrays", "Linked Lists", "Hash Maps", "Sets", "Trees", "Graphs", 
    "Prefix Sums", "Two Pointers", "Sliding Window", "Depth-First Search", 
    "Greedy Algorithms", "Dynamic Programming", "Dijkstra's Algorithm", 
    "Backtracking", "Tries", "Topological Sort"
]

def _clean_sentences(text):
    out = []
    for ln in text.split("\n"):
        ln = re.sub(r"^\s*[\d\-\*\.\)]+\s*", "", ln).strip()
        for s in re.split(r"(?<=[.!?])\s+", ln):
            s = s.strip()
            if len(s.split()) >= 4:
                out.append(s)
    return out

class UnifiedTrainer:
    def __init__(self, corpus=None, use_teacher=False, seeds=None,
                 teacher_model="qwen3-coder:480b-cloud", lm_epochs=120,
                 lm_dim=128, lm_layers=2, lm_ctx=32, refresh=False,
                 cache_path=os.path.join("trained", "teacher_cache_dsa.json")):
        self.corpus = list(corpus or CE.CORPUS)
        self.use_teacher = use_teacher
        self.seeds = seeds or TOPICS
        self.teacher_model = teacher_model
        self.lm_epochs, self.lm_dim, self.lm_layers, self.lm_ctx = lm_epochs, lm_dim, lm_layers, lm_ctx
        self.refresh = refresh
        self.cache_path = cache_path
        self.report = {}
        from training import knowledge_distill as KD
        from engines.reasoning.means_ends import PolicyMemory
        self.fkb = KD.SimpleKB()
        self.mem = PolicyMemory()

    _PROMPT = ("For the data structures and algorithms topic '%s', output ONLY lines of the form  <sentence> => <structure>  where\n"
               "structure is  FACT: algorithm_name | optimization_op | runtime_complexity\n"
               "Example: dynamic programming uses memoization to cache subproblems => FACT: DynamicProgramming | Memoize | O(N)\n"
               "Give 12-15 such lines. Vary the phrasing across lines. No prose.")

    def _teach_topic(self, teacher, topic):
        pairs, structs = [], []
        for ln in teacher.complete(self._PROMPT % topic).splitlines():
            if "=>" not in ln:
                continue
            left, right = (x.strip() for x in ln.split("=>", 1))
            if len(left.split()) >= 3 and (right.startswith("FACT:") or right.startswith("LAW:")):
                pairs.append("%s => %s" % (left.lower(), right))
                structs.append(right)
        return {"pairs": pairs, "structs": structs}

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as fh:
                return json.load(fh)
        return {}

    def stage_teach_knowledge(self):
        from training import knowledge_distill as KD
        cache = self._load_cache()
        teacher = None
        need = self.refresh or any(t not in cache for t in self.seeds)
        if need and self.use_teacher:
            from adapters.llm_adapter import OllamaClient, SafeClient
            teacher = SafeClient(OllamaClient(self.teacher_model))
        parse_pairs, struct_lines = [], []
        for topic in self.seeds:
            if teacher is not None and (self.refresh or topic not in cache):
                cache[topic] = self._teach_topic(teacher, topic)
                os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
                with open(self.cache_path, "w") as fh:
                    json.dump(cache, fh, indent=1)
            entry = cache.get(topic)
            if not entry:
                continue
            parse_pairs += entry["pairs"]
            struct_lines += entry["structs"]
        self.report["teacher_calls"] = 0 if teacher is None else sum(
            1 for t in self.seeds if self.refresh or t in cache)
        self.report["cache_topics"] = len([t for t in self.seeds if t in cache])
        f, l, _ = KD.parse_teacher("\n".join(struct_lines))
        adm, rej = KD.teach(self.fkb, self.mem, f, l)
        self.corpus = parse_pairs
        self.report["parse_pairs"] = len(parse_pairs)
        self.report["facts_learned"] = len(f)
        self.report["laws_admitted"] = len(adm)
        self.report["laws_rejected"] = len(rej)

    def stage_distill(self):
        if self.use_teacher:
            from adapters.llm_adapter import OllamaClient, SafeClient
            teacher = SafeClient(OllamaClient(self.teacher_model))
            for s in self.seeds:
                out = teacher.complete("Write four short simple factual sentences about %s. Plain sentences, no numbering." % s)
                self.corpus += _clean_sentences(out)
        self.report["corpus_sentences"] = len(self.corpus)

    def stage_lm(self):
        try:
            from engines.neural.neural_lm_torch import NeuralLMTorch
            self.lm = NeuralLMTorch(dim=self.lm_dim, layers=self.lm_layers,
                                    ctx=self.lm_ctx, epochs=self.lm_epochs).train(self.corpus)
            self.report["lm_backend"] = "torch/%s (%d params)" % (self.lm.device, self.lm.param_count())
        except ImportError:
            self.lm = NeuralLM(epochs=self.lm_epochs).train(self.corpus)
            self.report["lm_backend"] = "numpy"
        self.report["lm_vocab"] = len(self.lm.w2i)
        if hasattr(self.lm, "generate"):
            try:
                self.report["lm_sample"] = " ".join(self.lm.generate(seed=0))
            except:
                pass
        if hasattr(self.lm, "save"):
            os.makedirs("trained", exist_ok=True)
            self.lm.save("trained/owned_lm.pt")
            self.report["lm_saved"] = "trained/owned_lm.pt"

    def stage_train_brain(self, epochs=3):
        import glob
        chapter_files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "dsa_book_chapters", "*.txt"))
        try:
            import brain2
            b = brain2.Brain(som_rows=16, som_cols=16, n_dims=32)
            lines_read = 0
            for _ in range(epochs):
                for line in self.corpus:
                    b.perceive_text(line)
                    lines_read += 1
                for filepath in chapter_files:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            clean_line = line.strip()
                            if len(clean_line) > 5:
                                b.perceive_text(clean_line)
                                lines_read += 1
            self.brain = b
            self.report["brain_trained"] = "SOM+predictor, %d lines x%d epochs" % (lines_read // epochs, epochs)
            self.report["brain_oov"] = b.oov_count
        except Exception as e:
            self.report["brain_trained"] = "skipped (%s)" % type(e).__name__

    def stage_ground(self):
        try:
            from engines.grounding import grounding as G
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
        if self.use_teacher or os.path.exists(self.cache_path):
            self.stage_teach_knowledge()
        else:
            self.stage_distill()
        self.report["corpus_sentences"] = len(self.corpus)
        self.stage_lm()
        self.stage_train_brain()
        self.stage_ground()
        self.stage_proposer(tasks)
        return self.report

def _demo(real=False):
    import math
    msub = lambda a: max(sum(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a)))
    tasks = [("factorial", "int1", math.factorial, [0, 1, 4, 5, 6]),
             ("gcd", "int2", math.gcd, [(12, 8), (48, 36), (7, 5)]),
             ("subarray", "list", msub, [[1, -2, 3], [-1, -2], [2, 3, 4]])]
    print("=== train_dsa_pipeline — %s run ===" % ("REAL (qwen-coder teacher, scaled)" if real else "tiny wiring proof"))
    if real:
        cfg = dict(lm_dim=int(os.environ.get("LM_DIM", 384)),
                   lm_layers=int(os.environ.get("LM_LAYERS", 6)),
                   lm_ctx=int(os.environ.get("LM_CTX", 64)),
                   lm_epochs=int(os.environ.get("LM_EPOCHS", 300)))
        t = UnifiedTrainer(use_teacher=True, seeds=TOPICS, refresh="--refresh" in sys.argv, **cfg)
    else:
        # Use our mock cache for the tiny proof
        t = UnifiedTrainer(use_teacher=False, seeds=["Dynamic Programming", "Dijkstra's Algorithm"], refresh=False)
        
    rep = t.run(tasks)
    print("\n=======================================================")
    print("  DSA NEURO-SYMBOLIC PIPELINE RESULTS")
    print("=======================================================")
    for k in ["cache_topics", "teacher_calls", "parse_pairs", "facts_learned", "laws_admitted",
              "laws_rejected", "verified_answers", "corpus_sentences", "lm_backend", "lm_vocab",
              "lm_sample", "brain_trained", "brain_oov", "grounding_acc", "proposer_attempts"]:
        if k in rep:
            print("  %-18s %s" % (k, rep.get(k)))
    print("=======================================================\n")

def _cache_only():
    t = UnifiedTrainer(use_teacher=True, seeds=TOPICS, refresh="--refresh" in sys.argv)
    t.stage_teach_knowledge()
    print("cache: %d/%d topics | facts=%s laws=%s -> %s" % (
        t.report.get("cache_topics", 0), len(TOPICS), t.report.get("facts_learned"),
        t.report.get("laws_admitted"), t.cache_path))

if __name__ == "__main__":
    if "--cache-only" in sys.argv:
        _cache_only()
    else:
        _demo(real="--real" in sys.argv)
