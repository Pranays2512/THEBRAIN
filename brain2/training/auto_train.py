#!/usr/bin/env python3
"""auto_train.py — fully autonomous: discover ALL data, train EVERY component, persist.

The brain reads its own data, decides what each file is, and trains every subsystem
automatically. No flags, no manual file lists, no decisions — one command, everything trains.

  Phase 0  DISCOVER   scan data/ → classify tagged / raw / JSON / skip
  Phase 1  SYMBOLIC   facts + VERIFIED laws (multi-hop) + type closure + morph
  Phase 2  MEMBRANE   pooled verb constraints + event predictor + questions
  Phase 3  DIMENSIONS unit/dimension verifier over the laws
  Phase 4  PERCEPTION C++ SOM grounding over all text                    (auto-detect brain2)
  Phase 5  LANGUAGE   owned neural LM over parse pairs                   (auto-detect torch)
  Phase 6  READING    raw text → EventReader pipeline, no LLM needed
  Phase 7  FACULTIES  WholeBrain self-extend + self-check + curiosity loop
  Phase 8  PERSIST    policies + facts + concepts + semantic → brain_store

    python3 auto_train.py                    # run everything (auto-detects capabilities)
    venv2/bin/python3 auto_train.py          # same, with brain2 + torch available
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ── file format classification ───────────────────────────────────────────────

# Tags that mark a line as structured brain_data format
_TAGS = {"FACT:", "LAW:", "ISA:", "EVENT:", "MORPH:", "SEQ:", "UNIT:",
         "ASK:", "CHAIN:", "PROP:", "PART:"}
# Extensions to skip (binary, compressed, models)
_SKIP_EXT = {".zip", ".gz", ".csv", ".pt", ".so", ".pyc", ".o", ".bin", ".npy"}


def _classify_file(path):
    """Probe first 100 non-empty lines to decide: 'tagged', 'raw', or 'skip'."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _SKIP_EXT:
        return "skip"
    if ext == ".json":
        return "json"
    if ext != ".txt":
        return "skip"
    tag_count, pair_count, total = 0, 0, 0
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                total += 1
                if total > 100:
                    break
                if any(ln.startswith(t) for t in _TAGS):
                    tag_count += 1
                if "=>" in ln:
                    pair_count += 1
    except Exception:
        return "skip"
    # If ≥20% of sampled lines are tagged or pairs, treat as tagged
    if total > 0 and (tag_count + pair_count) / total >= 0.15:
        return "tagged"
    if total > 0:
        return "raw"
    return "skip"


def _clean_raw_text(path, max_lines=50000):
    """Extract clean sentences from a raw text file (may be OCR'd with noise).
    Filters out lines shorter than 3 words or lines that are mostly non-alpha."""
    sents = []
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                words = ln.split()
                if len(words) < 3:
                    continue
                # Skip lines that are mostly symbols/noise (< 50% alphabetic)
                alpha = sum(c.isalpha() or c.isspace() for c in ln)
                if alpha / max(len(ln), 1) < 0.5:
                    continue
                # Split on sentence boundaries
                for s in re.split(r'(?<=[.!?])\s+', ln):
                    s = s.strip()
                    if len(s.split()) >= 3 and any(c.isalpha() for c in s):
                        sents.append(s)
                if len(sents) >= max_lines:
                    break
    except Exception:
        pass
    return sents


def _extract_json_sentences(path, max_sents=10000):
    """Extract text/sentence fields from a JSON corpus file."""
    sents = []
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Common structures: {"data": [...]}, {"paragraphs": [...]}
            items = []
            for v in data.values():
                if isinstance(v, list):
                    items.extend(v)
                    break
            if not items:
                items = [data]
        else:
            return []
        for item in items:
            if isinstance(item, str):
                for s in re.split(r'(?<=[.!?])\s+', item):
                    s = s.strip()
                    if len(s.split()) >= 3:
                        sents.append(s)
            elif isinstance(item, dict):
                for key in ("text", "sentence", "context", "question", "answer",
                            "input", "output", "content"):
                    v = item.get(key)
                    if isinstance(v, str) and len(v.split()) >= 3:
                        sents.append(v.strip())
            if len(sents) >= max_sents:
                break
    except Exception:
        pass
    return sents[:max_sents]


# ── the autonomous trainer ───────────────────────────────────────────────────

class AutoTrainer:
    """One command, zero flags — discovers data, trains everything, persists."""

    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.report = {}
        self.t0 = time.time()

        # populated by discover()
        self.tagged_files = []
        self.raw_files = []
        self.json_files = []
        self.datas = []          # list of BrainData objects (from tagged files)
        self.raw_sents = []      # sentences from raw text + JSON
        self.all_texts = []      # everything combined (for SOM/LM)

        # populated by training phases
        self.fkb = None
        self.mem = None
        self.oracle = None
        self.predictor = None

    # ── Phase 0: Discovery & Ingestion ───────────────────────────────────────

    def discover(self):
        """Walk data/ and classify every file. Load tagged files via BrainData."""
        from training.brain_data import BrainData

        classified = {"tagged": [], "raw": [], "json": [], "skip": []}
        if not os.path.isdir(self.data_dir):
            print(f"  [!] data directory '{self.data_dir}' not found")
            self.report["discover"] = {"error": "data directory not found"}
            return

        for name in sorted(os.listdir(self.data_dir)):
            path = os.path.join(self.data_dir, name)
            if not os.path.isfile(path):
                continue
            kind = _classify_file(path)
            classified[kind].append(name)

        self.tagged_files = classified["tagged"]
        self.raw_files = classified["raw"]
        self.json_files = classified["json"]

        # Load tagged files via BrainData
        for name in self.tagged_files:
            path = os.path.join(self.data_dir, name)
            try:
                d = BrainData.from_file(path)
                d.load_morph()
                self.datas.append(d)
            except Exception as e:
                print(f"  [!] failed to load {name}: {e}")

        # Extract sentences from raw text files
        for name in self.raw_files:
            path = os.path.join(self.data_dir, name)
            sents = _clean_raw_text(path)
            self.raw_sents.extend(sents)

        # Extract sentences from JSON files
        for name in self.json_files:
            path = os.path.join(self.data_dir, name)
            sents = _extract_json_sentences(path)
            self.raw_sents.extend(sents)

        self.report["discover"] = {
            "tagged": len(self.tagged_files),
            "raw": len(self.raw_files),
            "json": len(self.json_files),
            "skipped": len(classified["skip"]),
            "brain_data_loaded": len(self.datas),
            "raw_sentences": len(self.raw_sents),
            "tagged_files": self.tagged_files,
        }
        print(f"  discovered: {len(self.tagged_files)} tagged, "
              f"{len(self.raw_files)} raw, {len(self.json_files)} JSON, "
              f"{len(classified['skip'])} skipped")
        print(f"  loaded {len(self.datas)} BrainData files, "
              f"{len(self.raw_sents)} raw sentences")

    # ── Phase 1: Symbolic Core ───────────────────────────────────────────────

    def train_symbolic(self):
        """Pool all tagged files → types, morph, facts, laws (verified to fixpoint)."""
        import sys as _sys
        old_limit = _sys.getrecursionlimit()
        _sys.setrecursionlimit(max(old_limit, 5000))   # deep ISA chains need room

        from training import knowledge_distill as KD
        from engines.store.type_oracle import TypeOracle
        from engines.reasoning.means_ends import PolicyMemory

        self.fkb, self.mem = KD.SimpleKB(), PolicyMemory()
        all_isa, all_ents, all_verbs = [], set(), set()
        facts, laws, morph = [], [], {}
        self.parse_pairs = []  # always set, even on partial failure
        seen_facts, seen_laws = set(), set()  # deduplicate across files

        for d in self.datas:
            for e, r, v in d.facts:
                key = (e.lower(), r.lower(), str(v))
                if key not in seen_facts:
                    seen_facts.add(key)
                    self.fkb.learn(e, r, v)
                    facts.append((e, r, v))
            # extract laws from each BrainData (dedup by target name)
            for law in d.laws:
                body = law[4:].strip() if law.startswith("LAW:") else law
                m = re.match(r"([a-zA-Z_]\w*)\s*=\s*(.+)", body)
                if not m:
                    continue
                law_key = m.group(1).lower()
                if law_key in seen_laws:
                    continue
                t = KD.infix_to_tree(m.group(2))
                if t is not None and any(isinstance(x, str) for x in KD._vars(t)):
                    seen_laws.add(law_key)
                    laws.append((law_key, t))
            all_isa += [(c, "isa", p) for c, p in d.isa]
            all_ents |= d.entities()
            all_verbs |= d.verbs()
            self.parse_pairs += d.parse_pairs
            morph.update(d.morph)

        # Multi-hop law admission to fixpoint (guarded against recursion)
        try:
            admitted, rejected = KD.teach(self.fkb, self.mem, facts, laws)
        except RecursionError:
            print("  [!] recursion in multi-hop teach — falling back to single-pass")
            # Single-pass: teach facts only, skip law chaining
            admitted, rejected = [], [t for t, _ in laws]

        # Type oracle from ISA closure
        self.oracle = TypeOracle(triples=all_isa)

        # Collect all text for SOM perception (always set)
        self.all_texts = []
        for d in self.datas:
            self.all_texts += [s for s, _ in d.events] + d.sequences
        self.all_texts += self.raw_sents

        self.report["symbolic"] = {
            "facts": len(facts),
            "laws_admitted": len(admitted),
            "laws_rejected": len(rejected),
            "types": len(self.oracle.closure),
            "entities": len(all_ents),
            "verbs": len(all_verbs),
            "morph": len(morph),
            "parse_pairs": len(self.parse_pairs),
        }
        _sys.setrecursionlimit(old_limit)
        print(f"  symbolic: {len(facts)} facts, {len(admitted)} laws admitted, "
              f"{len(rejected)} rejected, {len(self.oracle.closure)} types")

    # ── Phase 2: Event Membrane ──────────────────────────────────────────────

    def train_membrane(self):
        """Pool verb constraints, train predictor, learn question templates."""
        from training.brain_data import BrainData
        from faculties.event_predict import EventPredictor
        from engines.store.type_oracle import TypeOracle

        self.predictor = EventPredictor()

        # Use the oracle from Phase 1, or build a fresh one if Phase 1 failed
        oracle = self.oracle or TypeOracle()

        # Pooled verb constraints across all files
        constraints = BrainData.learn_verb_constraints_pooled(
            self.datas, oracle, frac=0.5)

        # Train predictor on event streams from each file
        npred = 0
        for d in self.datas:
            d.train_predictor(self.predictor)
            npred += len(d.events)

        # Learn question templates
        nq = 0
        for d in self.datas:
            _, q = d.learn_questions(d.entities())
            nq += q or 0

        self.report["membrane"] = {
            "verb_constraints": len(constraints),
            "predictor_transitions": npred,
            "questions_learned": nq,
        }
        print(f"  membrane: {len(constraints)} verb constraints, "
              f"{npred} predictor transitions, {nq} questions")

    # ── Phase 3: Dimensional Verification ────────────────────────────────────

    def verify_dimensions(self):
        """Run dimensional verifier over all laws using UNIT data."""
        dim = {"consistent": 0, "violation": 0, "unknown": 0}
        for d in self.datas:
            r = d.dim_report()
            if r:
                for k in dim:
                    dim[k] += r.get(k, 0)
        self.report["dimensional"] = dim
        print(f"  dimensional: {dim}")

    # ── Phase 4: C++ SOM/Perception ──────────────────────────────────────────

    def train_perception(self):
        """Auto-detect brain2, perceive all text on SOM + predictor."""
        try:
            import brain2
            b = brain2.Brain(som_rows=24, som_cols=24, n_dims=32)
            texts = self.all_texts[:100000]  # cap to prevent insane runtimes
            import sys as _sys
            _t0 = time.time()
            total_epochs = 2
            total_items = total_epochs * len(texts)
            for epoch in range(total_epochs):
                for i, t in enumerate(texts):
                    b.perceive_text(t, brain2.ErrorMode.FULL)
                    
                    # Progress bar every 500 texts
                    iter_done = (epoch * len(texts)) + i + 1
                    if iter_done % 500 == 0 or iter_done == total_items:
                        elapsed = time.time() - _t0
                        rate = elapsed / iter_done
                        rem = (total_items - iter_done) * rate
                        eta_str = f"{int(rem//60)}m {int(rem%60)}s"
                        
                        pct = 100 * iter_done / max(total_items, 1)
                        bar_len = 30
                        filled = int(bar_len * pct // 100)
                        bar = '█' * filled + '-' * (bar_len - filled)
                        _sys.stdout.write(f"\r    [SOM] |{bar}| {pct:.1f}%  Epoch {epoch+1}/{total_epochs}  ETA: {eta_str}   ")
                        _sys.stdout.flush()
            print()
            self.report["perception"] = {
                "backend": "brain2 (C++)",
                "perceived": len(texts),
                "epochs": 2,
                "oov": b.oov_count,
                "crisp_conflicts": b.crisp_conflicts,
            }
            print(f"  perception: {len(texts)} texts × 2 epochs, "
                  f"OOV={b.oov_count}")
        except ImportError:
            self.report["perception"] = {"backend": "skipped (brain2 not available)"}
            print("  perception: skipped (brain2 not available)")
        except Exception as e:
            self.report["perception"] = {"backend": f"error ({type(e).__name__}: {e})"}
            print(f"  perception: error ({e})")

    # ── Phase 5: Owned Neural LM ─────────────────────────────────────────────

    def train_language_model(self):
        """Auto-detect torch, train owned LM on parse pairs. Falls back to numpy."""
        corpus = getattr(self, 'parse_pairs', []) or []
        if not corpus:
            self.report["language_model"] = {"status": "skipped (no parse pairs)"}
            print("  language model: skipped (no parse pairs)")
            return

        # Try PyTorch MPS Transformer first
        try:
            from engines.neural.neural_lm_torch import NeuralLMTorch
            lm = NeuralLMTorch(dim=256, layers=4, ctx=16, epochs=40).train(corpus)
            os.makedirs("trained", exist_ok=True)
            lm.save("trained/owned_lm_auto.pt")
            self.report["language_model"] = {
                "backend": f"torch/{lm.device}",
                "params": lm.param_count(),
                "vocab": len(lm.w2i),
                "saved": "trained/owned_lm_auto.pt",
                "sample": " ".join(lm.generate(seed=0)),
            }
            print(f"  language model: torch/{lm.device}, "
                  f"{lm.param_count()} params, vocab={len(lm.w2i)}")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"  [!] torch LM failed ({e}), falling back to numpy")

        # Fallback to numpy NeuralLM
        try:
            from engines.neural.neural_lm import NeuralLM
            lm = NeuralLM(epochs=120).train(corpus)
            self.report["language_model"] = {
                "backend": "numpy",
                "vocab": len(lm.w2i),
                "sample": " ".join(lm.generate(seed=0)),
            }
            print(f"  language model: numpy, vocab={len(lm.w2i)}")
        except Exception as e:
            self.report["language_model"] = {"status": f"error ({type(e).__name__}: {e})"}
            print(f"  language model: error ({e})")

    # ── Phase 6: Raw Text Reading ────────────────────────────────────────────

    def read_raw_texts(self):
        """Push raw sentences through EventReader — local parsing + membrane, no LLM."""
        if not self.raw_sents:
            self.report["reading"] = {"status": "skipped (no raw text)"}
            print("  reading: skipped (no raw text)")
            return

        try:
            from faculties.reading_loop import EventReader
            from engines.store.type_oracle import TypeOracle
            from engines.events.verb_learn import VerbLearner
            from faculties.event_predict import EventPredictor

            # Use the oracle from Phase 1 (which has the full ISA closure)
            oracle = self.oracle or TypeOracle()
            entities = set()
            verbs = set()
            for d in self.datas:
                entities |= d.entities()
                verbs |= d.verbs()

            reader = EventReader(
                entities, verbs,
                type_of=oracle,
                learner=VerbLearner(oracle),
                predictor=self.predictor or EventPredictor(),
            )

            # Feed raw sentences, capped at 20k to keep runtime reasonable
            cap = min(len(self.raw_sents), 20000)
            import sys as _sys
            _t0 = time.time()
            for i, s in enumerate(self.raw_sents[:cap]):
                try:
                    reader.read(s)
                except Exception:
                    pass
                    
                # Progress bar every 50 sentences or end
                if i % 50 == 0 or i == cap - 1:
                    elapsed = time.time() - _t0
                    iter_done = i + 1
                    rate = elapsed / iter_done
                    rem = (cap - iter_done) * rate
                    eta_str = f"{int(rem//60)}m {int(rem%60)}s"
                    
                    pct = 100 * iter_done / cap
                    bar_len = 30
                    filled = int(bar_len * pct // 100)
                    bar = '█' * filled + '-' * (bar_len - filled)
                    _sys.stdout.write(f"\r    [READ] |{bar}| {pct:.1f}%  {iter_done}/{cap} sents  ETA: {eta_str}   ")
                    _sys.stdout.flush()
            print()

            # Acquire learned verbs
            learned_verbs = reader.acquire()

            self.report["reading"] = {
                "sentences_fed": cap,
                "events_admitted": reader.stats.get("admit", 0),
                "events_rejected": reader.stats.get("reject", 0),
                "events_abstained": reader.stats.get("abstain", 0),
                "nomatch": reader.stats.get("nomatch", 0),
                "verbs_learned": len(learned_verbs),
                "total_events": len(reader.events),
            }
            print(f"  reading: fed {cap} sentences → "
                  f"{reader.stats.get('admit', 0)} admitted, "
                  f"{reader.stats.get('reject', 0)} rejected, "
                  f"{reader.stats.get('abstain', 0)} abstained")
        except Exception as e:
            self.report["reading"] = {"status": f"error ({type(e).__name__}: {e})"}
            print(f"  reading: error ({e})")

    # ── Phase 7: Faculties & Self-Improvement Loop ───────────────────────────

    def train_faculties(self):
        """Wire trained knowledge into WholeBrain, run self-improving loop."""
        try:
            from faculties.whole_brain import WholeBrain

            wb = WholeBrain()

            # Inject corpus-learned policies into the faculty substrate
            if self.mem:
                for tgt, p in self.mem.by_target.items():
                    wb.mem.add(p)

            # Seed associative memory with learned facts
            if self.fkb:
                n_seeded = 0
                for (e, r), v in list(self.fkb.facts.items())[:2000]:
                    if isinstance(v, str) and not v.replace('.', '').replace('-', '').isdigit():
                        wb.remember(e, r, v)
                        n_seeded += 1

            # Run the self-improving loop (ticks=3 — conjecture→sandbox→bank each tick)
            loop = wb.run_loop(ticks=3, verbose=True)

            self.report["faculties"] = {
                "loop_ticks": loop["ticks_run"],
                "final": loop["final"],
                "policies_seeded": len(self.mem.by_target) if self.mem else 0,
            }
            print(f"  faculties: {loop['ticks_run']} ticks, "
                  f"persisted: {loop['final']}")

            # Keep the WholeBrain for the persist phase
            self._wb = wb
        except Exception as e:
            self.report["faculties"] = {"status": f"error ({type(e).__name__}: {e})"}
            print(f"  faculties: error ({e})")
            self._wb = None

    # ── Phase 8: Persist ─────────────────────────────────────────────────────

    def persist(self):
        """Save everything to brain_store/ and trained/. Write the full report."""
        persisted = {}

        # Save WholeBrain state (policies, concepts, semantic)
        if getattr(self, '_wb', None) is not None:
            try:
                persisted["whole_brain"] = self._wb.save_state()
            except Exception as e:
                persisted["whole_brain"] = f"error ({e})"

        # Write the training report
        self.report["total_seconds"] = round(time.time() - self.t0, 1)
        os.makedirs("trained", exist_ok=True)
        report_path = "trained/auto_train_report.json"
        try:
            with open(report_path, "w") as f:
                json.dump(self.report, f, indent=2, default=str)
            persisted["report"] = report_path
        except Exception as e:
            persisted["report"] = f"error ({e})"

        self.report["persist"] = persisted
        print(f"  persisted: {persisted}")

    # ── run all phases ───────────────────────────────────────────────────────

    def run(self):
        """Execute all 8 phases in dependency order."""
        phases = [
            ("Phase 0: DISCOVER", self.discover),
            ("Phase 1: SYMBOLIC", self.train_symbolic),
            ("Phase 2: MEMBRANE", self.train_membrane),
            ("Phase 3: DIMENSIONS", self.verify_dimensions),
            ("Phase 4: PERCEPTION", self.train_perception),
            ("Phase 5: LANGUAGE MODEL", self.train_language_model),
            ("Phase 6: RAW TEXT READING", self.read_raw_texts),
            ("Phase 7: FACULTIES", self.train_faculties),
            ("Phase 8: PERSIST", self.persist),
        ]
        for label, fn in phases:
            print(f"\n{'─' * 60}")
            print(f"  {label}")
            print(f"{'─' * 60}")
            t = time.time()
            try:
                fn()
            except Exception as e:
                print(f"  [!!] {label} FAILED: {type(e).__name__}: {e}")
                self.report[label] = {"FAILED": str(e)}
            dt = round(time.time() - t, 1)
            print(f"  ({dt}s)")

        return self.report


def main():
    print("=" * 66)
    print("  auto_train — the brain reads its own data, trains everything")
    print("=" * 66)

    trainer = AutoTrainer(data_dir="data/")
    report = trainer.run()

    print("\n" + "=" * 66)
    print("  RESULT SUMMARY")
    print("=" * 66)
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"\n  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2:24s} {v2}")
        else:
            print(f"  {k:24s} {v}")

    print(f"\n  total time: {report.get('total_seconds', '?')}s")
    print("  everything trained + persisted. next session loads it.")


if __name__ == "__main__":
    main()
