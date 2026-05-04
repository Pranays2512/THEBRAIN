"""
train.py — Main training loop for Brain v2

Phases:
  1. Math sequences    — teaches Predictor sequence structure + arithmetic
  2. ConceptNet        — teaches causal/relational understanding
  3. Curiosity loop    — brain explores via imagination, revisits high-error regions

Usage:
  python train.py --phase 1 --steps 100000
  python train.py --phase 2 --steps 500000 --conceptnet path/to/cn.csv.gz
  python train.py --phase all --steps 1000000

Brain config (recommended for real training):
  som_rows=16, som_cols=16, n_dims=32, hidden_dim=512, wm_capacity=12
"""

import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import brain2
from concept_encoder import ConceptEncoder
from math_sequences import MathSequenceGenerator

# ── Config ────────────────────────────────────────────────────────────

DEFAULT_BRAIN_CONFIG = dict(
    som_rows   = 16,
    som_cols   = 16,
    n_dims     = 32,
    hidden_dim = 512,
    wm_capacity= 12,
    episodic_max = 5000,
    self_neurons = 32,
    seed       = 42,
)

SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

# ── Helpers ───────────────────────────────────────────────────────────

def make_brain(cfg: dict = DEFAULT_BRAIN_CONFIG) -> brain2.Brain:
    return brain2.Brain(**cfg)

def log(step: int, loss: float, extra: str = "", interval: int = 1000):
    if step % interval == 0:
        print(f"  step={step:>8,}  err={loss:.4f}  {extra}")

def save_checkpoint(brain: brain2.Brain, name: str):
    os.makedirs(SAVE_PATH, exist_ok=True)
    brain.predictor.save(os.path.join(SAVE_PATH, f"{name}_predictor.bin"))
    brain.language.save(os.path.join(SAVE_PATH, f"{name}_language.bin"))
    brain.som.save(os.path.join(SAVE_PATH, f"{name}_som.bin"))
    brain.episodic.save(os.path.join(SAVE_PATH, f"{name}_episodic.bin"))
    brain.emotion.save(os.path.join(SAVE_PATH, f"{name}_emotion.bin"))
    brain.self_model.save(os.path.join(SAVE_PATH, f"{name}_self.bin"))
    brain.symbolic.save(os.path.join(SAVE_PATH, f"{name}_symbolic.bin"))
    print(f"  Saved checkpoint: {name}")

# ── Phase 1: Math sequences ───────────────────────────────────────────

def train_math(brain: brain2.Brain, n_steps: int, log_interval: int = 1000):
    """
    Feed arithmetic/causal sequences to brain.
    Brain learns: sequence structure, number patterns, cause-effect chains.
    """
    print(f"\n[Phase 1] Math + Causal sequences  ({n_steps:,} steps)")
    print("=" * 60)

    n_dims  = brain.n_dims
    gen     = MathSequenceGenerator(n_dims=n_dims)
    seq_gen = gen.all_types()

    total_error = 0.0
    step = 0

    while step < n_steps:
        seq = next(seq_gen)
        encoded = gen.encode_seq(seq)  # List[(vec, word)]

        # Reset at sequence boundary — clears hidden state + prev_act buffer
        brain.reset_sequence()

        seq_error = 0.0
        for vec, word in encoded:
            result = brain.perceive(vec)
            seq_error += result.prediction_error
            if word:
                brain.hear(word)

            step += 1
            if step >= n_steps:
                break

        total_error += seq_error / max(len(encoded), 1)

        if step % log_interval == 0:
            avg_err = total_error / max(step // max(len(encoded), 1), 1)
            print(f"  step={step:>8,}  avg_err={avg_err:.4f}"
                  f"  emotion=({brain.emotion.valence:+.2f}, {brain.emotion.arousal:.2f})"
                  f"  episodes={brain.episodic.episode_count}"
                  f"  vocab={brain.language.vocab_size}")

    print(f"  Phase 1 complete. Final avg_err={total_error/max(step,1):.4f}")
    return total_error / max(step, 1)

# ── Phase 2: ConceptNet ───────────────────────────────────────────────

def train_conceptnet(brain: brain2.Brain, n_steps: int,
                     cn_path: str, log_interval: int = 1000):
    """
    Feed ConceptNet causal/relational triples to brain.
    Brain learns: causality, hierarchy, properties, actions.
    """
    print(f"\n[Phase 2] ConceptNet  ({n_steps:,} steps)")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(__file__))
    from conceptnet_loader import ConceptNetLoader

    n_dims = brain.n_dims
    loader = ConceptNetLoader(n_dims=n_dims)

    total_error = 0.0
    step        = 0

    # Cycle through ConceptNet multiple times if needed
    while step < n_steps:
        for seq in loader.sequences(cn_path, max_seqs=n_steps * 2):
            if step >= n_steps:
                break

            enc = ConceptEncoder(n_dims)
            encoded = [(enc.encode(concept), word) for concept, word in seq]

            brain.reset_sequence()
            seq_error = 0.0

            for vec, word in encoded:
                result = brain.perceive(vec)
                seq_error += result.prediction_error
                if word:
                    brain.hear(word)
                step += 1
                if step >= n_steps:
                    break

            total_error += seq_error / max(len(encoded), 1)

            if step % log_interval == 0:
                avg = total_error / max(step // max(len(encoded), 1), 1)
                print(f"  step={step:>8,}  avg_err={avg:.4f}"
                      f"  vocab={brain.language.vocab_size}"
                      f"  episodes={brain.episodic.episode_count}")

    print(f"  Phase 2 complete.")
    return total_error / max(step, 1)

# ── Phase 3: Curiosity-driven exploration ────────────────────────────

def train_curiosity(brain: brain2.Brain, n_steps: int,
                    log_interval: int = 500):
    """
    Brain generates its own training data via imagination.
    High prediction-error regions = unexplored = explore more.
    Self-generated thought sequences reinforce learned patterns.
    """
    print(f"\n[Phase 3] Curiosity / Self-play  ({n_steps:,} steps)")
    print("=" * 60)

    step        = 0
    total_error = 0.0

    while step < n_steps:
        # Think: generate concept sequence via inner speech
        thought = brain.think(steps=8)

        # Feed thought concepts back as perception (self-generated experience)
        brain.predictor.reset()
        for concept_vec in thought.concepts:
            vec = np.array(concept_vec, dtype=np.float32)
            if len(vec) == brain.n_dims:
                result = brain.perceive(vec)
                total_error += result.prediction_error
                step += 1

        # Periodically dream — consolidate episodic + explore
        if step % 5000 == 0 and step > 0:
            brain.dream(n_dreams=10, steps_per_dream=12)
            print(f"  [dream]  step={step:>8,}  episodes={brain.episodic.episode_count}"
                  f"  drift={brain.self_model.drift(brain2.InternalState()):.3f}")

        if step % log_interval == 0:
            avg = total_error / max(step, 1)
            thought_words = thought.words[:5] if thought.words else []
            print(f"  step={step:>8,}  avg_err={avg:.4f}"
                  f"  words={thought_words}"
                  f"  coh={thought.coherence:.3f}")

    print(f"  Phase 3 complete.")
    return total_error / max(step, 1)

# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(brain: brain2.Brain, n_dims: int):
    """Quick sanity checks after training."""
    print("\n[Eval] Post-training sanity checks")
    print("=" * 60)

    enc = ConceptEncoder(n_dims)

    tests = [
        # (description, input_seq, expected_last_word)
        ("2 + 3 = ?",    [("2", "2"), ("+", "plus"), ("3", "3"), ("=", "equals")], "5"),
        ("fire causes ?", [("fire", "fire"), ("causes", "causes")], "heat"),
        ("dog isa ?",     [("dog", "dog"), ("isa", "isa")], "animal"),
        ("4 > 2 = ?",    [("4", "4"), (">", "greater"), ("2", "2"), ("=", "equals")], "true"),
    ]

    passed = 0
    for desc, seq, expected in tests:
        brain.reset_sequence()
        brain.predictor.set_offline(True)
        last_vec = None
        prev_act = None
        for concept, word in seq:
            vec     = enc.encode(concept)
            act_map = np.array(brain.som.activation_map(vec))
            if prev_act is not None:
                last_vec = np.array(brain.predictor.step(prev_act))
            prev_act = act_map
        # Final step: feed last known, get prediction of next
        if prev_act is not None:
            last_vec = np.array(brain.predictor.step(prev_act))
        brain.predictor.set_offline(False)

        if last_vec is not None and brain.language.vocab_size > 0:
            predicted = brain.language.best_word(last_vec)
            ok = predicted == expected
            if ok:
                passed += 1
            status = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
            print(f"  [{status}] {desc}  → predicted='{predicted}'  expected='{expected}'")
        else:
            print(f"  [SKIP] {desc} (no vocab yet)")

    print(f"\n  Eval: {passed}/{len(tests)} correct")
    return passed

# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Brain v2 Training")
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="1")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--conceptnet", type=str, default=None,
                        help="Path to conceptnet-assertions-5.7.0.csv.gz")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Save/load checkpoint name")
    parser.add_argument("--som-size", type=int, default=16,
                        help="SOM rows=cols (default 16 → 16x16=256 neurons)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-dims", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()

    cfg = dict(
        som_rows    = args.som_size,
        som_cols    = args.som_size,
        n_dims      = args.n_dims,
        hidden_dim  = args.hidden,
        wm_capacity = 12,
        episodic_max= 5000,
        self_neurons= 32,
        seed        = 42,
    )

    print(f"\nBrain v2 Training")
    print(f"  SOM:      {cfg['som_rows']}x{cfg['som_cols']} = {cfg['som_rows']*cfg['som_cols']} neurons")
    print(f"  n_dims:   {cfg['n_dims']}")
    print(f"  hidden:   {cfg['hidden_dim']}")
    n_neurons = cfg['som_rows'] * cfg['som_cols']
    lstm1 = 4*n_neurons*cfg['hidden_dim'] + 4*cfg['hidden_dim']**2 + 4*cfg['hidden_dim']
    lstm2 = 8*cfg['hidden_dim']**2 + 4*cfg['hidden_dim']
    out   = cfg['hidden_dim']*n_neurons + n_neurons
    total = lstm1 + lstm2 + out
    print(f"  Params:   ~{total/1e6:.1f}M")
    print(f"  Phase:    {args.phase}  ({args.steps:,} steps)")
    print()

    t0    = time.time()
    brain = make_brain(cfg)

    if args.phase in ("1", "all"):
        train_math(brain, args.steps, args.log_interval)
        if args.checkpoint:
            save_checkpoint(brain, args.checkpoint + "_p1")

    if args.phase in ("2", "all"):
        if not args.conceptnet:
            print("Phase 2 needs --conceptnet path. Skipping.")
            print("Download: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz")
        else:
            train_conceptnet(brain, args.steps, args.conceptnet, args.log_interval)
            if args.checkpoint:
                save_checkpoint(brain, args.checkpoint + "_p2")

    if args.phase in ("3", "all"):
        train_curiosity(brain, args.steps // 5, args.log_interval // 2)
        if args.checkpoint:
            save_checkpoint(brain, args.checkpoint + "_p3")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s  ({args.steps/elapsed:.0f} steps/sec)")

    evaluate(brain, cfg['n_dims'])

    if args.checkpoint:
        save_checkpoint(brain, args.checkpoint + "_final")

if __name__ == "__main__":
    main()
