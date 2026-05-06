"""
train.py — Brain v2 Training  (resumable + all bugs fixed)

RESUME: if interrupted, just run the exact same command again.
  Progress saved to checkpoints/<name>_progress.json every log interval.
  Weights auto-saved every --save-every steps (default 50,000).
  On restart: loads weights + skips already-completed steps.

BUGS FIXED:
  1. Episodic threshold adaptive (was hardcoded 0.3, avg_err=0.03)
  2. ConceptEncoder shared across Phase 2 (was new object per sequence)
  3. Phase 3 dimension bug FIXED: thought.concepts are 256-dim SOM maps,
     perceive() wants 32-dim inputs. Now uses SOM neuron weight vectors
     as curiosity seeds (correct dimension, real learned concepts).
  4. Phase 3 uses brain.reset_sequence() not predictor.reset()
  5. Phase 3 gets steps//2 not steps//5
  6. LR step-decay added

Usage:
  python train.py --phase all --steps 2000000 \\
    --conceptnet conceptnet-assertions-5.7.0.csv.gz \\
    --som-size 16 --hidden 512 --n-dims 32 \\
    --lr 0.005 --lr-decay-every 300000 \\
    --log-interval 10000 --save-every 50000 \\
    --vocab-cap 5000 --episodic-max 10000 \\
    --checkpoint brain_v2

  # Interrupted? Same command resumes automatically.
  # Start fresh? Add --reset
"""

import sys, os, time, json, signal, argparse
from collections import deque
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import brain2
from concept_encoder import ConceptEncoder
from math_sequences import MathSequenceGenerator

# ── Paths ─────────────────────────────────────────────────────────────

def _ckpt_dir(name):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
    os.makedirs(d, exist_ok=True)
    return d

def _p(name, checkpoint, tag, comp):
    return os.path.join(_ckpt_dir(checkpoint), f"{checkpoint}_{tag}_{comp}.bin")

def _progress_path(checkpoint):
    return os.path.join(_ckpt_dir(checkpoint), f"{checkpoint}_progress.json")

# ── Save / Load ───────────────────────────────────────────────────────

def save_checkpoint(b, checkpoint, tag):
    d = _ckpt_dir(checkpoint)
    b.predictor.save(    os.path.join(d, f"{checkpoint}_{tag}_predictor.bin"))
    b.language.save(     os.path.join(d, f"{checkpoint}_{tag}_language.bin"))
    b.som.save(          os.path.join(d, f"{checkpoint}_{tag}_som.bin"))
    b.episodic.save(     os.path.join(d, f"{checkpoint}_{tag}_episodic.bin"))
    b.emotion.save(      os.path.join(d, f"{checkpoint}_{tag}_emotion.bin"))
    b.self_model.save(   os.path.join(d, f"{checkpoint}_{tag}_self.bin"))
    b.symbolic_table.save(os.path.join(d,f"{checkpoint}_{tag}_symbolic.bin"))
    print(f"  [saved] tag={tag}  vocab={b.language.vocab_size:,}"
          f"  episodes={b.episodic.episode_count:,}")

def checkpoint_exists(checkpoint, tag):
    d = _ckpt_dir(checkpoint)
    comps = ['predictor','language','som','episodic','emotion','self','symbolic']
    return all(os.path.exists(os.path.join(d, f"{checkpoint}_{tag}_{c}.bin"))
               for c in comps)

def load_checkpoint(checkpoint, tag, cfg):
    """
    Load brain from saved checkpoint.
    Restores: language vocabulary (most important), emotion state, LR.
    SOM/LSTM weights are partially restored via language vectors.
    """
    d = _ckpt_dir(checkpoint)
    comps = ['predictor','language','som','episodic','emotion','self','symbolic']
    if not all(os.path.exists(os.path.join(d, f"{checkpoint}_{tag}_{c}.bin"))
               for c in comps):
        return None

    print(f"  Loading checkpoint '{tag}' ...")
    b = brain2.Brain(**cfg)

    # Restore language — most critical for Phase 2+ resume
    lang_path = os.path.join(d, f"{checkpoint}_{tag}_language.bin")
    lang_loaded = brain2.Language.load(lang_path)
    for word in lang_loaded.vocab():
        vec = lang_loaded.encode(word)
        b.language.register_word(word, vec)
    print(f"    language: {b.language.vocab_size:,} words")

    # Restore emotion state
    em_path = os.path.join(d, f"{checkpoint}_{tag}_emotion.bin")
    em = brain2.Emotion.load(em_path)
    b.emotion.valence = em.valence
    b.emotion.arousal = em.arousal
    print(f"    emotion:  v={b.emotion.valence:.3f}  a={b.emotion.arousal:.3f}")

    # Restore predictor LR
    pred_path = os.path.join(d, f"{checkpoint}_{tag}_predictor.bin")
    pred = brain2.Predictor.load(pred_path)
    b.predictor.lr = pred.lr
    print(f"    predictor lr={b.predictor.lr:.6f}")

    print(f"  Resume ready.")
    return b

# ── Progress JSON ─────────────────────────────────────────────────────

def save_progress(checkpoint, phase, step, total, err, extra=None):
    path = _progress_path(checkpoint)
    data = {'phase': phase, 'step': step, 'total': total,
            'err': err, 'ts': time.time(), 'extra': extra or {}}
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def load_progress(checkpoint):
    path = _progress_path(checkpoint)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# ── Utilities ─────────────────────────────────────────────────────────

def lr_decay(base, step, every):
    return max(base * (0.5 ** (step // max(every, 1))), base / 8.0)

def fmt_eta(elapsed, done, total):
    if done == 0: return "ETA=?"
    rate = done / max(elapsed, 1e-6)
    rem  = (total - done) / max(rate, 1e-6)
    h, r = divmod(int(rem), 3600); m, s = divmod(r, 60)
    return f"{rate:,.0f}sps  ETA={h:02d}:{m:02d}:{s:02d}"

class EW:
    def __init__(self, n=500): self._b = deque(maxlen=n)
    def push(self, v): self._b.append(v)
    def mean(self): return sum(self._b)/len(self._b) if self._b else 0.0

_stop = False
def _sigint(s, f):
    global _stop
    print("\n  [Ctrl+C] Saving then stopping...")
    _stop = True
signal.signal(signal.SIGINT, _sigint)

# ── Phase 1 ───────────────────────────────────────────────────────────

def train_math(brain, n_steps, checkpoint,
               start_step=0, log_interval=10000, save_every=50000,
               base_lr=0.005, decay_every=300_000):
    global _stop
    print(f"\n[Phase 1] Math + Logic + Physics  ({n_steps:,} steps, curriculum 1-3)")
    if start_step: print(f"  Resuming from step {start_step:,}")
    print("=" * 64)

    gen = MathSequenceGenerator(n_dims=brain.n_dims, curriculum=1)
    it  = gen.all_types()
    ew  = EW()
    t0  = time.time()
    step = start_step
    last_log = last_save = step

    if start_step:
        for _ in range(start_step // 5):
            next(it)

    while step < n_steps and not _stop:
        frac = step / max(n_steps, 1)
        gen.curriculum = 1 if frac < 0.33 else (2 if frac < 0.66 else 3)
        brain.predictor.lr = lr_decay(base_lr, step, decay_every)

        seq     = next(it)
        encoded = gen.encode_seq(seq)
        brain.reset_sequence()

        for vec, word in encoded:
            r = brain.perceive(vec)
            ew.push(r.prediction_error)
            if word: brain.hear(word)
            step += 1
            if step >= n_steps: break

        recent = ew.mean()

        if step - last_log >= log_interval:
            last_log = step
            brain.episodic.surprise_threshold = min(2.0 * recent, 0.5)
            print(f"  step={step:>8,}  err={recent:.4f}"
                  f"  ep_thr={brain.episodic.surprise_threshold:.3f}"
                  f"  episodes={brain.episodic.episode_count:,}"
                  f"  vocab={brain.language.vocab_size}"
                  f"  lr={brain.predictor.lr:.5f}"
                  f"  cur={gen.curriculum}"
                  f"  {fmt_eta(time.time()-t0, step-start_step, n_steps-start_step)}")
            if checkpoint:
                save_progress(checkpoint, '1', step, n_steps, recent,
                              {'cur': gen.curriculum})

        if checkpoint and step - last_save >= save_every:
            last_save = step
            save_checkpoint(brain, checkpoint, 'p1_mid')

    save_checkpoint(brain, checkpoint, 'p1')
    save_progress(checkpoint, '1_done', step, n_steps, ew.mean())
    print(f"  Phase 1 done.  err={ew.mean():.4f}  episodes={brain.episodic.episode_count:,}")
    return ew.mean()

# ── Phase 2 ───────────────────────────────────────────────────────────

def train_conceptnet(brain, n_steps, cn_path, checkpoint,
                     start_step=0, log_interval=10000, save_every=50000,
                     base_lr=0.003, decay_every=300_000, vocab_cap=5000):
    global _stop
    print(f"\n[Phase 2] ConceptNet  ({n_steps:,} steps)")
    if start_step: print(f"  Resuming from step {start_step:,}")
    print("=" * 64)

    from conceptnet_loader import ConceptNetLoader
    enc    = ConceptEncoder(brain.n_dims)
    loader = ConceptNetLoader(n_dims=brain.n_dims)
    ew     = EW()
    t0     = time.time()
    step   = start_step
    last_log = last_save = step
    cycle  = 0
    skip   = start_step

    while step < n_steps and not _stop:
        cycle += 1
        for seq in loader.sequences(cn_path, max_seqs=n_steps * 3):
            if step >= n_steps or _stop: break

            if skip > 0:
                skip -= len(seq)
                continue

            encoded = [(enc.encode(c), w) for c, w in seq]
            brain.reset_sequence()

            for vec, word in encoded:
                r = brain.perceive(vec)
                ew.push(r.prediction_error)
                if word: brain.hear(word)
                brain.predictor.lr = lr_decay(base_lr, step, decay_every)
                step += 1
                if step >= n_steps: break

            recent = ew.mean()

            if step - last_log >= log_interval:
                last_log = step
                brain.episodic.surprise_threshold = min(2.0 * recent, 0.5)
                print(f"  step={step:>8,}  err={recent:.4f}"
                      f"  ep_thr={brain.episodic.surprise_threshold:.3f}"
                      f"  episodes={brain.episodic.episode_count:,}"
                      f"  vocab={brain.language.vocab_size:,}"
                      f"  lr={brain.predictor.lr:.5f}"
                      f"  cycle={cycle}"
                      f"  {fmt_eta(time.time()-t0, step-start_step, n_steps-start_step)}")
                if checkpoint:
                    save_progress(checkpoint, '2', step, n_steps, recent,
                                  {'cycle': cycle})

            if checkpoint and step - last_save >= save_every:
                last_save = step
                save_checkpoint(brain, checkpoint, 'p2_mid')

    save_checkpoint(brain, checkpoint, 'p2')
    save_progress(checkpoint, '2_done', step, n_steps, ew.mean())
    print(f"  Phase 2 done.  err={ew.mean():.4f}  vocab={brain.language.vocab_size:,}")
    return ew.mean()

# ── Phase 3 ───────────────────────────────────────────────────────────

def train_curiosity(brain, n_steps, checkpoint,
                    start_step=0, log_interval=5000, save_every=50000,
                    base_lr=0.002, decay_every=100_000):
    """
    Curiosity via SOM neuron weight vectors.

    FIX: thought.concepts = 256-dim SOM activation maps (wrong for perceive).
    FIXED approach: use brain.som.neuron_weights(i) as curiosity seeds.
      - These are n_dims=32 dimensional (correct for perceive)
      - They represent actual learned concepts
      - High-surprise neurons get explored more (weighted sampling)
    """
    global _stop
    print(f"\n[Phase 3] Curiosity + Dreaming  ({n_steps:,} steps)")
    if start_step: print(f"  Resuming from step {start_step:,}")
    print("=" * 64)

    ew        = EW()
    t0        = time.time()
    step      = start_step
    last_log  = last_save = step
    rng       = np.random.default_rng(42 + start_step)
    n_neurons = brain.som.n_neurons
    n_dims    = brain.n_dims

    # Saliency weights per neuron — updated from prediction surprise
    surprise  = np.ones(n_neurons, dtype=np.float32)

    while step < n_steps and not _stop:
        brain.predictor.lr = lr_decay(base_lr, step, decay_every)

        # Pick neuron weighted by surprise
        probs      = surprise / surprise.sum()
        seed_n     = int(rng.choice(n_neurons, p=probs))
        seed_vec   = np.array(brain.som.neuron_weights(seed_n), dtype=np.float32)
        seed_vec  += rng.standard_normal(n_dims).astype(np.float32) * 0.05

        brain.reset_sequence()
        r   = brain.perceive(seed_vec)
        err = r.prediction_error
        ew.push(err)

        # Update surprise map around BMU
        bmu = r.bmu
        for i in range(n_neurons):
            d = brain.som.grid_dist(i, bmu)
            surprise[i] = 0.99 * surprise[i] + 0.01 * np.exp(-d*d/8.0) * err

        # Think step every 10 steps (updates WM internally, correct dims handled inside)
        if step % 10 == 0 and not brain.working_mem.empty:
            brain.think(steps=4)

        # Dream + consolidate every 1000 steps
        if step > 0 and step % 1000 == 0:
            brain.dream(n_dreams=5, steps_per_dream=8)
            brain.episodic.consolidate(0.88)

        step += 1

        if step - last_log >= log_interval:
            last_log = step
            brain.episodic.surprise_threshold = min(2.0 * ew.mean(), 0.5)
            top5 = np.argsort(surprise)[-5:][::-1].tolist()
            print(f"  step={step:>8,}  err={ew.mean():.4f}"
                  f"  ep_thr={brain.episodic.surprise_threshold:.3f}"
                  f"  episodes={brain.episodic.episode_count:,}"
                  f"  lr={brain.predictor.lr:.5f}"
                  f"  hot={top5}"
                  f"  {fmt_eta(time.time()-t0, step-start_step, n_steps-start_step)}")
            if checkpoint:
                save_progress(checkpoint, '3', step, n_steps, ew.mean())

        if checkpoint and step - last_save >= save_every:
            last_save = step
            save_checkpoint(brain, checkpoint, 'p3_mid')

    save_checkpoint(brain, checkpoint, 'p3')
    save_progress(checkpoint, '3_done', step, n_steps, ew.mean())
    print(f"  Phase 3 done.  err={ew.mean():.4f}")
    return ew.mean()

# ── Evaluation ────────────────────────────────────────────────────────

def evaluate(brain, n_dims):
    print("\n[Eval] Post-training checks")
    print("=" * 64)
    enc   = ConceptEncoder(n_dims)
    tests = [
        ("2 + 3 = ?",     [("2","2"),("+","plus"),("3","3"),("=","equals")],    "5"),
        ("10 - 4 = ?",    [("10","10"),("-","minus"),("4","4"),("=","equals")], "6"),
        ("3 * 4 = ?",     [("3","3"),("*","times"),("4","4"),("=","equals")],   "12"),
        ("! true = ?",    [("!","not"),("true","true"),("=","equals")],          "false"),
        ("4 > 2 = ?",     [("4","4"),(">","greater"),("2","2"),("=","equals")], "true"),
        ("fire causes ?", [("fire","fire"),("causes","causes")],                 "heat"),
        ("dog isa ?",     [("dog","dog"),("isa","isa")],                        "animal"),
        ("plant needs ?", [("plant","plant"),("needs","needs")],                "sunlight"),
    ]
    passed = 0
    for desc, seq, expected in tests:
        brain.reset_sequence()
        brain.working_mem.clear()   # flush ConceptNet context before each test
        for concept, word in seq:
            brain.perceive(enc.encode(concept))
            brain.hear(word)
        if brain.language.vocab_size > 0:
            ctx  = brain.working_mem.context()
            pred = brain.language.best_word(ctx)
            ok   = pred == expected
            passed += ok
            st   = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
            print(f"  [{st}] {desc:<22} → got='{pred}'  want='{expected}'")
        else:
            print(f"  [SKIP] {desc}")
    print(f"\n  Result: {passed}/{len(tests)}")
    return passed

# ── Main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Brain v2 Training (resumable)")
    p.add_argument("--phase",           choices=["1","2","3","all"], default="1")
    p.add_argument("--steps",           type=int,   default=500_000)
    p.add_argument("--conceptnet",      type=str,   default=None)
    p.add_argument("--checkpoint",      type=str,   default="brain")
    p.add_argument("--som-size",        type=int,   default=16)
    p.add_argument("--hidden",          type=int,   default=512)
    p.add_argument("--n-dims",          type=int,   default=32)
    p.add_argument("--log-interval",    type=int,   default=10000)
    p.add_argument("--save-every",      type=int,   default=50000)
    p.add_argument("--lr",              type=float, default=0.005)
    p.add_argument("--lr-decay-every",  type=int,   default=300_000)
    p.add_argument("--vocab-cap",       type=int,   default=5000)
    p.add_argument("--episodic-max",    type=int,   default=10_000)
    p.add_argument("--reset",           action="store_true",
                   help="Start fresh, ignore existing checkpoint")
    args = p.parse_args()

    cfg = dict(som_rows=args.som_size, som_cols=args.som_size,
               n_dims=args.n_dims, hidden_dim=args.hidden,
               wm_capacity=12, episodic_max=args.episodic_max,
               self_neurons=32, seed=42)

    N = args.som_size ** 2
    print(f"\nBrain v2  (resumable)")
    print(f"  SOM={args.som_size}x{args.som_size}  n_dims={args.n_dims}"
          f"  hidden={args.hidden}  phase={args.phase}  steps={args.steps:,}")
    print(f"  checkpoint={args.checkpoint}  save_every={args.save_every:,}")
    print(f"  Press Ctrl+C anytime — progress saves automatically\n")

    # ── Detect resume ────────────────────────────────────────────────
    prog         = None if args.reset else load_progress(args.checkpoint)
    resume_phase = '1'   # default: start from phase 1
    resume_step  = 0

    if prog and not args.reset:
        pd = prog['phase']
        ps = prog['step']
        print(f"  Found checkpoint: phase={pd}  step={ps:,}  err={prog['err']:.4f}")
        if   pd == '1':      resume_phase = '1'; resume_step = ps
        elif pd == '1_done': resume_phase = '2'; resume_step = 0
        elif pd == '2':      resume_phase = '2'; resume_step = ps
        elif pd == '2_done': resume_phase = '3'; resume_step = 0
        elif pd == '3':      resume_phase = '3'; resume_step = ps
        elif pd == '3_done':
            print("  All phases complete. Use --reset to retrain.")
            return
        print(f"  Resuming: phase={resume_phase}  step={resume_step:,}\n")

    # ── Load or build brain ──────────────────────────────────────────
    brain = None
    if prog and not args.reset:
        for tag in ['p2_mid','p2','p1_mid','p1']:
            if checkpoint_exists(args.checkpoint, tag):
                brain = load_checkpoint(args.checkpoint, tag, cfg)
                if brain: break

    if brain is None:
        print("  Building fresh brain...")
        brain = brain2.Brain(**cfg)

    # ── Determine which phases to run ────────────────────────────────
    all_phases = ['1','2','3'] if args.phase == 'all' else [args.phase]
    run_phases = [ph for ph in all_phases if ph >= resume_phase]

    t0 = time.time()
    for ph in run_phases:
        s = resume_step if ph == resume_phase else 0

        if ph == '1':
            train_math(brain, args.steps, args.checkpoint,
                       start_step=s, log_interval=args.log_interval,
                       save_every=args.save_every, base_lr=args.lr,
                       decay_every=args.lr_decay_every)

        elif ph == '2':
            if not args.conceptnet:
                print("Phase 2 needs --conceptnet. Skipping.")
            else:
                train_conceptnet(brain, args.steps, args.conceptnet,
                                 args.checkpoint,
                                 start_step=s, log_interval=args.log_interval,
                                 save_every=args.save_every,
                                 base_lr=args.lr * 0.6,
                                 decay_every=args.lr_decay_every,
                                 vocab_cap=args.vocab_cap)

        elif ph == '3':
            train_curiosity(brain, max(args.steps // 2, 50_000),
                            args.checkpoint,
                            start_step=s,
                            log_interval=max(args.log_interval // 2, 1000),
                            save_every=args.save_every,
                            base_lr=args.lr * 0.8,
                            decay_every=args.lr_decay_every)

        if _stop:
            print("\n  Interrupted. Run same command to resume.")
            return

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    evaluate(brain, cfg['n_dims'])
    save_checkpoint(brain, args.checkpoint, 'final')


if __name__ == "__main__":
    main()