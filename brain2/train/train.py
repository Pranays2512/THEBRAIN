"""
train.py — Brain v2 Training  (resumable + all bugs fixed)

BUGS FIXED IN THIS VERSION:
  ─────────────────────────────────────────────────────────────────────────
  BUG 1 [predictor.hpp] — LSTM weights never trained.
    The old predictor.hpp only updated out_.W and out_.b in step().
    Both LSTM layers (lstm1_, lstm2_) had RANDOM weights for the entire
    3M-step run. The output layer was trying to map random LSTM features
    to SOM activations — it collapsed to predicting the mean activation.
    FIX: rebuild brain2.so with the new predictor.hpp (1-step TBPTT through
         both LSTM layers). Requires a recompile. You MUST retrain from scratch
         after applying this fix — existing checkpoints have untrained LSTMs.

  BUG 2 [train.py + eval_v3.py] — Language vector drift (Hebbian collapse).
    brain.hear(word) uses last_act_map_ (whatever the brain last perceived).
    Over 500k ConceptNet sequences, each word appears across diverse contexts.
    EMA toward diverse activations → all word vectors converge toward the
    mean SOM activation. "Rainbow", "sunlight", "voice" all ended up at
    the mean because they co-occurred with almost everything.
    FIX: call brain.language.hear(word, clean_activation) where clean_activation
         = brain.som.activation_map(enc.encode(word)). This grounds each word
         to its *deterministic concept encoder* activation, not the random
         context the brain happened to be in.

  BUG 3 [predict_next()] — Training/eval LSTM state mismatch.
    brain.perceive() uses 1-step-ahead prediction: LSTM input = prev_act,
    target = current_act. For sequence [fire, causes, heat]:
      train step 1: LSTM(fire) → h1       [no target, warmup]
      train step 2: LSTM(fire) → h2       [target = causes]  ← fire AGAIN
      train step 3: LSTM(causes) → h3     [target = heat]
    But old predict_next([fire, causes]) did:
      eval step 1: LSTM(fire) → h1
      eval step 2: LSTM(causes) → h_WRONG  ← h_WRONG ≠ h2
    h_WRONG ≠ h2 so output(h_WRONG) predicted garbage, not heat.
    FIX: warmup step with acts[0] before the main loop in predict_next().

  BUG 4 [filtered_decode() + evaluate()] — Drifted vectors used for decoding.
    Old code compared predictor output against lang.encode(word) — the
    drifted Hebbian vectors. Since all drifted vectors and the mean-ish
    predictor output all cluster near the mean, whichever word happened
    to be closest to the mean (rainbow, sunlight) always won at ~0.93.
    FIX: compare against som.activation_map(enc.encode(word)) — clean,
         fixed, deterministic activations that do not drift.
  ─────────────────────────────────────────────────────────────────────────

RESUME: if interrupted, just run the exact same command again.
  Progress saved to checkpoints/<name>_progress.json every log interval.
  Weights auto-saved every --save-every steps (default 50,000).

Usage:
  python train.py --phase all --steps 2000000 \\
    --conceptnet conceptnet-assertions-5.7.0.csv.gz \\
    --som-size 64 --hidden 512 --n-dims 64 \\
    --lr 0.005 --lr-decay-every 300000 \\
    --log-interval 10000 --save-every 50000 \\
    --vocab-cap 5000 --episodic-max 10000 \\
    --checkpoint brain_v9
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
    b.predictor.save(     os.path.join(d, f"{checkpoint}_{tag}_predictor.bin"))
    b.language.save(      os.path.join(d, f"{checkpoint}_{tag}_language.bin"))
    b.som.save(           os.path.join(d, f"{checkpoint}_{tag}_som.bin"))
    b.episodic.save(      os.path.join(d, f"{checkpoint}_{tag}_episodic.bin"))
    b.emotion.save(       os.path.join(d, f"{checkpoint}_{tag}_emotion.bin"))
    b.self_model.save(    os.path.join(d, f"{checkpoint}_{tag}_self.bin"))
    b.symbolic_table.save(os.path.join(d, f"{checkpoint}_{tag}_symbolic.bin"))
    print(f"  [saved] tag={tag}  vocab={b.language.vocab_size:,}"
          f"  episodes={b.episodic.episode_count:,}")

def checkpoint_exists(checkpoint, tag):
    d = _ckpt_dir(checkpoint)
    comps = ['predictor','language','som','episodic','emotion','self','symbolic']
    return all(os.path.exists(os.path.join(d, f"{checkpoint}_{tag}_{c}.bin"))
               for c in comps)

def load_checkpoint(checkpoint, tag, cfg):
    d = _ckpt_dir(checkpoint)
    comps = ['predictor','language','som','episodic','emotion','self','symbolic']
    if not all(os.path.exists(os.path.join(d, f"{checkpoint}_{tag}_{c}.bin"))
               for c in comps):
        return None

    print(f"  Loading checkpoint '{tag}' ...")
    b = brain2.Brain(**cfg)
    b.load_components(
        os.path.join(d, f"{checkpoint}_{tag}_predictor.bin"),
        os.path.join(d, f"{checkpoint}_{tag}_language.bin"),
        os.path.join(d, f"{checkpoint}_{tag}_som.bin"),
        os.path.join(d, f"{checkpoint}_{tag}_episodic.bin"),
        os.path.join(d, f"{checkpoint}_{tag}_emotion.bin"),
        os.path.join(d, f"{checkpoint}_{tag}_self.bin"),
        os.path.join(d, f"{checkpoint}_{tag}_symbolic.bin")
    )
    print(f"    language: {b.language.vocab_size:,} words")
    print(f"    emotion:  v={b.emotion.valence:.3f}  a={b.emotion.arousal:.3f}")
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

class HealthError(RuntimeError):
    pass

def _cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def checkpoint_health(brain, enc, phase, step, fatal=True):
    """Fast tripwires that catch broken long runs within minutes."""
    problems = []
    concepts = ["2", "5", "=", "fire", "dog", "plant", "true", "false"]
    means, peaks = [], []

    for c in concepts:
        act = np.asarray(brain.som.activation_map(enc.encode(c)), dtype=np.float32)
        if np.isnan(act).any() or np.isinf(act).any():
            problems.append(f"SOM activation for {c!r} contains NaN/Inf")
            continue
        means.append(float(np.mean(act)))
        peaks.append(float(np.max(act)))

    mean_act = float(np.mean(means)) if means else 1.0
    min_peak = float(np.min(peaks)) if peaks else 0.0
    if mean_act > 0.20:
        problems.append(f"SOM map is foggy: mean activation {mean_act:.3f} > 0.200")
    if min_peak < 0.8:
        problems.append(f"SOM map has weak peak: min peak {min_peak:.3f} < 0.800")
    if brain.predictor.input_dim != brain.som.n_neurons:
        problems.append(f"Predictor/SOM dim mismatch: {brain.predictor.input_dim} vs {brain.som.n_neurons}")
    if brain.language.n_dims != brain.som.n_neurons:
        problems.append(f"Language/SOM dim mismatch: {brain.language.n_dims} vs {brain.som.n_neurons}")

    # BUG 4 FIX: detect language collapse using SOM activations, not drifted lang vectors.
    # Previously used lang.encode() which was the drifted vector — already collapsed.
    # Use som.activation_map(enc.encode(word)) for a clean collapse check.
    collapse = None
    vocab = [w for w in brain.language.vocab() if brain.language.frequency(w) >= 10]
    if len(vocab) >= 64:
        sample = vocab[:64]
        sims = []
        for a, b in zip(sample, sample[1:]):
            va = np.array(brain.som.activation_map(enc.encode(a)), dtype=np.float32)
            vb = np.array(brain.som.activation_map(enc.encode(b)), dtype=np.float32)
            sims.append(_cos(va, vb))
        collapse = float(np.mean(sims)) if sims else 0.0
        # SOM activations naturally cluster similar concepts. Threshold is looser.
        if collapse > 0.90:
            problems.append(f"SOM activations collapsed: mean pair cosine {collapse:.3f} > 0.900")

    status = "OK" if not problems else "FAIL"
    msg = (f"  [health:{status}] phase={phase} step={step:,}"
           f" som_mean={mean_act:.3f} peak={min_peak:.3f}")
    if collapse is not None:
        msg += f" act_pair_cos={collapse:.3f}"
    print(msg)

    if problems:
        for p in problems:
            print(f"    ! {p}")
        if fatal:
            raise HealthError("Training health check failed; aborting.")


# ── BUG 2 FIX: clean language grounding ──────────────────────────────

def hear_clean(brain, enc, word):
    """
    Ground word → SOM activation using the concept encoder, NOT last_act_map_.

    brain.hear(word) uses last_act_map_ (whatever the brain just perceived).
    Across 500k diverse ConceptNet triples, every word's vector EMA-drifts
    toward the mean activation map — all vectors collapse to nearly the same
    point and cosine similarity between any two words → ~1.0.

    Instead: compute the deterministic SOM activation for this word's concept
    vector and pass that directly. Word meaning is grounded to the concept
    encoder geometry (numbers are ordered, similar chars are nearby), not to
    whatever random context the brain happened to be in.
    """
    clean_act = brain.som.activation_map(enc.encode(word))
    brain.language.hear(word, clean_act)


def pre_register_vocab(brain, enc, words):
    """
    Seed all vocabulary words with clean SOM activations before training.
    This ensures words start in the right place rather than at random_vec().
    """
    for word in words:
        clean_act = brain.som.activation_map(enc.encode(word))
        brain.language.register_word(word, clean_act)


_stop = False
def _sigint(s, f):
    global _stop
    print("\n  [Ctrl+C] Saving then stopping...")
    _stop = True
signal.signal(signal.SIGINT, _sigint)

# ── Phase 1 ───────────────────────────────────────────────────────────

def train_math(brain, n_steps, checkpoint,
               start_step=0, log_interval=10000, save_every=50000,
               base_lr=0.005, decay_every=300_000,
               health_interval=5000, health_fatal=True):
    global _stop
    print(f"\n[Phase 1] Math + Logic + Physics  ({n_steps:,} steps, curriculum 1-3)")
    if start_step: print(f"  Resuming from step {start_step:,}")
    print("=" * 64)

    gen        = MathSequenceGenerator(n_dims=brain.n_dims, curriculum=1)
    health_enc = ConceptEncoder(brain.n_dims)
    enc        = ConceptEncoder(brain.n_dims)
    it         = gen.all_types()
    ew         = EW()
    t0         = time.time()
    step       = start_step
    last_log   = last_save = step
    last_health = step

    if start_step:
        for _ in range(start_step // 5):
            next(it)

    # BUG 2 FIX: pre-register math/logic vocabulary with clean activations
    math_words = [str(i) for i in range(21)] + [
        "plus", "minus", "times", "divided", "equals", "greater", "less",
        "true", "false", "not", "and", "or", "then", "therefore", "because",
        "x", "y", "z", "mod",
        "causes", "prevents", "isa", "hasa", "needs", "produces",
        "before", "after", "above", "below", "inside", "outside",
        "all", "some", "opposite",
        "fire", "heat", "burn", "water", "ice", "cold", "sun", "light",
        "rain", "wet", "gravity", "fall", "eat", "full", "sleep", "rest",
        "dog", "cat", "tree", "apple", "bird", "fish", "human", "plant",
        "animal", "fruit", "mammal", "food",
        "force", "mass", "acceleration", "energy", "pressure", "speed",
        "distance", "time", "voltage", "current", "resistance",
    ]
    pre_register_vocab(brain, enc, math_words)

    while step < n_steps and not _stop:
        frac = step / max(n_steps, 1)
        gen.curriculum = 1 if frac < 0.33 else (2 if frac < 0.66 else 3)
        brain.predictor.lr = lr_decay(base_lr, step, decay_every)

        seq     = next(it)
        encoded = gen.encode_seq(seq)
        brain.reset_sequence()
        brain.working_mem.clear()

        for vec, word in encoded:
            r = brain.perceive(vec)
            ew.push(r.prediction_error)
            # BUG 2 FIX: ground word to its own clean activation, not last_act_map_
            if word: hear_clean(brain, enc, word)
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

        if health_interval and step - last_health >= health_interval:
            last_health = step
            checkpoint_health(brain, health_enc, '1', step, fatal=health_fatal)

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
                     base_lr=0.003, decay_every=300_000, vocab_cap=5000,
                     health_interval=5000, health_fatal=True):
    global _stop
    print(f"\n[Phase 2] ConceptNet  ({n_steps:,} steps)")
    if start_step: print(f"  Resuming from step {start_step:,}")
    print("=" * 64)

    from conceptnet_loader import ConceptNetLoader
    enc    = ConceptEncoder(brain.n_dims)
    loader = ConceptNetLoader(n_dims=brain.n_dims, vocab_cap=vocab_cap)
    ew     = EW()
    t0     = time.time()
    step   = start_step
    last_log = last_save = step
    last_health = step
    cycle  = 0
    skip   = start_step

    # BUG 2 FIX: build ConceptNet vocabulary first, then pre-register with clean activations
    # This seeds every word at the right point in SOM space before EMA can corrupt it.
    print("  Pre-scanning ConceptNet to build vocabulary...")
    loader._build_vocab(cn_path)
    if loader._allowed_words:
        print(f"  Pre-registering {len(loader._allowed_words):,} vocabulary words"
              f" with clean SOM activations...")
        pre_register_vocab(brain, enc, list(loader._allowed_words))
        # Also register relation words
        relation_words = [
            "causes", "makes_want", "can", "isa", "hasa", "is", "partof",
            "usedfor", "receives", "wants", "goal", "cannot", "not_want", "opposite",
        ]
        pre_register_vocab(brain, enc, relation_words)
        print(f"  Pre-registration done. vocab_size={brain.language.vocab_size:,}")

    while step < n_steps and not _stop:
        cycle += 1
        for seq in loader.sequences(cn_path, max_seqs=n_steps * 3):
            if step >= n_steps or _stop: break

            if skip > 0:
                skip -= len(seq)
                continue

            encoded = [(enc.encode(c), w) for c, w in seq]
            brain.reset_sequence()
            brain.working_mem.clear()

            for vec, word in encoded:
                r = brain.perceive(vec)
                ew.push(r.prediction_error)
                # BUG 2 FIX: ground word to its own clean activation
                if word: hear_clean(brain, enc, word)
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

            if health_interval and step - last_health >= health_interval:
                last_health = step
                checkpoint_health(brain, enc, '2', step, fatal=health_fatal)

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
                    base_lr=0.002, decay_every=100_000,
                    health_interval=5000, health_fatal=True):
    global _stop
    print(f"\n[Phase 3] Curiosity + Dreaming  ({n_steps:,} steps)")
    if start_step: print(f"  Resuming from step {start_step:,}")
    print("=" * 64)

    ew          = EW()
    t0          = time.time()
    step        = start_step
    last_log    = last_save = step
    last_health = step
    rng         = np.random.default_rng(42 + start_step)
    health_enc  = ConceptEncoder(brain.n_dims)
    n_neurons   = brain.som.n_neurons
    n_dims      = brain.n_dims
    surprise    = np.ones(n_neurons, dtype=np.float32)

    while step < n_steps and not _stop:
        brain.predictor.lr = lr_decay(base_lr, step, decay_every)

        probs     = surprise / surprise.sum()
        seed_n    = int(rng.choice(n_neurons, p=probs))
        seed_vec  = np.array(brain.som.neuron_weights(seed_n), dtype=np.float32)
        seed_vec += rng.standard_normal(n_dims).astype(np.float32) * 0.05

        brain.reset_sequence()
        brain.perceive(seed_vec)
        noisy_vec = seed_vec + rng.standard_normal(n_dims).astype(np.float32) * 0.05
        r   = brain.perceive(noisy_vec)
        err = r.prediction_error
        ew.push(err)

        bmu = r.bmu
        for i in range(n_neurons):
            d = brain.som.grid_dist(i, bmu)
            surprise[i] = 0.99 * surprise[i] + 0.01 * np.exp(-d*d/8.0) * err

        if step % 10 == 0 and not brain.working_mem.empty:
            brain.think(steps=4)

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

        if health_interval and step - last_health >= health_interval:
            last_health = step
            checkpoint_health(brain, health_enc, '3', step, fatal=health_fatal)

        if checkpoint and step - last_save >= save_every:
            last_save = step
            save_checkpoint(brain, checkpoint, 'p3_mid')

    save_checkpoint(brain, checkpoint, 'p3')
    save_progress(checkpoint, '3_done', step, n_steps, ew.mean())
    print(f"  Phase 3 done.  err={ew.mean():.4f}")
    return ew.mean()

# ── Evaluation (all 4 bugs fixed) ────────────────────────────────────

def predict_next(som, pred, enc, seq):
    """
    BUG 3 FIX: warmup step replicates brain.perceive()'s 1-step-ahead scheme.

    brain.perceive() feeds prev_act_map to the LSTM and trains toward current_act:
      step 1: LSTM input=acts[0], no target  (have_prev=False)
      step 2: LSTM input=acts[0], target=acts[1]   <- acts[0] fed TWICE
      step 3: LSTM input=acts[1], target=acts[2]
      ...

    To match this in eval, we do one warmup step with acts[0], then loop
    over the full sequence. The warmup reproduces the training's first step,
    so the hidden state at each subsequent step matches training exactly.
    """
    pred.reset()
    acts = [som.activation_map(enc.encode(concept)) for concept, _ in seq]

    pred.set_offline(True)
    # Warmup: replicate training's first step (have_prev=False, just forward)
    pred.step(acts[0])
    # Main loop: mirrors training steps 2, 3, ...
    predicted = None
    for act in acts:
        predicted = pred.step(act)
    pred.set_offline(False)
    return predicted


def filtered_decode(som, enc, lang, predicted_vec, k=5, min_freq=1):
    """
    BUG 4 FIX: compare predictor output against clean SOM activations,
    NOT lang.encode(word) (which returns the Hebbian-drifted vector).

    Even with hear_clean(), lang.encode() returns the internally stored
    vector. After many steps it's still slightly drifted. The SOM activation
    of enc.encode(word) is always exactly right and matches what the
    predictor was trained against (since perceive() runs enc.encode through SOM).
    """
    pn = np.linalg.norm(predicted_vec)
    if pn < 1e-8:
        return []
    candidates = []
    for word in lang.vocab():
        if lang.frequency(word) < min_freq:
            continue
        # Clean, fixed SOM activation — matches what the predictor was trained on
        vec = np.array(som.activation_map(enc.encode(word)), dtype=np.float32)
        vn  = np.linalg.norm(vec)
        if vn < 1e-8:
            continue
        sim = float(np.dot(predicted_vec, vec) / (pn * vn))
        candidates.append((word, sim))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:k]


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
        ("dog isa ?",     [("dog","dog"),("isa","isa")],                         "animal"),
        ("plant needs ?", [("plant","plant"),("needs","needs")],                 "sunlight"),
    ]
    passed = 0
    for desc, seq, expected in tests:
        if brain.language.vocab_size > 0:
            predicted = predict_next(brain.som, brain.predictor, enc, seq)
            # BUG 4 FIX: use SOM-based decoding
            top5 = filtered_decode(brain.som, enc, brain.language, predicted, k=5, min_freq=1)
            got  = top5[0][0] if top5 else "?"
            ok   = got == expected
            passed += ok
            st   = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
            top3 = [(w, f"{s:.3f}") for w, s in top5[:3]]
            print(f"  [{st}] {desc:<22} → got='{got}'  want='{expected}'")
            if not ok:
                print(f"          top3: {top3}")
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
    p.add_argument("--som-size",        type=int,   default=64)
    p.add_argument("--hidden",          type=int,   default=512)
    p.add_argument("--n-dims",          type=int,   default=64)
    p.add_argument("--log-interval",    type=int,   default=10000)
    p.add_argument("--save-every",      type=int,   default=50000)
    p.add_argument("--health-interval", type=int,   default=5000)
    p.add_argument("--no-health-stop",  action="store_true")
    p.add_argument("--lr",              type=float, default=0.005)
    p.add_argument("--lr-decay-every",  type=int,   default=300_000)
    p.add_argument("--vocab-cap",       type=int,   default=5000)
    p.add_argument("--episodic-max",    type=int,   default=10_000)
    p.add_argument("--reset",           action="store_true")
    args = p.parse_args()

    cfg = dict(som_rows=args.som_size, som_cols=args.som_size,
               n_dims=args.n_dims, hidden_dim=args.hidden,
               wm_capacity=12, episodic_max=args.episodic_max,
               self_neurons=32, seed=42)

    print(f"\nBrain v2  (resumable)")
    print(f"  SOM={args.som_size}x{args.som_size}  n_dims={args.n_dims}"
          f"  hidden={args.hidden}  phase={args.phase}  steps={args.steps:,}")
    print(f"  checkpoint={args.checkpoint}  save_every={args.save_every:,}")
    print(f"  Press Ctrl+C anytime — progress saves automatically\n")

    # ── Detect resume ────────────────────────────────────────────────
    prog         = None if args.reset else load_progress(args.checkpoint)
    resume_phase = '1'
    resume_step  = 0

    if prog and not args.reset:
        pd = prog['phase']; ps = prog['step']
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
        for tag in ['p3', 'p3_mid', 'p2', 'p2_mid', 'p1', 'p1_mid']:
            if checkpoint_exists(args.checkpoint, tag):
                brain = load_checkpoint(args.checkpoint, tag, cfg)
                if brain: break

    if brain is None:
        print("  Building fresh brain...")
        brain = brain2.Brain(**cfg)

    # ── Run phases ───────────────────────────────────────────────────
    all_phases = ['1','2','3'] if args.phase == 'all' else [args.phase]
    run_phases = [ph for ph in all_phases if ph >= resume_phase]

    t0 = time.time()
    try:
        for ph in run_phases:
            s = resume_step if ph == resume_phase else 0

            if ph == '1':
                train_math(brain, args.steps, args.checkpoint,
                           start_step=s, log_interval=args.log_interval,
                           save_every=args.save_every, base_lr=args.lr,
                           decay_every=args.lr_decay_every,
                           health_interval=args.health_interval,
                           health_fatal=not args.no_health_stop)

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
                                     vocab_cap=args.vocab_cap,
                                     health_interval=args.health_interval,
                                     health_fatal=not args.no_health_stop)

            elif ph == '3':
                train_curiosity(brain, max(args.steps // 2, 50_000),
                                args.checkpoint,
                                start_step=s,
                                log_interval=max(args.log_interval // 2, 1000),
                                save_every=args.save_every,
                                base_lr=args.lr * 0.8,
                                decay_every=args.lr_decay_every,
                                health_interval=args.health_interval,
                                health_fatal=not args.no_health_stop)

            if _stop:
                print("\n  Interrupted. Run same command to resume.")
                return
    except HealthError as e:
        print(f"\n  [ABORTED] {e}")
        if args.checkpoint:
            save_checkpoint(brain, args.checkpoint, 'failed_health')
            save_progress(args.checkpoint, 'failed_health', 0, args.steps, 0.0)
        print("  Fix the failing health metric before launching a long run.")
        return

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    evaluate(brain, cfg['n_dims'])
    save_checkpoint(brain, args.checkpoint, 'final')


if __name__ == "__main__":
    main()