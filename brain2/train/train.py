"""
train.py — Brain v2 Training  (resumable, N-step BPTT, answer-only loss)

KEY CHANGES FROM PREVIOUS VERSION:
  ─────────────────────────────────────────────────────────────────────────
  WHY THE PREVIOUS RUN STILL FAILED (0/8 after 2 days):

  The 1-step TBPTT in step() cannot propagate gradient far enough back.
  For "2 + 3 = 5" (5 tokens), the loss at the "=" step only reaches back
  1 step to "3". The weights that processed "2" and "+" never get any
  signal to produce "5". The LSTM learns to echo the last input, not to
  reason through the sequence.

  FIX: predictor.train_sequence(inputs, target)
    — feeds entire input sequence through LSTM (recording snapshots)
    — computes loss ONLY at the final output against `target` (answer-only)
    — runs full N-step BPTT back through all stored snapshots
    — gradient from "predict 5" now flows all the way back to "2"

  TRAINING EXAMPLES:
    Math   "2 + 3 = ?":  inputs=[act(2),act(+),act(3),act(=)], target=act(5)
    ConceptNet "dog isa animal": inputs=[act(dog),act(isa)], target=act(animal)

  PREDICT_NEXT CHANGE:
    The warmup step was needed to match brain.perceive()'s "prev_act fed twice"
    scheme. train_sequence() feeds each token exactly once, so the warmup is
    removed from predict_next(). Eval now correctly mirrors training.

  PYBIND11: add to your bindings file before recompiling:
    .def("train_sequence",
         [](brain2::Predictor& p,
            const std::vector<std::vector<float>>& inputs,
            const std::vector<float>& target,
            int n_bptt) { return p.train_sequence(inputs, target, n_bptt); },
         py::arg("inputs"), py::arg("target"), py::arg("n_bptt") = -1)
  ─────────────────────────────────────────────────────────────────────────

Usage:
  python train/train.py --phase all --steps 2000000 \\
    --conceptnet train/conceptnet-assertions-5.7.0.csv.gz \\
    --som-size 64 --hidden 512 --n-dims 64 \\
    --lr 0.005 --lr-decay-every 400000 \\
    --log-interval 10000 --save-every 50000 \\
    --vocab-cap 5000 --episodic-max 10000 \\
    --checkpoint brain_v10 --reset
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

def _ckpt_dir(checkpoint):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
    os.makedirs(d, exist_ok=True)
    return d

def _progress_path(checkpoint):
    return os.path.join(_ckpt_dir(checkpoint), f"{checkpoint}_progress.json")

# ── Save / Load ───────────────────────────────────────────────────────

def save_checkpoint(b, checkpoint, tag):
    d = _ckpt_dir(checkpoint)
    b.predictor.save(      os.path.join(d, f"{checkpoint}_{tag}_predictor.bin"))
    b.language.save(       os.path.join(d, f"{checkpoint}_{tag}_language.bin"))
    b.som.save(            os.path.join(d, f"{checkpoint}_{tag}_som.bin"))
    b.episodic.save(       os.path.join(d, f"{checkpoint}_{tag}_episodic.bin"))
    b.emotion.save(        os.path.join(d, f"{checkpoint}_{tag}_emotion.bin"))
    b.self_model.save(     os.path.join(d, f"{checkpoint}_{tag}_self.bin"))
    b.symbolic_table.save( os.path.join(d, f"{checkpoint}_{tag}_symbolic.bin"))
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
    print(f"    vocab={b.language.vocab_size:,}  lr={b.predictor.lr:.6f}")
    return b

# ── Progress ──────────────────────────────────────────────────────────

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
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

def checkpoint_health(brain, enc, phase, step, fatal=True):
    problems = []
    concepts = ["2", "5", "=", "fire", "dog", "plant", "true", "false"]
    means, peaks = [], []
    for c in concepts:
        act = np.asarray(brain.som.activation_map(enc.encode(c)), dtype=np.float32)
        if np.isnan(act).any() or np.isinf(act).any():
            problems.append(f"SOM NaN/Inf for {c!r}")
            continue
        means.append(float(np.mean(act)))
        peaks.append(float(np.max(act)))

    mean_act = float(np.mean(means)) if means else 1.0
    min_peak  = float(np.min(peaks)) if peaks else 0.0
    if mean_act > 0.20: problems.append(f"SOM foggy: mean={mean_act:.3f}")
    if min_peak  < 0.80: problems.append(f"SOM weak peak: {min_peak:.3f}")
    if brain.predictor.input_dim != brain.som.n_neurons:
        problems.append("Predictor/SOM dim mismatch")
    if brain.language.n_dims != brain.som.n_neurons:
        problems.append("Language/SOM dim mismatch")

    collapse = None
    vocab = [w for w in brain.language.vocab() if brain.language.frequency(w) >= 10]
    if len(vocab) >= 64:
        sims = [_cos(np.array(brain.som.activation_map(enc.encode(a)), dtype=np.float32),
                     np.array(brain.som.activation_map(enc.encode(b)), dtype=np.float32))
                for a, b in zip(vocab[:64], vocab[1:65])]
        collapse = float(np.mean(sims))
        if collapse > 0.90:
            problems.append(f"SOM activations collapsed: {collapse:.3f}")

    status = "OK" if not problems else "FAIL"
    msg = (f"  [health:{status}] phase={phase} step={step:,}"
           f" som_mean={mean_act:.3f} peak={min_peak:.3f}")
    if collapse is not None: msg += f" act_pair_cos={collapse:.3f}"
    print(msg)
    if problems:
        for pr in problems: print(f"    ! {pr}")
        if fatal: raise HealthError("Health check failed; aborting.")

# ── Language grounding (clean activations, no Hebbian drift) ──────────

def hear_clean(brain, enc, word):
    """Ground word to its own deterministic SOM activation, not last_act_map_."""
    clean_act = brain.som.activation_map(enc.encode(word))
    brain.language.hear(word, clean_act)

def pre_register_vocab(brain, enc, words):
    """Seed words with clean SOM activations before training."""
    for word in words:
        clean_act = brain.som.activation_map(enc.encode(word))
        brain.language.register_word(word, clean_act)

# ── Core sequence trainer ─────────────────────────────────────────────

def run_sequence(brain, enc, seq, ew, use_train_sequence=True):
    """
    Process one training sequence through the brain.

    For Phases 1+2 (use_train_sequence=True):
      - SOM: updated via brain.perceive() with predictor offline
      - Language: grounded via hear_clean()
      - Predictor: trained via train_sequence() with full N-step BPTT
        and answer-only loss (loss only at the final token)

    For Phase 3 (use_train_sequence=False):
      - Full brain.perceive() online loop (1-step BPTT, fine for curiosity)

    Returns prediction error at the answer token.
    """
    if not seq:
        return 0.0

    # Build clean SOM activations for every token
    acts = [np.array(brain.som.activation_map(enc.encode(c)), dtype=np.float32)
            for c, _ in seq]
    words = [w for _, w in seq]

    if use_train_sequence:
        # ── SOM update (offline predictor — no weight update in perceive) ──
        brain.predictor.set_offline(True)
        brain.reset_sequence()
        brain.working_mem.clear()
        for act in acts:
            brain.perceive(act.tolist())
        brain.predictor.set_offline(False)

        # ── Language grounding ────────────────────────────────────────────
        for word in words:
            if word:
                hear_clean(brain, enc, word)

        # ── Predictor: answer-only loss + full N-step BPTT ───────────────
        # Input sequence = all tokens except the last.
        # Target         = the last token (the "answer").
        # For ConceptNet [A, rel, B]: inputs=[A,rel], target=B
        # For math [2,+,3,=,5]:      inputs=[2,+,3,=], target=5
        if len(acts) >= 2:
            brain.predictor.reset()
            err = brain.predictor.train_sequence(
                [a.tolist() for a in acts[:-1]],  # inputs
                acts[-1].tolist(),                  # target (answer)
                -1                                  # n_bptt = full sequence
            )
            ew.push(err)
            return err
        return 0.0
    else:
        # Online 1-step BPTT (Phase 3 curiosity)
        brain.reset_sequence()
        brain.working_mem.clear()
        err = 0.0
        for act, word in zip(acts, words):
            r = brain.perceive(act.tolist())
            err = r.prediction_error
            ew.push(err)
            if word: hear_clean(brain, enc, word)
        return err

_stop = False
def _sigint(s, f):
    global _stop
    print("\n  [Ctrl+C] Saving then stopping...")
    _stop = True
signal.signal(signal.SIGINT, _sigint)

# ── Phase 1: Math + Logic ─────────────────────────────────────────────

def train_math(brain, n_steps, checkpoint,
               start_step=0, log_interval=10000, save_every=50000,
               base_lr=0.005, decay_every=400_000,
               health_interval=5000, health_fatal=True):
    global _stop
    print(f"\n[Phase 1] Math + Logic + Physics  ({n_steps:,} steps, curriculum 1→3)")
    if start_step: print(f"  Resuming from step {start_step:,}")
    print("=" * 64)

    gen        = MathSequenceGenerator(n_dims=brain.n_dims, curriculum=1)
    enc        = ConceptEncoder(brain.n_dims)
    health_enc = ConceptEncoder(brain.n_dims)
    it         = gen.all_types()
    ew         = EW()
    t0         = time.time()
    step       = start_step
    last_log   = last_save = last_health = step

    if start_step:
        for _ in range(start_step // 5):
            next(it)

    # Pre-register math vocabulary with clean SOM activations
    math_words = [str(i) for i in range(21)] + [
        "plus","minus","times","divided","equals","greater","less",
        "true","false","not","and","or","then","therefore","because",
        "x","y","z","mod",
        "causes","prevents","isa","hasa","needs","produces",
        "before","after","above","below","inside","outside",
        "all","some","opposite",
        "fire","heat","burn","water","ice","cold","sun","light",
        "rain","wet","gravity","fall","eat","full","sleep","rest",
        "dog","cat","tree","apple","bird","fish","human","plant",
        "animal","fruit","mammal","food",
        "force","mass","acceleration","energy","pressure","speed",
        "distance","time","voltage","current","resistance",
    ]
    pre_register_vocab(brain, enc, math_words)

    while step < n_steps and not _stop:
        frac = step / max(n_steps, 1)
        gen.curriculum = 1 if frac < 0.33 else (2 if frac < 0.66 else 3)
        brain.predictor.lr = lr_decay(base_lr, step, decay_every)

        seq = next(it)
        run_sequence(brain, enc, seq, ew, use_train_sequence=True)
        step += len(seq)

        if step - last_log >= log_interval:
            last_log = step
            recent = ew.mean()
            brain.episodic.surprise_threshold = min(2.0 * recent, 0.5)
            print(f"  step={step:>8,}  err={recent:.4f}"
                  f"  ep_thr={brain.episodic.surprise_threshold:.3f}"
                  f"  episodes={brain.episodic.episode_count:,}"
                  f"  vocab={brain.language.vocab_size}"
                  f"  lr={brain.predictor.lr:.5f}"
                  f"  cur={gen.curriculum}"
                  f"  {fmt_eta(time.time()-t0, step-start_step, n_steps-start_step)}")
            save_progress(checkpoint, '1', step, n_steps, recent, {'cur': gen.curriculum})

        if health_interval and step - last_health >= health_interval:
            last_health = step
            checkpoint_health(brain, health_enc, '1', step, fatal=health_fatal)

        if checkpoint and step - last_save >= save_every:
            last_save = step
            save_checkpoint(brain, checkpoint, 'p1_mid')

    if not _stop:
        save_checkpoint(brain, checkpoint, 'p1')
        save_progress(checkpoint, '1_done', step, n_steps, ew.mean())
        print(f"  Phase 1 done.  err={ew.mean():.4f}  vocab={brain.language.vocab_size}")
    return ew.mean()

# ── Phase 2: ConceptNet ───────────────────────────────────────────────

def train_conceptnet(brain, n_steps, cn_path, checkpoint,
                     start_step=0, log_interval=10000, save_every=50000,
                     base_lr=0.003, decay_every=400_000, vocab_cap=5000,
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
    last_log = last_save = last_health = step
    cycle  = 0
    skip   = start_step

    # Pre-register entire ConceptNet vocab with clean SOM activations
    print("  Building ConceptNet vocabulary...")
    loader._build_vocab(cn_path)
    if loader._allowed_words:
        print(f"  Pre-registering {len(loader._allowed_words):,} words...")
        pre_register_vocab(brain, enc, list(loader._allowed_words))
        pre_register_vocab(brain, enc, [
            "causes","makes_want","can","isa","hasa","is","partof",
            "usedfor","receives","wants","goal","cannot","not_want","opposite",
        ])
        print(f"  Pre-registration done. vocab={brain.language.vocab_size:,}")

    while step < n_steps and not _stop:
        cycle += 1
        for seq in loader.sequences(cn_path, max_seqs=n_steps * 3):
            if step >= n_steps or _stop: break
            if skip > 0:
                skip -= len(seq)
                continue

            brain.predictor.lr = lr_decay(base_lr, step, decay_every)
            run_sequence(brain, enc, seq, ew, use_train_sequence=True)
            step += len(seq)

            if step - last_log >= log_interval:
                last_log = step
                recent = ew.mean()
                brain.episodic.surprise_threshold = min(2.0 * recent, 0.5)
                print(f"  step={step:>8,}  err={recent:.4f}"
                      f"  ep_thr={brain.episodic.surprise_threshold:.3f}"
                      f"  episodes={brain.episodic.episode_count:,}"
                      f"  vocab={brain.language.vocab_size:,}"
                      f"  lr={brain.predictor.lr:.5f}  cycle={cycle}"
                      f"  {fmt_eta(time.time()-t0, step-start_step, n_steps-start_step)}")
                save_progress(checkpoint, '2', step, n_steps, recent, {'cycle': cycle})

            if health_interval and step - last_health >= health_interval:
                last_health = step
                checkpoint_health(brain, enc, '2', step, fatal=health_fatal)

            if checkpoint and step - last_save >= save_every:
                last_save = step
                save_checkpoint(brain, checkpoint, 'p2_mid')

    if not _stop:
        save_checkpoint(brain, checkpoint, 'p2')
        save_progress(checkpoint, '2_done', step, n_steps, ew.mean())
        print(f"  Phase 2 done.  err={ew.mean():.4f}  vocab={brain.language.vocab_size:,}")
    return ew.mean()

# ── Phase 3: Curiosity + Dreaming ────────────────────────────────────

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
    last_log    = last_save = last_health = step
    rng         = np.random.default_rng(42 + start_step)
    health_enc  = ConceptEncoder(brain.n_dims)
    n_neurons   = brain.som.n_neurons
    n_dims      = brain.n_dims
    surprise    = np.ones(n_neurons, dtype=np.float32)

    while step < n_steps and not _stop:
        brain.predictor.lr = lr_decay(base_lr, step, decay_every)

        probs    = surprise / surprise.sum()
        seed_n   = int(rng.choice(n_neurons, p=probs))
        seed_vec = np.array(brain.som.neuron_weights(seed_n), dtype=np.float32)
        seed_vec += rng.standard_normal(n_dims).astype(np.float32) * 0.05

        # 2-step curiosity sequence — use online 1-step BPTT (fine here)
        brain.reset_sequence()
        brain.perceive(seed_vec.tolist())
        noisy = seed_vec + rng.standard_normal(n_dims).astype(np.float32) * 0.05
        r = brain.perceive(noisy.tolist())
        err = r.prediction_error
        ew.push(err)

        bmu = r.bmu
        for i in range(n_neurons):
            d = brain.som.grid_dist(i, bmu)
            surprise[i] = 0.99*surprise[i] + 0.01*np.exp(-d*d/8.0)*err

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
                  f"  lr={brain.predictor.lr:.5f}  hot={top5}"
                  f"  {fmt_eta(time.time()-t0, step-start_step, n_steps-start_step)}")
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

# ── Eval (matches train_sequence training exactly) ────────────────────

def predict_next(som, pred, enc, seq):
    """
    Feed all tokens as inputs, return prediction for the answer token.

    NO warmup step — train_sequence feeds each token exactly once,
    so eval does the same. The old warmup was only needed to match
    brain.perceive()'s 'prev_act fed twice' scheme, which is no
    longer used for Phases 1 and 2.

    Mirrors: train_sequence(inputs=acts[:-1], target=acts[-1])
    """
    pred.reset()
    acts = [np.array(som.activation_map(enc.encode(c)), dtype=np.float32)
            for c, _ in seq]
    pred.set_offline(True)
    predicted = None
    for act in acts:
        predicted = pred.step(act.tolist())
    pred.set_offline(False)
    return np.array(predicted, dtype=np.float32)


def filtered_decode(som, enc, lang, predicted_vec, k=5, min_freq=1):
    """
    Decode using clean SOM activations, not drifted language vectors.
    Matches what train_sequence was trained to predict.
    """
    pn = np.linalg.norm(predicted_vec)
    if pn < 1e-8:
        return []
    candidates = []
    for word in lang.vocab():
        if lang.frequency(word) < min_freq:
            continue
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
    enc = ConceptEncoder(n_dims)
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
        if brain.language.vocab_size == 0:
            print(f"  [SKIP] {desc}"); continue
        predicted = predict_next(brain.som, brain.predictor, enc, seq)
        top5 = filtered_decode(brain.som, enc, brain.language, predicted, k=5, min_freq=1)
        got  = top5[0][0] if top5 else "?"
        ok   = got == expected
        passed += ok
        st   = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
        print(f"  [{st}] {desc:<22} → got='{got}'  want='{expected}'")
        if not ok:
            print(f"          top3: {[(w,f'{s:.3f}') for w,s in top5[:3]]}")
    print(f"\n  Result: {passed}/{len(tests)}")
    return passed

# ── Main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
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
    p.add_argument("--lr-decay-every",  type=int,   default=400_000)
    p.add_argument("--vocab-cap",       type=int,   default=5000)
    p.add_argument("--episodic-max",    type=int,   default=10_000)
    p.add_argument("--reset",           action="store_true")
    args = p.parse_args()

    cfg = dict(som_rows=args.som_size, som_cols=args.som_size,
               n_dims=args.n_dims, hidden_dim=args.hidden,
               wm_capacity=12, episodic_max=args.episodic_max,
               self_neurons=32, seed=42)

    print(f"\nBrain v2  —  N-step BPTT + answer-only loss")
    print(f"  SOM={args.som_size}x{args.som_size}  n_dims={args.n_dims}"
          f"  hidden={args.hidden}  phase={args.phase}  steps={args.steps:,}")
    print(f"  checkpoint={args.checkpoint}  Press Ctrl+C to save+stop\n")

    prog         = None if args.reset else load_progress(args.checkpoint)
    resume_phase = '1'
    resume_step  = 0

    if prog and not args.reset:
        pd, ps = prog['phase'], prog['step']
        print(f"  Found checkpoint: phase={pd}  step={ps:,}  err={prog['err']:.4f}")
        if   pd == '1':      resume_phase='1'; resume_step=ps
        elif pd == '1_done': resume_phase='2'; resume_step=0
        elif pd == '2':      resume_phase='2'; resume_step=ps
        elif pd == '2_done': resume_phase='3'; resume_step=0
        elif pd == '3':      resume_phase='3'; resume_step=ps
        elif pd == '3_done':
            print("  All phases complete. Use --reset to retrain.")
            return
        print(f"  Resuming: phase={resume_phase}  step={resume_step:,}\n")

    brain = None
    if prog and not args.reset:
        for tag in ['p3','p3_mid','p2','p2_mid','p1','p1_mid']:
            if checkpoint_exists(args.checkpoint, tag):
                brain = load_checkpoint(args.checkpoint, tag, cfg)
                if brain: break

    if brain is None:
        print("  Building fresh brain...")
        brain = brain2.Brain(**cfg)

    all_phases = ['1','2','3'] if args.phase == 'all' else [args.phase]
    run_phases = [ph for ph in all_phases if ph >= resume_phase]

    t0 = time.time()
    try:
        for ph in run_phases:
            s = resume_step if ph == resume_phase else 0

            if ph == '1':
                train_math(brain, args.steps, args.checkpoint,
                           start_step=s,
                           log_interval=args.log_interval,
                           save_every=args.save_every,
                           base_lr=args.lr,
                           decay_every=args.lr_decay_every,
                           health_interval=args.health_interval,
                           health_fatal=not args.no_health_stop)

            elif ph == '2':
                if not args.conceptnet:
                    print("Phase 2 needs --conceptnet. Skipping.")
                else:
                    train_conceptnet(brain, args.steps, args.conceptnet,
                                     args.checkpoint,
                                     start_step=s,
                                     log_interval=args.log_interval,
                                     save_every=args.save_every,
                                     base_lr=args.lr * 0.6,
                                     decay_every=args.lr_decay_every,
                                     vocab_cap=args.vocab_cap,
                                     health_interval=args.health_interval,
                                     health_fatal=not args.no_health_stop)

            elif ph == '3':
                train_curiosity(brain, max(args.steps // 4, 50_000),
                                args.checkpoint,
                                start_step=s,
                                log_interval=max(args.log_interval // 2, 1000),
                                save_every=args.save_every,
                                base_lr=args.lr * 0.5,
                                decay_every=args.lr_decay_every,
                                health_interval=args.health_interval,
                                health_fatal=not args.no_health_stop)

            if _stop:
                print("\n  Interrupted. Run same command to resume.")
                return

    except HealthError as e:
        print(f"\n  [ABORTED] {e}")
        save_checkpoint(brain, args.checkpoint, 'failed_health')
        save_progress(args.checkpoint, 'failed_health', 0, args.steps, 0.0)
        return

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    evaluate(brain, cfg['n_dims'])
    save_checkpoint(brain, args.checkpoint, 'final')


if __name__ == "__main__":
    main()