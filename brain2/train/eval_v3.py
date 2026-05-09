"""
eval_v3.py — Evaluation with clean SOM-based decoding.

BUG FIXED: previous version used lang.encode(word) for decoding, which returns
the Hebbian-drifted language vector. High-frequency ConceptNet words like 'rainbow',
'sunlight', 'voice' drift toward the mean activation map, and the collapsed predictor
output (mean-ish vector from untrained LSTM) matched these drifted vectors at ~0.9 cosine.

THE FIX: decode by comparing predictor output to som.activation_map(enc.encode(word))
for each candidate word. These clean SOM activations are:
  - Deterministic and fixed (concept_encoder is hash-based, no drift)
  - Correctly represent each word's concept in SOM space
  - Not contaminated by diverse ConceptNet contexts

ALSO NEEDED: predictor.hpp must be rebuilt with BPTT (see predictor.hpp fix).
This eval fix alone recovers signal that was already there but obscured by bad decoding.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import brain2
import numpy as np
from concept_encoder import ConceptEncoder

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')

def latest_checkpoint():
    progress = []
    for name in os.listdir(CKPT_DIR):
        if name.endswith("_progress.json"):
            path = os.path.join(CKPT_DIR, name)
            try:
                import json
                with open(path) as f:
                    data = json.load(f)
                progress.append((data.get("ts", 0), name[:-len("_progress.json")]))
            except Exception:
                pass
    return max(progress)[1] if progress else "brain_v5"

CHECKPOINT = latest_checkpoint()
TAG        = "final"
N_DIMS     = 64
SOM_SIZE   = 64


def load_components(checkpoint=CHECKPOINT, tag=TAG):
    d = CKPT_DIR
    def p(comp): return os.path.join(d, f"{checkpoint}_{tag}_{comp}.bin")
    print(f"Loading {checkpoint}_{tag} ...")
    som  = brain2.SOM.load(p("som"))
    print(f"  SOM: {som.rows}x{som.cols}, n_dims={som.n_dims}, steps={som.step:,}")
    pred = brain2.Predictor.load(p("predictor"))
    print(f"  Predictor: input={pred.input_dim}, hidden={pred.hidden_dim}, lr={pred.lr:.6f}")
    lang = brain2.Language.load(p("language"))
    print(f"  Language: {lang.vocab_size:,} words  (n_dims={lang.n_dims})")
    print("Ready.\n")
    return som, pred, lang


def predict_next(som, pred, enc, seq):
    """Feed sequence through SOM + LSTM predictor, return predicted next activation."""
    pred.reset()
    predicted = None
    for concept, word in seq:
        act = som.activation_map(enc.encode(concept))
        pred.set_offline(True)
        predicted = pred.step(act)
        pred.set_offline(False)
    return predicted


# ── THE KEY FIX ──────────────────────────────────────────────────────────────
def filtered_decode(som, enc, lang, predicted_vec, k=5, min_freq=50):
    """
    Decode predicted activation to words.

    FIXED: uses som.activation_map(enc.encode(word)) for each candidate word
    instead of lang.encode(word). This gives a CLEAN, FIXED activation map per
    word that is immune to the Hebbian drift that contaminates language vectors
    for high-frequency words (rainbow, sunlight, voice, etc. all drifted toward
    the mean activation map and dominated every decode before this fix).
    """
    pn = np.linalg.norm(predicted_vec)
    if pn < 1e-8:
        return []

    candidates = []
    for word in lang.vocab():
        freq = lang.frequency(word)
        if freq < min_freq:
            continue
        # Clean, deterministic SOM activation — no drift, no collapse
        vec = np.array(som.activation_map(enc.encode(word)), dtype=np.float32)
        vn  = np.linalg.norm(vec)
        if vn < 1e-8:
            continue
        sim = float(np.dot(predicted_vec, vec) / (pn * vn))
        candidates.append((word, sim))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:k]
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(som, pred, lang, min_freq=50):
    n_dims = som.n_dims
    enc = ConceptEncoder(n_dims)

    tests = [
        ("2 + 3 = ?",     [("2","2"),("+","plus"),("3","3"),("=","equals")],     "5"),
        ("10 - 4 = ?",    [("10","10"),("-","minus"),("4","4"),("=","equals")],   "6"),
        ("3 * 4 = ?",     [("3","3"),("*","times"),("4","4"),("=","equals")],     "12"),
        ("! true = ?",    [("!","not"),("true","true"),("=","equals")],           "false"),
        ("4 > 2 = ?",     [("4","4"),(">","greater"),("2","2"),("=","equals")],   "true"),
        ("fire causes ?", [("fire","fire"),("causes","causes")],                  "heat"),
        ("dog isa ?",     [("dog","dog"),("isa","isa")],                          "animal"),
        ("plant needs ?", [("plant","plant"),("needs","needs")],                  "sunlight"),
    ]

    print(f"[Eval] Clean SOM decode  (freq≥{min_freq},  checkpoint={CHECKPOINT})")
    print("=" * 68)
    passed = 0
    for desc, seq, expected in tests:
        predicted = predict_next(som, pred, enc, seq)
        top5 = filtered_decode(som, enc, lang, predicted, k=5, min_freq=min_freq)
        got  = top5[0][0] if top5 else "?"

        target_sim = 0.0
        target_vec = np.array(som.activation_map(enc.encode(expected)), dtype=np.float32)
        pn = np.linalg.norm(predicted)
        tn = np.linalg.norm(target_vec)
        if pn > 1e-8 and tn > 1e-8:
            target_sim = float(np.dot(predicted, target_vec) / (pn * tn))

        ok  = (got == expected)
        passed += ok
        st  = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
        top3_str = [(w, f"{s:.3f}") for w, s in top5[:3]]
        print(f"  [{st}] {desc:<22} → got='{got}'  want='{expected}' (sim={target_sim:.3f})")
        if not ok:
            print(f"          top3: {top3_str}")

    print(f"\n  Result: {passed}/{len(tests)}")
    print()
    print("  NOTE: If score is still low, the predictor's LSTM weights were never")
    print("  updated during training (predictor.hpp only updated out_.W, not Wh/Wx).")
    print("  Rebuild with the fixed predictor.hpp and retrain from scratch.")
    return passed


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--tag",        default=TAG)
    p.add_argument("--min-freq",   type=int, default=50)
    args = p.parse_args()

    som, pred, lang = load_components(args.checkpoint, args.tag)
    evaluate(som, pred, lang, args.min_freq)