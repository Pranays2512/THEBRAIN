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


def load_brain(checkpoint=CHECKPOINT, tag=TAG):
    d = CKPT_DIR
    def p(comp): 
        path = os.path.join(d, f"{checkpoint}_{tag}_{comp}.bin")
        return path if os.path.exists(path) else ""
    
    print(f"Loading {checkpoint}_{tag} ...")
    b = brain2.Brain(SOM_SIZE, SOM_SIZE, N_DIMS)
    b.load_components(
        p("predictor"), p("language"), p("som"), p("episodic"),
        p("emotion"), p("self"), p("symbolic"),
        p("binding"), p("bg"), p("procedures"), p("hpred")
    )
    print(f"  SOM: {b.som.rows}x{b.som.cols}, n_dims={b.som.n_dims}")
    print(f"  Language: {b.language.vocab_size:,} words")
    print(f"  Binding: {b.binding.size} triples")
    print("Ready.\n")
    return b


def predict_next(brain, enc, seq):
    """
    V3 Answering Logic:
    1. Try BindingMemory query (if it's a 2-term query like 'dog isa ?')
    2. Fallback to Predictor LSTM
    """
    # 1. Try Binding Memory for ConceptNet triples
    if len(seq) == 2:
        subj = brain.som.activation_map(enc.encode(seq[0][0]))
        rel  = brain.som.activation_map(enc.encode(seq[1][0]))
        bind_ans = np.array(brain.binding_query(subj, rel, True), dtype=np.float32)
        if np.linalg.norm(bind_ans) > 1e-5:
            return bind_ans

    # 2. Try Binding Memory for Math (if we stored (0.8*op1+0.2*op2, op, ans))
    if len(seq) == 4 and any(w in [x[0] for x in seq] for w in ["+", "-", "*", "plus", "minus", "times"]):
        a_vec = np.array(brain.som.activation_map(enc.encode(seq[0][0])), dtype=np.float32)
        b_vec = np.array(brain.som.activation_map(enc.encode(seq[2][0])), dtype=np.float32)
        subj = 0.8 * a_vec + 0.2 * b_vec
        subj_norm = np.linalg.norm(subj)
        if subj_norm > 1e-8:
            subj /= subj_norm
        
        rel  = brain.som.activation_map(enc.encode(seq[1][0]))
        bind_ans = np.array(brain.binding_query(subj.tolist(), rel, True), dtype=np.float32)
        if np.linalg.norm(bind_ans) > 1e-5:
            return bind_ans

    # 3. Fallback: Predictor N-step
    brain.predictor.reset()
    predicted = None
    brain.predictor.set_offline(True)
    for concept, word in seq:
        act = brain.som.activation_map(enc.encode(concept))
        predicted = np.array(brain.predictor.step(act), dtype=np.float32)
    brain.predictor.set_offline(False)
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


def evaluate(brain, min_freq=50):
    n_dims = brain.som.n_dims
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

    print(f"[Eval] V3 Integration  (freq≥{min_freq},  checkpoint={CHECKPOINT})")
    print("=" * 68)
    passed = 0
    for desc, seq, expected in tests:
        predicted = predict_next(brain, enc, seq)
        top5 = filtered_decode(brain.som, enc, brain.language, predicted, k=5, min_freq=min_freq)
        got  = top5[0][0] if top5 else "?"

        target_sim = 0.0
        target_vec = np.array(brain.som.activation_map(enc.encode(expected)), dtype=np.float32)
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

    brain = load_brain(args.checkpoint, args.tag)
    evaluate(brain, args.min_freq)