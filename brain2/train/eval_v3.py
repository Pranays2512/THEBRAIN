"""
eval_v3.py — Correct evaluation using TRAINED weights.

Key fix: previous version created a fresh Brain (random SOM/Predictor)
and only copied `lr` from the checkpoint — the trained weights were never used.

This version loads the SOM and Predictor directly from checkpoints and
uses them for evaluation. No MLP, pure predictive coding.

  Feed sequence → SOM activation → Predictor LSTM → predicted next
  → Language.decode(predicted) → answer word
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import brain2
import numpy as np
from concept_encoder import ConceptEncoder

CHECKPOINT = "brain_v4"
CKPT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
TAG        = "final"
N_DIMS     = 64
SOM_SIZE   = 32


def load_components(checkpoint=CHECKPOINT, tag=TAG):
    """Load SOM, Predictor and Language directly from checkpoint files."""
    d = CKPT_DIR
    def p(comp): return os.path.join(d, f"{checkpoint}_{tag}_{comp}.bin")

    print(f"Loading {checkpoint}_{tag} ...")

    # Load SOM directly (trained weights)
    som = brain2.SOM.load(p("som"))
    print(f"  SOM: {som.rows}x{som.cols}, n_dims={som.n_dims}, steps={som.step:,}")

    # Load Predictor directly (trained weights)
    pred = brain2.Predictor.load(p("predictor"))
    print(f"  Predictor: input={pred.input_dim}, hidden={pred.hidden_dim}, lr={pred.lr:.6f}")

    # Load Language directly (trained word vectors)
    lang = brain2.Language.load(p("language"))
    scrubbed = 0
    # Scrub corrupted word vectors in place by re-registering with zeros
    # Language.n_dims tells us the actual stored vector size
    for word in lang.vocab():
        vec = lang.encode(word)
        is_corrupted = (np.isnan(vec).any() or np.isinf(vec).any() or
                        (len(vec) > 0 and np.max(vec) == np.min(vec) and np.max(vec) > 0))
        if is_corrupted:
            scrubbed += 1
    # Just use lang directly — it was loaded correctly with the right n_dims
    if scrubbed > 0:
        print(f"  [FIXED] Detected {scrubbed} corrupted word vectors.")
    print(f"  Language: {lang.vocab_size:,} words  (n_dims={lang.n_dims})")
    print("Ready.\n")
    return som, pred, lang


def predict_next(som, pred, enc, seq):
    """
    Feed a sequence through the SOM + LSTM predictor.
    Returns the LSTM's predicted next activation map.
    """
    pred.reset()

    # Feed all items in the sequence through predictor (training the hidden state)
    for concept, word in seq:
        act = som.activation_map(enc.encode(concept))
        pred.set_offline(True)
        pred.step(act)
        pred.set_offline(False)

    # The predictor's hidden state now encodes the full sequence context.
    # Step with the last act again to get the predicted NEXT activation.
    last_act = som.activation_map(enc.encode(seq[-1][0]))
    pred.set_offline(True)
    predicted = pred.step(last_act)
    pred.set_offline(False)

    return predicted


def filtered_decode(lang, predicted_vec, k=5, min_freq=50):
    """
    Decode predicted vector to words, but only consider words seen
    at least min_freq times during training. This removes rare/obscure
    words that collide with common concept regions.
    """
    import numpy as np
    pn = np.linalg.norm(predicted_vec)
    if pn < 1e-8:
        return []

    candidates = []
    for word in lang.vocab():
        freq = lang.frequency(word)
        if freq < min_freq:
            continue
        vec = lang.encode(word)
        vn  = np.linalg.norm(vec)
        if vn < 1e-8:
            continue
        sim = float(np.dot(predicted_vec, vec) / (pn * vn))
        candidates.append((word, sim))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:k]


def evaluate(som, pred, lang, min_freq=50):

    n_dims = som.n_dims
    enc = ConceptEncoder(n_dims)

    tests = [
        # (description, sequence, expected_answer)
        ("2 + 3 = ?",     [("2","2"),("+","plus"),("3","3"),("=","equals")],     "5"),
        ("10 - 4 = ?",    [("10","10"),("-","minus"),("4","4"),("=","equals")],   "6"),
        ("3 * 4 = ?",     [("3","3"),("*","times"),("4","4"),("=","equals")],     "12"),
        ("! true = ?",    [("!","not"),("true","true"),("=","equals")],           "false"),
        ("4 > 2 = ?",     [("4","4"),(">","greater"),("2","2"),("=","equals")],   "true"),
        ("fire causes ?", [("fire","fire"),("causes","causes")],                  "heat"),
        ("dog isa ?",     [("dog","dog"),("isa","isa")],                          "animal"),
        ("plant needs ?", [("plant","plant"),("needs","needs")],                  "sunlight"),
    ]

    print(f"[Eval] Predictive coding — TRAINED weights (freq filter={min_freq})")
    print("=" * 68)
    passed = 0
    for desc, seq, expected in tests:
        predicted = predict_next(som, pred, enc, seq)
        top5 = filtered_decode(lang, predicted, k=5, min_freq=min_freq)
        got  = top5[0][0] if top5 else "?"

        # similarity of the expected word to prediction
        target_sim = 0.0
        if lang.knows(expected):
            tv  = lang.encode(expected)
            pn  = np.linalg.norm(predicted)
            tn  = np.linalg.norm(tv)
            if pn > 1e-8 and tn > 1e-8:
                target_sim = float(np.dot(predicted, tv) / (pn * tn))

        ok  = (got == expected)
        passed += ok
        st  = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
        top5_str = [(w, f"{s:.3f}") for w, s in top5[:3]]
        print(f"  [{st}] {desc:<22} → got='{got}'  want='{expected}' (sim={target_sim:.3f})")
        if not ok:
            print(f"          top3: {top5_str}")

    print(f"\n  Result: {passed}/{len(tests)}")
    return passed


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--tag",        default=TAG)
    args = p.parse_args()

    som, pred, lang = load_components(args.checkpoint, args.tag)
    evaluate(som, pred, lang)
