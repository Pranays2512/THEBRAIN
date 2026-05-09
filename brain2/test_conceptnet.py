import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import brain2
import numpy as np
from train.concept_encoder import ConceptEncoder

def main():
    checkpoint = "brain_v8"
    tag = "final"
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    def p(comp): return os.path.join(ckpt_dir, f"{checkpoint}_{tag}_{comp}.bin")

    som = brain2.SOM.load(p("som"))
    pred = brain2.Predictor.load(p("predictor"))
    lang = brain2.Language.load(p("language"))
    enc = ConceptEncoder(som.n_dims)

    def predict_seq(seq_words):
        pred.reset()
        predicted = None
        for word in seq_words:
            # We don't have ConceptNet 'concept' strings easily available here, 
            # but in train.py `train_conceptnet`, it encodes the word itself if no concept.
            # Wait, ConceptNet training encodes the concept string?
            # Actually, `train_conceptnet` uses ConceptNet embeddings if available, else random.
            # But `eval_v3.py` uses `ConceptEncoder` which maps words to deterministic random vectors based on hash.
            act = som.activation_map(enc.encode(word))
            pred.set_offline(True)
            predicted = pred.step(act)
            pred.set_offline(False)
        return predicted

    tests = [
        ["dog", "isa"],
        ["cat", "isa"],
        ["bird", "isa"],
        ["car", "isa"],
        ["fire", "causes"],
        ["rain", "causes"],
        ["water", "is"],
        ["apple", "isa"]
    ]

    for test in tests:
        predicted = predict_seq(test)
        pn = np.linalg.norm(predicted)
        
        candidates = []
        for word in lang.vocab():
            if lang.frequency(word) < 50:
                continue
            vec = lang.encode(word)
            vn = np.linalg.norm(vec)
            if vn < 1e-8: continue
            sim = float(np.dot(predicted, vec) / (pn * vn))
            candidates.append((word, sim))
            
        candidates.sort(key=lambda x: -x[1])
        top = candidates[:5]
        print(f"{' '.join(test)} ? ->")
        for w, s in top:
            print(f"   {w}: {s:.3f}")
        print()

if __name__ == "__main__":
    main()
