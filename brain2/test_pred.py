import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import brain2
from train.concept_encoder import ConceptEncoder
from train.math_sequences import MathSequenceGenerator
import numpy as np

def test_predictor():
    n_dims = 32
    enc = ConceptEncoder(n_dims)
    gen = MathSequenceGenerator(n_dims=n_dims, curriculum=1)
    
    # 32 -> 512 is the predictor spec
    pred = brain2.Predictor(32, 512, 42)
    pred.set_lr(0.01)
    
    # Let's train it purely on 2+3=5
    seq = [("2","2"),("+","plus"),("3","3"),("=","equals"), ("5","5")]
    encoded = [(enc.encode(c), w) for c, w in seq]
    
    for epoch in range(100):
        pred.reset()
        pred.set_offline(False)
        total_err = 0
        for i in range(len(encoded) - 1):
            act = encoded[i][0]
            nxt = encoded[i+1][0]
            pred.step(act, nxt)
            total_err += pred.last_error()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Err: {total_err}")
            
    # Eval
    pred.reset()
    pred.set_offline(True)
    predicted = None
    for i in range(len(encoded) - 1):
        predicted = pred.step(encoded[i][0])
        
    nxt_actual = encoded[-1][0]
    
    # Cosine sim
    pn = np.linalg.norm(predicted)
    an = np.linalg.norm(nxt_actual)
    sim = np.dot(predicted, nxt_actual) / (pn * an)
    print(f"Final sim: {sim:.4f}")

if __name__ == "__main__":
    test_predictor()
