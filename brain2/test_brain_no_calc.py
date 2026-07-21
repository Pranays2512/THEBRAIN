#!/usr/bin/env python3
"""
test_brain_no_calc.py — Stress test with the strict math engines DISABLED.
We force the brain to guess the answers using ONLY its Neural Language Model,
just like an LLM (ChatGPT) would, to see if it actually learned the math
or if it just memorizes/hallucinates without its calculator.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

def load():
    print("Loading Neural LM...", flush=True)
    from engines.neural.neural_lm_torch import NeuralLMTorch
    lm = NeuralLMTorch.load("trained/owned_lm_auto.pt")
    print(f"  LM loaded: {lm.param_count()} params on {lm.device}")
    return lm

def run_no_calc_test(lm):
    print("\n" + "=" * 60)
    print("  Brain2 — No-Calculator Test (Pure Neural LM)")
    print("=" * 60)
    print("  The strict symbolic engines are DISABLED.")
    print("  We are forcing the Neural LM to complete the sentences.")
    print("=" * 60 + "\n")

    queries = [
        "what is the force on the rocket?",
        "the force on the rocket is",
        "rocket force is",
        "what is the momentum of the sample?",
        "sample momentum is",
        "the density of the rocket is",
    ]

    for q in queries:
        print(f"  > {q}")
        
        # We prime the language model with the query and let it generate 15 words
        # using its dist() method directly to see what it predicts
        words = q.lower().replace("?", "").split()
        import torch
        for _ in range(15):
            d = lm.dist(words)
            for sp in ("<s>", "<unk>", "<pad>"):
                d.pop(sp, None)
            vocab = list(d.keys())
            probs = torch.tensor([d[w] for w in vocab])
            nxt = vocab[torch.argmax(probs).item()] # take most likely for clear result
            if nxt == "</s>":
                break
            words.append(nxt)
            
        ans = " ".join(words[len(q.split()):]) # only show what it added
        print(f"    [neural guess] {ans}\n")

if __name__ == "__main__":
    lm = load()
    run_no_calc_test(lm)
