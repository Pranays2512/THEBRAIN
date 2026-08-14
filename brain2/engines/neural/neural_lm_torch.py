#!/usr/bin/env python3
"""
neural_lm_torch.py — the REAL owned LM for Mac training: a small decoder-only Transformer on
Apple-Silicon GPU (MPS). Same interface as neural_lm (train/generate/dist) so it drops into
train_pipeline as the student; scales far past the numpy proof.

Word-level, causal Transformer. Config (dim/layers/heads/ctx) tiny by default, Mac-trainable;
raise for a real run. device = mps (Apple GPU) if available, else cpu. Given the symbolic
core offloads facts + reasoning, a SMALL LM here (10-100M params) is the practical target and
is Mac-feasible.

Requires PyTorch:  pip install torch    (MPS ships with the standard macOS wheel)

    venv2/bin/python3 neural_lm_torch.py
"""

import re

import torch
import torch.nn as nn


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class _LM(nn.Module):
    def __init__(self, vocab, dim, heads, layers, ctx):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(ctx, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        T = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


class NeuralLMTorch:
    def __init__(self, dim=128, heads=4, layers=2, ctx=32, lr=3e-4, epochs=30, batch=64, seed=0):
        self.dim, self.heads, self.layers, self.ctx = dim, heads, layers, ctx
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self.device = _device()
        torch.manual_seed(seed)

    def _tok(self, line):
        # keep parsing structure (words, numbers, => | : = operators) so the student can
        # learn text -> structured form, not just plain words.
        toks = re.findall(r"[a-zA-Z_]+|\d+\.?\d*|=>|[|:=()+\-*/]", line.lower())
        return ["<s>"] + toks + ["</s>"]

    def train(self, corpus, ckpt_path="trained/lm_checkpoint.pt"):
        import os
        import sys
        import time as _time
        
        # Load checkpoint if exists to resume
        if os.path.exists(ckpt_path):
            d = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.w2i, self.i2w, self._pad = d["w2i"], d["i2w"], d["pad"]
            start_ep, start_step = d.get("ep", 0), d.get("step", 0)
            words = self.i2w
            print(f"    [LM] Resuming from {ckpt_path} (Epoch {start_ep+1}, Step {start_step})")
        else:
            words = sorted({w for ln in corpus for w in self._tok(ln)} | {"<unk>", "<pad>"})
            self.w2i = {w: i for i, w in enumerate(words)}
            self.i2w = words
            self._pad = self.w2i["<pad>"]
            start_ep, start_step = 0, 0

        pad = self._pad
        seqs = [[self.w2i.get(w, self.w2i["<unk>"]) for w in self._tok(ln)] for ln in corpus]
        X, Y = [], []
        for s in seqs:
            for i in range(1, len(s)):
                ctx = s[max(0, i - self.ctx):i]
                ctx = [pad] * (self.ctx - len(ctx)) + ctx
                X.append(ctx); Y.append(s[i])
                
        X = torch.tensor(X, device=self.device)
        Y = torch.tensor(Y, device=self.device)
        
        self.model = _LM(len(words), self.dim, self.heads, self.layers, self.ctx).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(d["state"])
            opt.load_state_dict(d["opt"])
            
        lossf = nn.CrossEntropyLoss()
        self.model.train()
        n = X.size(0)
        _t0 = _time.time()
        
        steps = list(range(0, n, self.batch))
        total_steps = len(steps)
        total_iters = self.epochs * total_steps
        iters_done = (start_ep * total_steps) + start_step
        
        try:
            for ep in range(start_ep, self.epochs):
                # Ensure deterministic permutation per epoch based on epoch index
                torch.manual_seed(ep)
                perm = torch.randperm(n, device=self.device)
                _el = 0.0
                
                step_start_idx = start_step if ep == start_ep else 0
                
                for step_idx in range(step_start_idx, total_steps):
                    i = steps[step_idx]
                    idx = perm[i:i + self.batch]
                    logits = self.model(X[idx])[:, -1, :]
                    loss = lossf(logits, Y[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    _el = loss.item()
                    iters_done += 1
                    
                    if step_idx % 10 == 0 or step_idx == total_steps - 1:
                        elapsed = _time.time() - _t0
                        recent_iters = iters_done - ((start_ep * total_steps) + start_step)
                        if recent_iters > 0:
                            rate = elapsed / recent_iters
                            rem = (total_iters - iters_done) * rate
                            eta_str = f"{int(rem//60)}m {int(rem%60)}s"
                        else:
                            eta_str = "..."
                            
                        pct = 100 * iters_done / total_iters
                        bar_len = 30
                        filled = int(bar_len * pct // 100)
                        bar = '█' * filled + '-' * (bar_len - filled)
                        sys.stdout.write(f"\r    [LM] |{bar}| {pct:.1f}%  Epoch {ep+1}/{self.epochs}  Loss {_el:.3f}  ETA: {eta_str}   ")
                        sys.stdout.flush()
                        
                # Epoch complete, save checkpoint
                os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
                torch.save({"state": self.model.state_dict(), "opt": opt.state_dict(), "w2i": self.w2i, "i2w": self.i2w,
                            "pad": self._pad, "ep": ep + 1, "step": 0}, ckpt_path)
                start_step = 0
                
        except KeyboardInterrupt:
            print(f"\n    [LM] Paused by user. Saving checkpoint to {ckpt_path}...")
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save({"state": self.model.state_dict(), "opt": opt.state_dict(), "w2i": self.w2i, "i2w": self.i2w,
                        "pad": self._pad, "ep": ep, "step": step_idx}, ckpt_path)
            print("    [LM] Exiting safely.")
            sys.exit(0)
            
        print()
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)  # Cleanup checkpoint when fully done
        self.model.eval()
        return self

    @torch.no_grad()
    def dist(self, context):
        ids = [self.w2i.get(w, self.w2i["<unk>"]) for w in (["<s>"] + context)]
        ids = ids[-self.ctx:]
        ids = [self._pad] * (self.ctx - len(ids)) + ids
        x = torch.tensor([ids], device=self.device)
        p = torch.softmax(self.model(x)[0, -1], dim=-1).tolist()
        return {self.i2w[i]: p[i] for i in range(len(p))}

    def generate(self, prompt=None, max_len=14, seed=0):
        torch.manual_seed(seed)
        out = []
        if prompt:
            out.extend(prompt)
        for _ in range(max_len):
            d = self.dist(out)
            for sp in ("<s>", "<unk>", "<pad>"):
                d.pop(sp, None)
            words = list(d)
            if not words:
                break
            probs = torch.tensor([d[w] for w in words])
            nxt = words[torch.multinomial(probs / probs.sum(), 1).item()]
            if nxt == "</s>":
                break
            out.append(nxt)
        return out

    def param_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def save(self, path):
        torch.save({"state": self.model.state_dict(), "w2i": self.w2i, "i2w": self.i2w,
                    "cfg": (self.dim, self.heads, self.layers, self.ctx), "pad": self._pad}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location=_device(), weights_only=True)  # only tensors + simple data
        lm = cls()
        lm.dim, lm.heads, lm.layers, lm.ctx = d["cfg"]
        lm.w2i, lm.i2w, lm._pad = d["w2i"], d["i2w"], d["pad"]
        lm.model = _LM(len(lm.i2w), lm.dim, lm.heads, lm.layers, lm.ctx).to(lm.device)
        lm.model.load_state_dict(d["state"])
        lm.model.eval()
        return lm


def _demo():
    from engines.grounding import context_embed as CE
    try:
        from engines.store import corpus_scale as CS
        corpus = CS.LARGE
    except Exception:
        corpus = CE.CORPUS
    print("=== neural_lm_torch — small Transformer LM on Apple-Silicon GPU (MPS) ===\n")
    lm = NeuralLMTorch(epochs=40).train(corpus)
    print("  device: %s | params: %d | vocab: %d" % (lm.device, lm.param_count(), len(lm.w2i)))
    print("  generate: " + " ".join(lm.generate(seed=0)))
    print("  generate: " + " ".join(lm.generate(seed=1)))
    print("\n  Real Transformer student, Mac-GPU trainable. Scale dim/layers/ctx + corpus for a")
    print("  fluent owned LM. Distill from qwen-coder via train_pipeline --real.")


if __name__ == "__main__":
    _demo()
