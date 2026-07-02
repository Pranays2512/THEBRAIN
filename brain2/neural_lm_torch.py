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
        return ["<s>"] + re.findall(r"[a-z]+", line.lower()) + ["</s>"]

    def train(self, corpus):
        words = sorted({w for ln in corpus for w in self._tok(ln)} | {"<unk>", "<pad>"})
        self.w2i = {w: i for i, w in enumerate(words)}
        self.i2w = words
        pad = self.w2i["<pad>"]
        seqs = [[self.w2i[w] for w in self._tok(ln)] for ln in corpus]
        # sliding windows -> (context, next)
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
        lossf = nn.CrossEntropyLoss()
        self.model.train()
        n = X.size(0)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch):          # MINI-BATCH: caps memory regardless of corpus
                idx = perm[i:i + self.batch]
                logits = self.model(X[idx])[:, -1, :]  # predict next from last position
                loss = lossf(logits, Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model.eval()
        self._pad = pad
        return self

    @torch.no_grad()
    def dist(self, context):
        ids = [self.w2i.get(w, self.w2i["<unk>"]) for w in (["<s>"] + context)]
        ids = ids[-self.ctx:]
        ids = [self._pad] * (self.ctx - len(ids)) + ids
        x = torch.tensor([ids], device=self.device)
        p = torch.softmax(self.model(x)[0, -1], dim=-1).tolist()
        return {self.i2w[i]: p[i] for i in range(len(p))}

    def generate(self, max_len=14, seed=0):
        torch.manual_seed(seed)
        out = []
        for _ in range(max_len):
            d = self.dist(out)
            for sp in ("<s>", "<unk>", "<pad>"):
                d.pop(sp, None)
            words = list(d)
            probs = torch.tensor([d[w] for w in words])
            nxt = words[torch.multinomial(probs / probs.sum(), 1).item()]
            if nxt == "</s>":
                break
            out.append(nxt)
        return out

    def param_count(self):
        return sum(p.numel() for p in self.model.parameters())


def _demo():
    import context_embed as CE
    try:
        import corpus_scale as CS
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
