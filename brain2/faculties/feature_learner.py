#!/usr/bin/env python3
"""
feature_learner.py — the proposer DISCOVERS which task signals matter (no hand-picked features).

online_proposer2 keyed on a hand-picked signature (out_exceeds_max, neg_in, ...). This removes
that last hand-coded piece: generate a BROAD pool of cheap candidate signals, then LEARN which
ones predict the winning space — from outcomes alone. The predictive features earn weight; the
noise features are ignored. The proposer routes by the signals it discovered itself.

  gen_features : ~16 cheap auto-signals over a task's examples (most are noise)
  per space    : keep a running feature PROFILE of the tasks that space won
  discriminate : a feature's weight = how much it VARIES across space profiles (separates them)
  route        : order spaces by weighted similarity of the task to each space's profile;
                 reward the winner -> its profile sharpens, the useful features surface.

Measured vs the static order: fewer attempts on a mixed workload, AND it reports which
features it FOUND predictive (the discriminative ones emerge with no human naming them).

Honest limit: features are drawn from a fixed generator pool; inventing brand-new signal
TYPES is the deeper rung. But which signals MATTER is now learned, not told.
"""

import random

from engines.synthesis import synth_engine as SE
from engines.synthesis.online_proposer import fixed_solve


def _flat(a):
    out = []
    for x in a:
        out += list(x) if isinstance(x, (list, tuple)) else [x]
    return [v for v in out if isinstance(v, (int, float))]


def gen_features(examples, kind):
    """A broad pool of cheap 0/1 signals. Most are noise; the learner finds the useful ones."""
    args = [a for a, _ in examples]
    ys = [y for _, y in examples]
    nin = [v for a in args for v in _flat(a)]
    nout = [y for y in ys if isinstance(y, (int, float))]
    f = {}
    f["kind_scalar"]    = kind in ("int1", "int2")
    f["kind_list"]      = kind in ("list", "listt")
    f["out_is_list"]    = all(isinstance(y, list) for y in ys)
    f["any_neg_in"]     = any(v < 0 for v in nin)
    f["any_neg_out"]    = any(v < 0 for v in nout)
    f["all_pos_out"]    = bool(nout) and all(v > 0 for v in nout)
    f["out_exceeds_max"] = any(_flat(a) and isinstance(y, (int, float)) and y > max(_flat(a))
                               for a, y in examples)
    f["out_below_min"]  = any(_flat(a) and isinstance(y, (int, float)) and y < min(_flat(a))
                              for a, y in examples)
    f["out_eq_len"]     = any(isinstance(a[0], (list, tuple)) and y == len(a[0])
                              for a, y in examples if a)
    f["big_inputs"]     = bool(nin) and max(abs(v) for v in nin) > 5
    f["mean_out_gt_in"] = bool(nout) and bool(nin) and (sum(nout) / len(nout)) > (sum(nin) / len(nin))
    f["zero_in_out"]    = 0 in nout
    f["out_even_all"]   = bool(nout) and all(int(v) % 2 == 0 for v in nout)
    f["small_data"]     = len(examples) <= 4
    f["one_arg"]        = bool(args) and (not isinstance(args[0], (list, tuple)) or len(args[0]) == 1)
    f["two_args"]       = bool(args) and isinstance(args[0], (list, tuple)) and len(args[0]) == 2
    return {k: 1.0 if v else 0.0 for k, v in f.items()}


class LearnedProposer:
    def __init__(self):
        self.proto = {}      # space -> {feat: mean}, accumulated over wins
        self.count = {}

    def _disc_weights(self):
        """A feature's weight = its variance across space profiles (how well it separates them)."""
        if len(self.proto) < 2:
            return None
        feats = next(iter(self.proto.values())).keys()
        w = {}
        for fk in feats:
            vals = [self.proto[s][fk] for s in self.proto]
            mu = sum(vals) / len(vals)
            w[fk] = sum((v - mu) ** 2 for v in vals) / len(vals)
        tot = sum(w.values()) or 1.0
        return {k: v / tot for k, v in w.items()}

    def _sim(self, feats, proto, w):
        num = den = 0.0
        for fk, fv in feats.items():
            wt = (w[fk] if w else 1.0)
            num += wt * (1.0 - abs(fv - proto.get(fk, 0.0)))
            den += wt
        return num / (den or 1.0)

    def solve(self, examples, kind, oracle=None):
        feats = gen_features(examples, kind)
        backs = SE.ROUTES.get(kind, [])
        w = self._disc_weights()
        known = [(self._sim(feats, self.proto[n], w), n, b) for n, b in backs if n in self.proto]
        unknown = [(0.5, n, b) for n, b in backs if n not in self.proto]   # explore cold spaces
        order = [(n, b) for _, n, b in sorted(known, reverse=True)] + [(n, b) for _, n, b in unknown]
        attempts = 0
        for name, backend in order:
            attempts += 1
            try:
                code = backend(examples)
            except Exception:
                code = None
            if code and SE._verify(code, examples) and (not oracle or SE.stress(code, oracle, kind)[0]):
                self._update(name, feats)
                return name, code, attempts
        return None, None, attempts

    def _update(self, name, feats):
        c = self.count.get(name, 0)
        p = self.proto.setdefault(name, {k: 0.0 for k in feats})
        for k, v in feats.items():
            p[k] = (p[k] * c + v) / (c + 1)          # running mean of winning-task features
        self.count[name] = c + 1


def _demo():
    rng = random.Random(2)
    msub = lambda a: max(sum(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a)))

    def rlist(n=6):
        return [rng.randint(-9, 9) for _ in range(n)]

    workload = []
    for _ in range(6):
        workload.append(("subarray", "list", msub, [rlist() for _ in range(5)]))
    for _ in range(6):
        workload.append(("sort", "list", sorted, [rlist() for _ in range(5)]))
    rng.shuffle(workload)

    print("=== feature_learner — the proposer discovers which signals matter ===\n")
    lp = LearnedProposer()
    st = lt = 0
    print("  task       static  learned   (attempts)")
    for name, kind, oracle, ins in workload:
        ex = SE._ex(kind, oracle, ins)
        _, _, sa = fixed_solve(ex, kind)
        _, _, la = lp.solve(ex, kind, oracle)
        st += sa; lt += la
        print("  %-9s %5d %8d" % (name, sa, la))
    print("\n  total attempts:  static %d   learned %d  (%.0f%% fewer)"
          % (st, lt, (1 - lt / st) * 100))
    w = lp._disc_weights() or {}
    top = sorted(w.items(), key=lambda x: x[1], reverse=True)[:4]
    print("\n  features the proposer DISCOVERED as predictive (highest discriminative weight):")
    for fk, wt in top:
        print("    %-16s weight %.2f" % (fk, wt))
    print("  None of these were hand-selected — the proposer found them from outcomes.")


if __name__ == "__main__":
    _demo()
