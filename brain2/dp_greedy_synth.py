#!/usr/bin/env python3
"""
dp_greedy_synth.py — DP + greedy synthesis, gated by STRESS vs a brute-force oracle.

Toward "the brain writes the exotic algorithms too" — honestly. Two truths:

  1. The brain CAN search DP/greedy templates and find the recurrence/strategy.
  2. Fitting a few examples is NOT proof. DP recurrences overfit; greedy rules that
     work on small cases are often WRONG in general. So the verifier here is
     stronger: run the candidate against a BRUTE-FORCE oracle on hundreds of random
     inputs. Admit only what matches everywhere tried; REJECT greedy that's a trap.

  DP   (max subarray): search cur/best updates -> stress vs brute -> finds Kadane.
  GREEDY (coin change): fixed greedy strategy -> stress vs DP-optimal oracle ->
         ADMITTED for canonical coins, REJECTED for {1,3,4} (greedy isn't optimal).

The point: the brain proposes the clever algorithm; the brute-force oracle decides
whether it's actually correct — so a synthesized "exotic" algorithm is trustworthy
only when it survives the stronger gate, and the gate honestly fails the traps.

    python3 dp_greedy_synth.py
"""

import random

# ── DP: max-subarray (search the 1D recurrence; oracle = brute force) ────────
CUR = {
    "max(x, cur + x)": lambda cur, x: max(x, cur + x),     # Kadane
    "cur + x":         lambda cur, x: cur + x,
    "max(cur, x)":     lambda cur, x: max(cur, x),
}
BEST = {"max(best, cur)": lambda best, cur: max(best, cur),
        "best + cur":     lambda best, cur: best + cur}


def run_dp(curf, bestf, arr):
    best = cur = arr[0]
    for x in arr[1:]:
        cur = curf(cur, x)
        best = bestf(best, cur)
    return best


def brute_max_subarray(arr):
    return max(sum(arr[i:j + 1]) for i in range(len(arr)) for j in range(i, len(arr)))


def synth_dp(rng, trials=400):
    pos = [[rng.randint(-9, 9) for _ in range(rng.randint(1, 7))] for _ in range(trials)]
    for cc, cf in CUR.items():
        for bc, bf in BEST.items():
            if all(run_dp(cf, bf, a) == brute_max_subarray(a) for a in pos):
                return dict(cur=cc, best=bc)
    return None


# ── GREEDY: coin change (fixed strategy; oracle = DP-optimal) ────────────────
def greedy_coins(coins, amount):
    n, amt = 0, amount
    for c in sorted(coins, reverse=True):
        while amt >= c:
            amt -= c
            n += 1
    return n if amt == 0 else -1


def dp_min_coins(coins, amount):
    INF = float("inf")
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] < INF else -1


def stress_greedy(coins, rng, trials=300):
    """Greedy == DP-optimal on every random amount? Return a counterexample or None."""
    for _ in range(trials):
        amt = rng.randint(1, 60)
        if greedy_coins(coins, amt) != dp_min_coins(coins, amt):
            return amt
    return None


def _demo():
    rng = random.Random(0)
    print("=== dp_greedy_synth — synthesize DP/greedy, gate by stress vs brute ===\n")

    # DP: discover the max-subarray recurrence, verified vs brute force
    dp = synth_dp(rng)
    if dp:
        print(f"  DP max-subarray: cur = {dp['cur']}; best = {dp['best']}")
        print("      -> matches brute force on 400 random arrays  [VERIFIED ✓ — Kadane]\n")
    else:
        print("  DP max-subarray: no recurrence survived the brute-force gate\n")

    # GREEDY: same strategy, two coin systems — stress decides
    print("  greedy coin-change (sort desc, take largest), gated vs DP-optimal:")
    for coins in ([1, 5, 10, 25], [1, 3, 4]):
        ce = stress_greedy(coins, random.Random(1))
        if ce is None:
            print(f"    coins {coins}: ADMITTED ✓ (greedy optimal on 300 random amounts)")
        else:
            g, o = greedy_coins(coins, ce), dp_min_coins(coins, ce)
            print(f"    coins {coins}: REJECTED ✗ — amount {ce}: greedy {g} coins, "
                  f"optimal {o}. Greedy looked right, ISN'T.")
    print("\n  the brain proposes the clever algorithm; the brute-force oracle decides.")
    print("  DP that survives is trustworthy; greedy that's a trap is caught — examples")
    print("  alone would have shipped the wrong greedy.")


if __name__ == "__main__":
    _demo()
