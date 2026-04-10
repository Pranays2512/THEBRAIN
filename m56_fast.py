"""
M56_FAST — Numba-compiled kernels for Actor-Critic hot path.

Called ~500k times per test run. Each function is JIT-compiled to native
code on first call (cached to __pycache__ after that).

Falls back to pure numpy if numba is not installed — import still works,
just slower. Install with: pip install numba
"""

import numpy as np

try:
    from numba import njit as _njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def _njit(**kw):
        def decorator(f):
            return f
        return decorator

# ──────────────────────────────────────────────────────────────────────────
# TRACE + POLICY GRADIENT UPDATE
# ──────────────────────────────────────────────────────────────────────────

@_njit(cache=True)
def decay_and_pg_trace(e_trace, prev_bmu, prev_action, prev_probs, trace_decay):
    """
    Decay the entire eligibility trace, then accumulate the policy-gradient
    score function into the prev_bmu row:

        e[prev_bmu] += one_hot(prev_action) - π(prev_action | prev_state)

    This replaces the old "stamp 1.0 at prev_action" with the proper gradient.
    Self-normalizing: when π(a) → 1, the gradient → 0 → no saturation.
    Competing actions get negative gradient → pushed down → differentiates.
    """
    n_bmu = e_trace.shape[0]
    n_act = e_trace.shape[1]

    # Decay entire trace
    for b in range(n_bmu):
        for a in range(n_act):
            e_trace[b, a] *= (1.0 - trace_decay)

    # Add PG score function for prev_bmu row
    for a in range(n_act):
        g = -prev_probs[a]
        if a == prev_action:
            g += 1.0
        val = e_trace[prev_bmu, a] + g
        if val > 3.0:
            val = 3.0
        elif val < -3.0:
            val = -3.0
        e_trace[prev_bmu, a] = val


@_njit(cache=True)
def update_all_pi_bmu(pi_bmu, e_trace, rpe, alpha, pi_min, pi_max):
    """
    TD(λ) actor update: π[b, a] += α × δ × e[b, a] for all (b, a).
    Non-visited BMUs have near-zero trace (decayed), so their updates are tiny.
    Clip to [pi_min, pi_max] in place.
    """
    n_bmu = pi_bmu.shape[0]
    n_act = pi_bmu.shape[1]
    for b in range(n_bmu):
        for a in range(n_act):
            val = pi_bmu[b, a] + alpha * rpe * e_trace[b, a]
            if val > pi_max:
                val = pi_max
            elif val < pi_min:
                val = pi_min
            pi_bmu[b, a] = val


# ──────────────────────────────────────────────────────────────────────────
# SOFTMAX + ACTION SELECTION
# ──────────────────────────────────────────────────────────────────────────

@_njit(cache=True)
def compute_softmax(pi_eff, temp):
    """
    Numerically stable softmax over a small float32 array.
    Returns float32 probability vector summing to 1.0.
    """
    n = pi_eff.shape[0]
    t = temp
    if t < 1e-6:
        t = 1e-6

    max_l = pi_eff[0] / t
    for i in range(1, n):
        v = pi_eff[i] / t
        if v > max_l:
            max_l = v

    probs = np.empty(n, dtype=np.float32)
    total = 0.0
    for i in range(n):
        probs[i] = np.exp(pi_eff[i] / t - max_l)
        total += probs[i]
    for i in range(n):
        probs[i] /= total
    return probs


@_njit(cache=True)
def sample_action(probs, rand_val):
    """
    Sample an action index from a probability vector using a pre-drawn
    uniform random value (avoids np.random.choice overhead and array copying).
    """
    cum = 0.0
    n = probs.shape[0]
    for i in range(n):
        cum += probs[i]
        if rand_val <= cum:
            return i
    return n - 1


# ──────────────────────────────────────────────────────────────────────────
# COMPETITIVE INHIBITION (REPLAY + DIRECT UPDATES)
# ──────────────────────────────────────────────────────────────────────────

@_njit(cache=True)
def competitive_update(pi_row, act, credit, pi_min, pi_max):
    """
    Zero-sum policy update using fixed (n-1)/n ratio.
    DEPRECATED: use pg_replay_update which self-normalizes via actual probs.
    Kept for backward compat.
    """
    n = pi_row.shape[0]
    for a in range(n):
        if a == act:
            val = pi_row[a] + credit * (n - 1) / n
        else:
            val = pi_row[a] - credit / n
        if val > pi_max:
            val = pi_max
        elif val < pi_min:
            val = pi_min
        pi_row[a] = val


@_njit(cache=True)
def pg_replay_update(pi_row, act, credit, temp, pi_min, pi_max):
    """
    PG-based replay update — scales by ACTUAL current probabilities.

    Same formula as the online step() update:
        grad = one_hot(act) - softmax(π / temp)
        π += credit × grad

    Self-normalizing: when π[act] is already dominant (prob → 1),
    the gradient for act → (1-1) = 0 → update stops automatically.
    Other directions get -prob[b] × credit — small when they're already low.

    Contrast with competitive_update which always applies ±fixed ratio
    regardless of current policy state — too aggressive for replay.
    """
    n = pi_row.shape[0]
    # Compute current softmax
    t = temp
    if t < 1e-6:
        t = 1e-6
    max_l = pi_row[0] / t
    for i in range(1, n):
        v = pi_row[i] / t
        if v > max_l:
            max_l = v
    total = 0.0
    probs = np.empty(n, dtype=np.float32)
    for i in range(n):
        probs[i] = np.exp(pi_row[i] / t - max_l)
        total += probs[i]
    for i in range(n):
        probs[i] /= total
    # PG gradient update
    for a in range(n):
        g = -probs[a]
        if a == act:
            g += 1.0
        val = pi_row[a] + credit * g
        if val > pi_max:
            val = pi_max
        elif val < pi_min:
            val = pi_min
        pi_row[a] = val
