"""
M38 BREAK TEST
==============
Fixes vs M37 break_test:
  1. MULTICLASS SPLIT BUG FIXED
     Old code: block cycling meant test set only contained 1.5/2.0 Hz.
     Fix: Total time extended so test blocks cycle through all 4 frequencies.
     Verified by printing which frequencies appear in train/test sets.

  2. NOISE TEST FIXED
     Old test added noise to frequency (jitter). That tested a different
     thing than the system noise floor. 
     New test: Adds Gaussian noise DIRECTLY to the input signal amplitude.
     This tests whether PLV/entropy survive real signal degradation.

  3. FEATURE LABEL UPDATED
     'Variance' → 'Entropy' throughout (M38 change).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import M38 simulation
from neuron import (run_sim_m38, stabilization_time, block_duration,
                         transition_skip, window_seconds, ridge_alpha)

# =============================================================
# SHARED HELPERS
# =============================================================

def make_signal(freqs, block_dur=50.0, noise_level=0.0):
    """
    Cycles through freqs in blocks.
    noise_level = std of Gaussian noise added to signal AMPLITUDE (not frequency).
    This is the correct noise test — amplitude noise, not frequency jitter.
    """
    def sig(t):
        block = int(t / block_dur)
        idx = block % len(freqs)
        f = freqs[idx]
        I = np.sin(2 * np.pi * f * t)
        if noise_level > 0:
            I += noise_level * np.random.randn()  # amplitude noise
        return I, idx, f
    return sig


def make_regression_signal(freqs, block_dur=30.0):
    def sig(t):
        block = int(t / block_dur)
        idx = block % len(freqs)
        f = freqs[idx]
        I = np.sin(2 * np.pi * f * t)
        return I, f, f
    return sig


def classify_temporal(X, Y, T, block_dur=50.0, n_train_blocks=4, label=""):
    block_idx = (T / block_dur).astype(int)
    first_block = int(stabilization_time / block_dur)
    rel_block = block_idx - first_block
    train_mask = rel_block < n_train_blocks
    test_mask  = rel_block >= n_train_blocks

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test,  Y_test  = X[test_mask],  Y[test_mask]

    if len(X_test) < 5 or len(X_train) < 5:
        return {'test_acc': 0, 'per_class': {}}

    classes = np.unique(Y_train)

    # DIAGNOSTIC: Warn if test set is missing any training class
    test_classes = np.unique(Y_test)
    missing = set(classes) - set(test_classes)
    if missing and label:
        print(f"    [WARN] Test set missing classes: {missing} — extend total_time")

    if len(classes) < 2:
        return {'test_acc': 0.5, 'per_class': {}}

    # Balance training set
    min_c = min(np.sum(Y_train == c) for c in classes)
    bal_idx = []
    rng = np.random.default_rng(42)
    for c in classes:
        ci = np.where(Y_train == c)[0]
        if len(ci) > min_c: ci = rng.choice(ci, size=min_c, replace=False)
        bal_idx.extend(ci)
    X_tr = X_train[np.sort(bal_idx)]
    Y_tr = Y_train[np.sort(bal_idx)]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_test)

    n_pca = min(50, len(X_tr), X_tr_sc.shape[1])
    pca = PCA(n_components=n_pca)
    X_tr_p = pca.fit_transform(X_tr_sc)
    X_te_p = pca.transform(X_te_sc)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_tr_p, Y_tr)
    pred = model.predict(X_te_p)

    if len(classes) == 2:
        threshold = np.mean(classes)
        acc = np.mean((pred > threshold) == (Y_test > threshold))
        per_class = {}
        for c in classes:
            m = Y_test == c
            per_class[c] = np.mean((pred[m] > threshold) == (c > threshold)) if np.any(m) else 0
    else:
        pred_classes = np.array([classes[np.argmin(np.abs(p - classes))] for p in pred])
        acc = np.mean(pred_classes == Y_test)
        per_class = {}
        for c in classes:
            m = Y_test == c
            per_class[c] = np.mean(pred_classes[m] == c) if np.any(m) else 0

    return {'test_acc': acc, 'per_class': per_class}


# =============================================================
# TEST SUITE
# =============================================================
print("=" * 70)
print("  M38 BREAK TEST")
print("  Fixes: entropy feature | multiclass split | amplitude noise test")
print("=" * 70)


# --- TEST 0: BASELINE ---
print(f"\n{'='*70}")
print("  TEST 0: BASELINE (0.5 vs 2.0 Hz)")
print(f"{'='*70}")
sig = make_signal([0.5, 2.0])
plv, ent, spec, Y, T = run_sim_m38(sig)
X_comb = np.hstack([plv, ent, spec])

r_plv  = classify_temporal(plv, Y, T, label="baseline-plv")
r_ent  = classify_temporal(ent, Y, T, label="baseline-ent")
r_comb = classify_temporal(X_comb, Y, T, label="baseline-comb")
print(f"  PLV Only:     {r_plv['test_acc']*100:5.1f}%")
print(f"  Entropy Only: {r_ent['test_acc']*100:5.1f}%")
print(f"  Combined:     {r_comb['test_acc']*100:5.1f}%")


# --- TEST 1: FREQUENCY RESOLUTION ---
print(f"\n{'='*70}")
print("  TEST 1: FREQUENCY RESOLUTION")
print(f"{'='*70}")
print(f"  {'Pair':>16}  {'PLV%':>6}  {'Ent%':>6}  {'Comb%':>6}")
print(f"  {'─'*16}  {'─'*6}  {'─'*6}  {'─'*6}")

pairs = [(0.5, 2.0), (0.5, 0.8), (0.5, 0.7), (0.5, 0.6),
         (0.5, 0.55), (0.5, 0.52), (0.5, 0.51), (0.5, 0.505)]
for fa, fb in pairs:
    sig = make_signal([fa, fb])
    plv, ent, spec, Y, T = run_sim_m38(sig, verbose=False)
    r_p = classify_temporal(plv, Y, T)
    r_e = classify_temporal(ent, Y, T)
    r_c = classify_temporal(np.hstack([plv, ent, spec]), Y, T)
    print(f"  {fa:.3f} vs {fb:.3f}    {r_p['test_acc']*100:5.1f}  {r_e['test_acc']*100:5.1f}  {r_c['test_acc']*100:5.1f}")


# --- TEST 2: NOISE ROBUSTNESS (FIXED) ---
print(f"\n{'='*70}")
print("  TEST 2: NOISE ROBUSTNESS (amplitude noise on input signal)")
print(f"{'='*70}")
print(f"  {'Noise σ':>8}  {'PLV%':>6}  {'Ent%':>6}  {'Comb%':>6}")
print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}")

for noise_lvl in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
    sig = make_signal([0.5, 2.0], noise_level=noise_lvl)
    plv, ent, spec, Y, T = run_sim_m38(sig, verbose=False)
    r_p = classify_temporal(plv, Y, T)
    r_e = classify_temporal(ent, Y, T)
    r_c = classify_temporal(np.hstack([plv, ent, spec]), Y, T)
    print(f"  {noise_lvl:8.2f}  {r_p['test_acc']*100:5.1f}  {r_e['test_acc']*100:5.1f}  {r_c['test_acc']*100:5.1f}")


# --- TEST 3: MULTI-CLASS (FIXED) ---
# FIX: Use total_time=600 so test blocks cycle through all 4 frequencies.
# With block_dur=50, stabilization=120:
#   first harvest block = block 2 (1.5 Hz)
#   train blocks 0-3 = 1.5, 2.0, 0.5, 1.0  → all 4 classes in training
#   test blocks 4+   = 1.5, 2.0, 0.5, 1.0  → all 4 classes in test
# With 600s total we get ~4 train + 4 test blocks = balanced.
print(f"\n{'='*70}")
print("  TEST 3: MULTI-CLASS (4 Frequencies) — split bug fixed")
print(f"{'='*70}")
freqs_4 = [0.5, 1.0, 1.5, 2.0]
sig = make_signal(freqs_4)
plv, ent, spec, Y, T = run_sim_m38(sig, verbose=False, total_time=600.0)
r = classify_temporal(np.hstack([plv, ent, spec]), Y, T, label="multiclass")

# Show which frequencies appear in test set
block_idx = (T / block_duration).astype(int)
first_block = int(stabilization_time / block_duration)
rel_block = block_idx - first_block
test_mask = rel_block >= 4
test_freqs = np.unique(Y[test_mask])
print(f"  Frequencies in test set: {[freqs_4[int(i)] for i in test_freqs]}")
print(f"  Overall Acc: {r['test_acc']*100:.1f}%")
for c in sorted(r['per_class'].keys()):
    print(f"    Freq {freqs_4[int(c)]:.1f}Hz: {r['per_class'][c]*100:.1f}%")


# --- TEST 4: FREQUENCY REGRESSION ---
print(f"\n{'='*70}")
print("  TEST 4: FREQUENCY REGRESSION")
print(f"{'='*70}")

train_f  = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]
interp_f = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

sig_train = make_regression_signal(train_f)
plv, ent, spec, Y, T = run_sim_m38(sig_train, verbose=False, blk_dur=30.0)
X_train = np.hstack([plv, ent, spec])

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_train)
pca = PCA(n_components=50)
X_p = pca.fit_transform(X_sc)
model = Ridge(alpha=ridge_alpha)
model.fit(X_p, Y)

pred_train = model.predict(X_p)
mae_train = np.mean(np.abs(pred_train - Y))
print(f"  Train MAE: {mae_train:.4f} Hz")

sig_interp = make_regression_signal(interp_f)
plv_i, ent_i, spec_i, Y_i, T_i = run_sim_m38(sig_interp, verbose=False, blk_dur=30.0)
X_i = np.hstack([plv_i, ent_i, spec_i])
pred_i = model.predict(pca.transform(scaler.transform(X_i)))
mae_i = np.mean(np.abs(pred_i - Y_i))
print(f"  Interp MAE: {mae_i:.4f} Hz")

print(f"\n  {'Actual':>6}  {'Pred':>6}  {'Err':>6}")
for f in sorted(set(Y_i)):
    m = Y_i == f
    if np.any(m):
        p_mean = np.mean(pred_i[m])
        err = abs(p_mean - f)
        print(f"  {f:6.2f}  {p_mean:6.2f}  {err:6.3f}")


# --- SUMMARY ---
print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")
print(f"  Linear Limit:  {1/window_seconds:.2f} Hz")
print(f"  M38 Interp:    {mae_i:.4f} Hz")
print(f"  Amplification: {(1/window_seconds)/mae_i:.1f}x")
print(f"\n  Feature set: PLV + Energy Entropy + Spectral")
print(f"  (Variance removed — collapses under noise)")