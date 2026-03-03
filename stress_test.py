"""
BRAIN STRESS TEST SUITE
========================
Maps the dynamical limits of the attractor encoder:
  1. Settling Time Sweep  — minimum transition_skip for ~100% accuracy
  2. Frequency Resolution — smallest separable frequency difference
  3. Multi-Class Capacity  — 4 simultaneous frequencies
  4. Noise Robustness      — accuracy vs input noise level
  5. Real-Time Detection   — classification from short windows

All experiments use the same physics as M33.
Results saved to stress_test_results.png / .txt
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time as clock

# =============================================================
# SHARED PHYSICS PARAMETERS (identical to neuron.py M33)
# =============================================================
N = 500
lam = 0.8
gamma = 0.5
eps = 1e-6
dt = 0.05
target_energy = 2.5
input_gain = 1.5
eta_xi_up = 0.005
eta_xi_down = 0.002
xi_min = 0.1
xi_max = 3.0
tau_adapt = 1.0
kappa_adapt = 0.5
adapt_max = 2.0
alpha_base = 0.1
alpha_max = 0.3
target_lyap = 0.1
eta_alpha = 0.0005
lyap_window = 100
S_global = 1.0
learning_end_time = 100.0
learn_interval = 20
eta_hebb = 0.002
decay_hebb = 0.0001
noise_amp = 0.05
stabilization_time = 120.0
energy_gate = 0.5
pca_dims = 50
ridge_alpha = 1000.0
density = 0.02


def build_network(seed=None):
    """Build reservoir network (deterministic with seed)."""
    if seed is not None:
        np.random.seed(seed)
    W_real = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W_imag = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W = (W_real + 1j * W_imag)

    # Normalize spectral radius
    try:
        W_csr = W.tocsr()
        eigenvals = sp.linalg.eigs(W_csr, k=1, return_eigenvectors=False)
        max_eigen = np.abs(eigenvals[0])
        if max_eigen > 0:
            W = W * (0.9 / max_eigen)
    except:
        pass

    # Input weights
    np.random.seed(42)
    W_in = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.5

    # Diffusion Laplacian
    A_temp = sp.random(N, N, density=density, format='csr')
    A_temp = (A_temp + A_temp.T) * 0.5
    degrees = np.array(A_temp.sum(axis=1)).flatten()
    D_mat = sp.diags(degrees)
    Delta = D_mat - A_temp

    return W, W_in, Delta


def get_derivative(Psi_curr, xi_curr, adapt_curr, alpha_curr, noise_in, I_in, W_curr, W_in, Delta):
    W_eff = S_global * W_curr
    D = W_eff @ Psi_curr
    num = np.real(Psi_curr.conj() * D)
    den = (np.abs(Psi_curr)**2) + (np.abs(D)**2) + eps
    R = num / den
    g_vec = xi_curr * np.tanh(1.0 - R) - lam
    effective_gamma = gamma + adapt_curr
    dPsi = (1j*(W_eff @ Psi_curr)
            + alpha_curr*(Delta @ Psi_curr)
            + (g_vec * Psi_curr)
            - (effective_gamma * (np.abs(Psi_curr)**2) * Psi_curr))
    dPsi += noise_amp * noise_in
    dPsi += W_in * I_in * input_gain
    return dPsi


def run_simulation(signal_fn, total_time=400.0, seed=None, verbose=True):
    """
    Run the full brain simulation with a custom signal function.

    signal_fn(t) -> (value, label)  where label is numeric class label

    Returns: (states_X, targets_Y, harvest_times)
    """
    steps = int(total_time / dt)
    W, W_in, Delta = build_network(seed=seed)

    # State init
    Psi = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1
    xi_vec = np.ones(N) * 0.5
    A_vec = np.zeros(N)
    E_avg_vec = np.ones(N) * 0.1
    alpha_global = alpha_base

    Psi_ghost = Psi + (np.random.randn(N) + 1j*np.random.randn(N)) * 1e-5
    prev_dist = np.linalg.norm(Psi_ghost - Psi)
    Lyap_history = []

    xi_frozen = False
    xi_frozen_val = None

    states_X = []
    targets_Y = []
    harvest_times = []

    for t in range(steps):
        curr_time = t * dt

        noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
        I_val, Y_val = signal_fn(curr_time)
        W_snap = W.tocsr()

        # RK4
        k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global, noise_vec, I_val, W_snap, W_in, Delta)
        k2 = get_derivative(Psi + 0.5*dt*k1, xi_vec, A_vec, alpha_global, noise_vec, I_val, W_snap, W_in, Delta)
        k3 = get_derivative(Psi + 0.5*dt*k2, xi_vec, A_vec, alpha_global, noise_vec, I_val, W_snap, W_in, Delta)
        k4 = get_derivative(Psi + dt*k3, xi_vec, A_vec, alpha_global, noise_vec, I_val, W_snap, W_in, Delta)
        Psi = Psi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Ghost
        k1_g = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global, noise_vec, 0, W_snap, W_in, Delta)
        k2_g = get_derivative(Psi_ghost + 0.5*dt*k1_g, xi_vec, A_vec, alpha_global, noise_vec, 0, W_snap, W_in, Delta)
        k3_g = get_derivative(Psi_ghost + 0.5*dt*k2_g, xi_vec, A_vec, alpha_global, noise_vec, 0, W_snap, W_in, Delta)
        k4_g = get_derivative(Psi_ghost + dt*k3_g, xi_vec, A_vec, alpha_global, noise_vec, 0, W_snap, W_in, Delta)
        Psi_ghost = Psi_ghost + (dt/6.0)*(k1_g + 2*k2_g + 2*k3_g + k4_g)

        # Homeostasis
        instant_energy = np.abs(Psi)**2
        E_avg_vec = (1 - 0.01) * E_avg_vec + 0.01 * instant_energy
        mean_energy = np.mean(E_avg_vec)

        if curr_time >= stabilization_time and not xi_frozen:
            xi_frozen = True
            xi_frozen_val = xi_vec.copy()
            if verbose:
                print(f"    Xi FROZEN at t={curr_time:.1f}s, mean xi={np.mean(xi_vec):.3f}")

        if not xi_frozen:
            error_energy = target_energy - E_avg_vec
            if curr_time < 10.0:
                dXi = eta_xi_up * np.maximum(0, error_energy)
            else:
                rate = np.where(error_energy < 0, eta_xi_down, eta_xi_up)
                dXi = rate * error_energy
            xi_vec += dXi
            xi_vec = np.clip(xi_vec, xi_min, xi_max)
        else:
            xi_vec = xi_frozen_val.copy()

        excess_energy = np.maximum(0, E_avg_vec - target_energy)
        dA = (kappa_adapt * excess_energy - A_vec) / tau_adapt
        A_vec += dt * dA
        A_vec = np.clip(A_vec, 0.0, adapt_max)

        # Chaos control
        current_dist = np.linalg.norm(Psi_ghost - Psi)
        if current_dist < 1e-7 or current_dist > 1.0:
            Psi_ghost = Psi + (np.random.randn(N) + 1j*np.random.randn(N)) * 1e-4
            prev_dist = 1e-4
        else:
            instant_lyap = np.log(current_dist + 1e-12) - np.log(prev_dist + 1e-12)
            Lyap_history.append(instant_lyap)
            prev_dist = current_dist

        if len(Lyap_history) > lyap_window:
            Lyap_history.pop(0)
        lyap_smooth = np.mean(Lyap_history) if len(Lyap_history) > 0 else 0.0
        error_lyap = target_lyap - lyap_smooth
        alpha_global += eta_alpha * error_lyap
        alpha_global = np.clip(alpha_global, alpha_base, alpha_max)

        # Learning
        if curr_time < learning_end_time and (t % learn_interval == 0):
            from scipy.sparse import linalg as splinalg
            rows, cols = W.nonzero()
            amp_i = np.abs(Psi[rows])
            amp_j = np.abs(Psi[cols])
            corr = Psi[rows] * np.conj(Psi[cols])
            update = eta_hebb * corr * amp_i * amp_j
            W[rows, cols] += update - decay_hebb * W[rows, cols]
            # Re-normalize
            try:
                W_csr = W.tocsr()
                eigenvals = sp.linalg.eigs(W_csr, k=1, return_eigenvectors=False)
                max_eigen = np.abs(eigenvals[0])
                if max_eigen > 0:
                    W = W * (0.9 / max_eigen)
            except:
                pass

        # Harvest
        if curr_time > stabilization_time:
            if abs(mean_energy - target_energy) < energy_gate:
                states_X.append(np.concatenate([Psi.real, Psi.imag]))
                targets_Y.append(Y_val)
                harvest_times.append(curr_time)

        # Progress
        if verbose and t % 4000 == 0:
            print(f"    t={curr_time:6.1f}s  E={mean_energy:.3f}  xi={np.mean(xi_vec):.3f}")

    return np.array(states_X), np.array(targets_Y), np.array(harvest_times)


def classify(X, Y, T, block_duration, transition_skip, seed=42):
    """
    Classify with transition-skipped random split.
    Returns dict with accuracies and per-class results.
    """
    time_in_block = T % block_duration
    settled_mask = time_in_block >= transition_skip

    X_s = X[settled_mask]
    Y_s = Y[settled_mask]

    if len(Y_s) < 10:
        return {'test_acc': 0.0, 'train_acc': 0.0, 'per_class': {}, 'n_settled': 0}

    # Random 60/40 split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y_s))
    sp_pt = int(0.6 * len(Y_s))
    train_idx = np.sort(idx[:sp_pt])
    test_idx = np.sort(idx[sp_pt:])

    X_train, Y_train = X_s[train_idx], Y_s[train_idx]
    X_test, Y_test = X_s[test_idx], Y_s[test_idx]

    if len(X_test) == 0 or len(X_train) == 0:
        return {'test_acc': 0.0, 'train_acc': 0.0, 'per_class': {}, 'n_settled': len(Y_s)}

    # Balance training set
    classes = np.unique(Y_train)
    if len(classes) < 2:
        return {'test_acc': 0.0, 'train_acc': 0.0, 'per_class': {}, 'n_settled': len(Y_s)}

    class_counts = {c: np.sum(Y_train == c) for c in classes}
    min_count = min(class_counts.values())
    balanced_idx = []
    rng2 = np.random.default_rng(seed)
    for c in classes:
        c_idx = np.where(Y_train == c)[0]
        if len(c_idx) > min_count:
            c_idx = rng2.choice(c_idx, size=min_count, replace=False)
        balanced_idx.extend(c_idx)
    balanced_idx = np.sort(balanced_idx)
    X_train_bal = X_train[balanced_idx]
    Y_train_bal = Y_train[balanced_idx]

    # Scale + PCA + Ridge
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_bal)
    X_test_sc = scaler.transform(X_test)

    n_pca = min(pca_dims, len(X_train_bal), X_train_sc.shape[1])
    pca = PCA(n_components=n_pca)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)

    # For binary: Ridge regression with sign threshold
    # For multi-class: RidgeClassifier
    n_classes = len(classes)

    if n_classes == 2:
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train_pca, Y_train_bal)
        pred_train = model.predict(X_train_pca)
        pred_test = model.predict(X_test_pca)
        acc_train = np.mean((pred_train > 0) == (Y_train_bal > 0))
        acc_test = np.mean((pred_test > 0) == (Y_test > 0))

        per_class = {}
        for c in classes:
            mask = Y_test == c
            if np.any(mask):
                per_class[c] = np.mean((pred_test[mask] > 0) == (Y_test[mask] > 0))
            else:
                per_class[c] = 0.0
    else:
        model = RidgeClassifier(alpha=ridge_alpha)
        model.fit(X_train_pca, Y_train_bal)
        pred_train = model.predict(X_train_pca)
        pred_test = model.predict(X_test_pca)
        acc_train = np.mean(pred_train == Y_train_bal)
        acc_test = np.mean(pred_test == Y_test)

        per_class = {}
        for c in classes:
            mask = Y_test == c
            if np.any(mask):
                per_class[c] = np.mean(pred_test[mask] == Y_test[mask])
            else:
                per_class[c] = 0.0

    return {
        'test_acc': acc_test,
        'train_acc': acc_train,
        'per_class': per_class,
        'n_settled': len(Y_s),
        'n_train': len(Y_train_bal),
        'n_test': len(Y_test),
    }


# =============================================================
# SIGNAL GENERATORS
# =============================================================

def make_binary_signal(freq_a, freq_b, block_dur=50.0):
    """Two-frequency alternating blocks."""
    def signal(t):
        block = int(t / block_dur) % 2
        if block == 0:
            return np.sin(2 * np.pi * freq_a * t), -1
        else:
            return np.sin(2 * np.pi * freq_b * t), 1
    return signal


def make_multi_signal(freqs, block_dur=50.0):
    """Multi-frequency rotating blocks."""
    n = len(freqs)
    def signal(t):
        block = int(t / block_dur) % n
        freq = freqs[block]
        label = block  # 0, 1, 2, 3...
        return np.sin(2 * np.pi * freq * t), label
    return signal


def make_noisy_signal(freq_a, freq_b, noise_level, block_dur=50.0):
    """Two-frequency signal with additive Gaussian noise."""
    def signal(t):
        block = int(t / block_dur) % 2
        if block == 0:
            val = np.sin(2 * np.pi * freq_a * t)
            return val + noise_level * np.random.randn(), -1
        else:
            val = np.sin(2 * np.pi * freq_b * t)
            return val + noise_level * np.random.randn(), 1
    return signal


# =============================================================
# EXPERIMENTS
# =============================================================

results_text = []

def log(msg):
    print(msg)
    results_text.append(msg)


# ---- EXPERIMENT 1: SETTLING TIME SWEEP ----
log("=" * 65)
log("  EXPERIMENT 1: SETTLING TIME SWEEP")
log("=" * 65)
log("  Running simulation with 0.5 Hz vs 2.0 Hz (baseline)...")

t0 = clock.time()
signal_baseline = make_binary_signal(0.5, 2.0, block_dur=50.0)
X_base, Y_base, T_base = run_simulation(signal_baseline, total_time=400.0)
t1 = clock.time()
log(f"  Simulation done in {t1-t0:.1f}s. Samples: {len(Y_base)}")

log(f"\n  {'Skip(s)':>8s}  {'Settled':>8s}  {'Train%':>8s}  {'Test%':>8s}  {'ClassA':>8s}  {'ClassB':>8s}  {'Gap':>6s}")
log(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

settling_results = []
for skip in [0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25]:
    if skip >= 50:
        continue
    r = classify(X_base, Y_base, T_base, block_duration=50.0, transition_skip=skip)
    class_a = r['per_class'].get(-1, 0.0)
    class_b = r['per_class'].get(1, 0.0)
    gap = r['train_acc'] - r['test_acc']
    settling_results.append((skip, r))
    log(f"  {skip:8.0f}  {r['n_settled']:8d}  {r['train_acc']*100:7.1f}%  {r['test_acc']*100:7.1f}%  "
        f"{class_a*100:7.1f}%  {class_b*100:7.1f}%  {gap*100:5.1f}pp")

# Find minimum skip for ~100%
for skip, r in settling_results:
    if r['test_acc'] >= 0.99:
        log(f"\n  → Minimum skip for ≥99% accuracy: {skip}s")
        break
else:
    best_skip, best_r = max(settling_results, key=lambda x: x[1]['test_acc'])
    log(f"\n  → Best accuracy at skip={best_skip}s: {best_r['test_acc']*100:.1f}%")

# Class-specific settling (A vs B at each skip)
log(f"\n  Class-specific settling analysis:")
for skip, r in settling_results:
    a = r['per_class'].get(-1, 0.0)
    b = r['per_class'].get(1, 0.0)
    marker = " ← A settles" if a >= 0.99 and b < 0.99 else (" ← B settles" if b >= 0.99 and a < 0.99 else ("" if a < 0.99 else " ← both settled"))
    log(f"    skip={skip:2d}s  A={a*100:5.1f}%  B={b*100:5.1f}%  Δ={abs(a-b)*100:5.1f}pp{marker}")


# ---- EXPERIMENT 2: FREQUENCY RESOLUTION ----
log("\n" + "=" * 65)
log("  EXPERIMENT 2: FREQUENCY RESOLUTION")
log("=" * 65)

freq_pairs = [
    (0.5, 2.0),   # easy (baseline)
    (0.5, 1.0),   # moderate
    (0.5, 0.8),   
    (0.5, 0.7),   
    (0.5, 0.65),  
    (0.5, 0.6),   
    (0.5, 0.55),  # hard
]

log(f"\n  {'Pair':>16s}  {'Δf':>6s}  {'Ratio':>6s}  {'Train%':>8s}  {'Test%':>8s}  {'ClassA':>8s}  {'ClassB':>8s}")
log(f"  {'-'*16}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

freq_results = []
for fa, fb in freq_pairs:
    log(f"\n  Simulating {fa} Hz vs {fb} Hz...")
    sig = make_binary_signal(fa, fb, block_dur=50.0)
    t0 = clock.time()
    X_f, Y_f, T_f = run_simulation(sig, total_time=400.0, verbose=False)
    t1 = clock.time()

    r = classify(X_f, Y_f, T_f, block_duration=50.0, transition_skip=15.0)
    class_a = r['per_class'].get(-1, 0.0)
    class_b = r['per_class'].get(1, 0.0)
    freq_results.append((fa, fb, r))
    log(f"  {fa:.2f} vs {fb:.2f}Hz  {fb-fa:5.2f}  {fb/fa:5.2f}  {r['train_acc']*100:7.1f}%  {r['test_acc']*100:7.1f}%  "
        f"{class_a*100:7.1f}%  {class_b*100:7.1f}%  ({t1-t0:.0f}s)")

# Find resolution limit
log(f"\n  Frequency resolution analysis:")
for fa, fb, r in freq_results:
    status = "✓ SEPARABLE" if r['test_acc'] >= 0.75 else ("~ MARGINAL" if r['test_acc'] >= 0.6 else "✗ FAILED")
    log(f"    Δf={fb-fa:.2f}Hz (ratio={fb/fa:.2f}×)  →  {r['test_acc']*100:.1f}%  {status}")


# ---- EXPERIMENT 3: MULTI-CLASS (4 FREQUENCIES) ----
log("\n" + "=" * 65)
log("  EXPERIMENT 3: MULTI-CLASS CAPACITY")
log("=" * 65)

freqs_4 = [0.5, 1.0, 1.5, 2.0]
log(f"  Frequencies: {freqs_4}")
log(f"  Running simulation with 4-class rotating blocks...")

# With 4 classes, each block is 50s, one full cycle = 200s
# Total 400s = 2 full cycles. Harvest after 120s.
sig_multi = make_multi_signal(freqs_4, block_dur=50.0)
t0 = clock.time()
X_m, Y_m, T_m = run_simulation(sig_multi, total_time=400.0, verbose=True)
t1 = clock.time()
log(f"  Simulation done in {t1-t0:.1f}s. Samples: {len(Y_m)}")

# Classify
r_m = classify(X_m, Y_m, T_m, block_duration=50.0, transition_skip=15.0)
log(f"\n  Overall test accuracy: {r_m['test_acc']*100:.1f}%  (chance = 25%)")
log(f"  Per-class accuracy:")
for cls_label in sorted(r_m['per_class'].keys()):
    freq = freqs_4[cls_label]
    acc = r_m['per_class'][cls_label]
    log(f"    Class {cls_label} ({freq:.1f} Hz): {acc*100:.1f}%")


# ---- EXPERIMENT 4: NOISE ROBUSTNESS ----
log("\n" + "=" * 65)
log("  EXPERIMENT 4: NOISE ROBUSTNESS")
log("=" * 65)

noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

log(f"\n  {'Noise':>8s}  {'SNR(dB)':>8s}  {'Train%':>8s}  {'Test%':>8s}  {'ClassA':>8s}  {'ClassB':>8s}")
log(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

noise_results = []
for nl in noise_levels:
    sig = make_noisy_signal(0.5, 2.0, noise_level=nl, block_dur=50.0)
    t0 = clock.time()
    X_n, Y_n, T_n = run_simulation(sig, total_time=400.0, verbose=False)
    t1 = clock.time()

    r = classify(X_n, Y_n, T_n, block_duration=50.0, transition_skip=15.0)
    class_a = r['per_class'].get(-1, 0.0)
    class_b = r['per_class'].get(1, 0.0)
    # SNR: signal power = 0.5 (sine RMS^2), noise power = nl^2
    snr = 10 * np.log10(0.5 / (nl**2 + 1e-12)) if nl > 0 else float('inf')
    noise_results.append((nl, snr, r))
    snr_str = f"{snr:7.1f}" if nl > 0 else "    ∞"
    log(f"  {nl:8.3f}  {snr_str}  {r['train_acc']*100:7.1f}%  {r['test_acc']*100:7.1f}%  "
        f"{class_a*100:7.1f}%  {class_b*100:7.1f}%  ({t1-t0:.0f}s)")


# ---- EXPERIMENT 5: REAL-TIME DETECTION (Short Windows) ----
log("\n" + "=" * 65)
log("  EXPERIMENT 5: REAL-TIME DETECTION (short windows)")
log("=" * 65)
log("  Using baseline 0.5 vs 2.0 Hz simulation data")
log("  Testing classification from samples within short windows after transition")

# Use baseline data. For each block transition, take samples from
# [transition + offset, transition + offset + window]
# and test if those alone can classify.

block_dur = 50.0
transition_times = []
for t_val in np.arange(stabilization_time, 400.0, block_dur):
    transition_times.append(t_val)

windows = [(0, 1), (0, 2), (0, 3), (0, 5), (1, 3), (2, 5), (3, 5), (5, 10), (10, 15)]

log(f"\n  {'Window':>12s}  {'Samples':>8s}  {'Test%':>8s}")
log(f"  {'-'*12}  {'-'*8}  {'-'*8}")

for w_start, w_end in windows:
    # Select samples within [w_start, w_end) seconds after each block start
    time_in_block = T_base % block_dur
    mask = (time_in_block >= w_start) & (time_in_block < w_end)
    X_w = X_base[mask]
    Y_w = Y_base[mask]

    if len(Y_w) < 10:
        log(f"  {w_start}-{w_end}s         {len(Y_w):8d}  (too few)")
        continue

    # Random split (no transition skip needed — we selected the window)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Y_w))
    sp_pt = int(0.6 * len(Y_w))

    X_tr = X_w[idx[:sp_pt]]
    Y_tr = Y_w[idx[:sp_pt]]
    X_te = X_w[idx[sp_pt:]]
    Y_te = Y_w[idx[sp_pt:]]

    if len(X_tr) < 5 or len(X_te) < 5:
        log(f"  {w_start}-{w_end}s         {len(Y_w):8d}  (too few after split)")
        continue

    # Balance
    classes = np.unique(Y_tr)
    if len(classes) < 2:
        log(f"  {w_start}-{w_end}s         {len(Y_w):8d}  (single class)")
        continue

    min_c = min(np.sum(Y_tr == c) for c in classes)
    bal_idx = []
    for c in classes:
        c_idx = np.where(Y_tr == c)[0]
        if len(c_idx) > min_c:
            c_idx = np.random.default_rng(42).choice(c_idx, size=min_c, replace=False)
        bal_idx.extend(c_idx)
    X_tr_b = X_tr[np.sort(bal_idx)]
    Y_tr_b = Y_tr[np.sort(bal_idx)]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_b)
    X_te_sc = scaler.transform(X_te)

    n_pca = min(pca_dims, len(X_tr_b), X_tr_sc.shape[1])
    pca_m = PCA(n_components=n_pca)
    X_tr_p = pca_m.fit_transform(X_tr_sc)
    X_te_p = pca_m.transform(X_te_sc)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_tr_p, Y_tr_b)
    pred = model.predict(X_te_p)
    acc = np.mean((pred > 0) == (Y_te > 0))

    log(f"  {w_start:2d}-{w_end:2d}s        {len(Y_w):8d}  {acc*100:7.1f}%")


# =============================================================
# SUMMARY
# =============================================================
log("\n" + "=" * 65)
log("  SUMMARY: DYNAMICAL LIMITS OF THE ATTRACTOR ENCODER")
log("=" * 65)

# Settling
for skip, r in settling_results:
    if r['test_acc'] >= 0.99:
        log(f"  Settling time:       ≤{skip}s (minimum skip for ≥99% accuracy)")
        break
else:
    best_skip, best_r = max(settling_results, key=lambda x: x[1]['test_acc'])
    log(f"  Settling time:       ~{best_skip}s (best: {best_r['test_acc']*100:.1f}%)")

# Frequency resolution
for fa, fb, r in reversed(freq_results):
    if r['test_acc'] >= 0.75:
        log(f"  Freq resolution:     Δf≥{fb-fa:.2f}Hz ({fb/fa:.2f}× ratio) for ≥75% accuracy")
        break

# Multi-class
log(f"  Multi-class (4):     {r_m['test_acc']*100:.1f}% accuracy (chance=25%)")

# Noise
for nl, snr, r in noise_results:
    if r['test_acc'] < 0.75:
        log(f"  Noise threshold:     collapses at noise={nl:.2f} (SNR={snr:.0f}dB)")
        break
else:
    log(f"  Noise threshold:     robust at all tested levels (up to noise={noise_levels[-1]})")

log("")

# Save results
with open('/Users/pranay./Documents/THEBRAIN/stress_test_results.txt', 'w') as f:
    f.write('\n'.join(results_text))
print("Results saved to stress_test_results.txt")


# =============================================================
# VISUALIZATION
# =============================================================
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Brain Stress Test Suite — Dynamical Limits', fontsize=16, fontweight='bold')

# 1. Settling Time
ax = axs[0, 0]
skips = [s for s, _ in settling_results]
accs = [r['test_acc']*100 for _, r in settling_results]
acc_a = [r['per_class'].get(-1, 0)*100 for _, r in settling_results]
acc_b = [r['per_class'].get(1, 0)*100 for _, r in settling_results]
ax.plot(skips, accs, 'o-', color='cyan', linewidth=2, label='Overall')
ax.plot(skips, acc_a, 's--', color='coral', linewidth=1, label='Class A (slow)')
ax.plot(skips, acc_b, 'd--', color='lime', linewidth=1, label='Class B (fast)')
ax.axhline(y=99, color='yellow', linestyle=':', alpha=0.5, label='99% threshold')
ax.axhline(y=50, color='red', linestyle=':', alpha=0.3, label='Chance')
ax.set_xlabel('Transition Skip (s)')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Exp 1: Settling Time')
ax.legend(fontsize=7)
ax.set_ylim(40, 105)

# 2. Frequency Resolution
ax = axs[0, 1]
deltas = [fb - fa for fa, fb, _ in freq_results]
f_accs = [r['test_acc']*100 for _, _, r in freq_results]
ax.plot(deltas, f_accs, 'o-', color='cyan', linewidth=2, markersize=8)
for i, (fa, fb, r) in enumerate(freq_results):
    ax.annotate(f'{fb:.2f}', (deltas[i], f_accs[i]+2), fontsize=7, ha='center', color='white')
ax.axhline(y=75, color='yellow', linestyle=':', alpha=0.5, label='75% threshold')
ax.axhline(y=50, color='red', linestyle=':', alpha=0.3, label='Chance')
ax.set_xlabel('Frequency Difference (Hz)')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Exp 2: Frequency Resolution')
ax.legend(fontsize=7)
ax.invert_xaxis()
ax.set_ylim(40, 105)

# 3. Multi-Class
ax = axs[0, 2]
class_labels = sorted(r_m['per_class'].keys())
class_accs = [r_m['per_class'][c]*100 for c in class_labels]
class_names = [f"{freqs_4[c]:.1f}Hz" for c in class_labels]
bars = ax.bar(class_names, class_accs, color=['coral', 'gold', 'lime', 'cyan'])
ax.axhline(y=25, color='red', linestyle=':', alpha=0.5, label='Chance (25%)')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title(f'Exp 3: Multi-Class ({r_m["test_acc"]*100:.1f}% overall)')
ax.set_ylim(0, 110)
ax.legend(fontsize=7)
for bar, acc in zip(bars, class_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{acc:.0f}%', ha='center', fontsize=9, fontweight='bold')

# 4. Noise Robustness
ax = axs[1, 0]
nls = [nl for nl, _, _ in noise_results if nl > 0]
n_accs = [r['test_acc']*100 for nl, _, r in noise_results if nl > 0]
if nls:
    ax.semilogx(nls, n_accs, 'o-', color='cyan', linewidth=2, markersize=8)
    ax.axhline(y=75, color='yellow', linestyle=':', alpha=0.5, label='75% threshold')
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.3, label='Chance')
ax.set_xlabel('Noise Amplitude')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Exp 4: Noise Robustness')
ax.legend(fontsize=7)
ax.set_ylim(40, 105)

# 5. Real-Time Detection (placeholder — fill from results)
ax = axs[1, 1]
ax.text(0.5, 0.5, 'See terminal output\nfor Exp 5 results', ha='center', va='center',
        transform=ax.transAxes, fontsize=12, color='gray')
ax.set_title('Exp 5: Real-Time Detection')
ax.set_xlabel('Window (s after transition)')
ax.set_ylabel('Test Accuracy (%)')

# 6. Summary Text
ax = axs[1, 2]
ax.axis('off')
summary_lines = [
    "DYNAMICAL LIMITS SUMMARY",
    "─" * 30,
]
for skip, r in settling_results:
    if r['test_acc'] >= 0.99:
        summary_lines.append(f"Settling: ≤{skip}s for 99%+")
        break
for fa, fb, r in reversed(freq_results):
    if r['test_acc'] >= 0.75:
        summary_lines.append(f"Resolution: Δf≥{fb-fa:.2f}Hz")
        break
summary_lines.append(f"Multi-class: {r_m['test_acc']*100:.0f}% (4 freq)")
for nl, snr, r in noise_results:
    if r['test_acc'] < 0.75:
        summary_lines.append(f"Noise limit: σ={nl:.2f}")
        break
else:
    summary_lines.append(f"Noise: robust to σ={noise_levels[-1]}")

ax.text(0.1, 0.9, '\n'.join(summary_lines), transform=ax.transAxes,
        fontsize=12, fontfamily='monospace', va='top', color='cyan',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/pranay./Documents/THEBRAIN/stress_test_results.png', dpi=150, bbox_inches='tight')
print("Plot saved to stress_test_results.png")
plt.show()
