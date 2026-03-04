"""
BREAK THE BRAIN v2 — Invariant Features + Temporal Block Split
================================================================
Uses energy variance + spectral power features (M34) instead of raw Ψ.
All tests use temporal block splits (train early blocks, test later blocks).
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time as clock

# =============================================================
# PHYSICS (identical to neuron.py M34)
# =============================================================
N = 500
lam = 0.8
gamma_vec = np.linspace(0.1, 2.0, N)
eps = 1e-6
dt = 0.05
target_energy = 2.5
input_gain = 1.5
eta_xi_up = 0.005
eta_xi_down = 0.002
xi_min = 0.1
xi_max = 3.0
tau_adapt_vec = np.linspace(0.2, 5.0, N)
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
ridge_alpha = 1000.0
density = 0.02
block_duration = 50.0
transition_skip = 15.0

# Feature extraction params
window_seconds = 5.0
window_steps = int(window_seconds / dt)
feature_sample_interval = 10  # every 0.5s


def build_network():
    W_real = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W_imag = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W = (W_real + 1j * W_imag)
    try:
        eigenvals = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
        max_eigen = np.abs(eigenvals[0])
        if max_eigen > 0:
            W = W * (0.9 / max_eigen)
    except:
        pass
    np.random.seed(42)
    W_in = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.5
    A_temp = sp.random(N, N, density=density, format='csr')
    A_temp = (A_temp + A_temp.T) * 0.5
    degrees = np.array(A_temp.sum(axis=1)).flatten()
    Delta = sp.diags(degrees) - A_temp
    return W, W_in, Delta


def get_derivative(Psi_curr, xi_curr, adapt_curr, alpha_curr, noise_in, I_in, W_curr, W_in, Delta):
    W_eff = S_global * W_curr
    D = W_eff @ Psi_curr
    num = np.real(Psi_curr.conj() * D)
    den = (np.abs(Psi_curr)**2) + (np.abs(D)**2) + eps
    R = num / den
    g_vec = xi_curr * np.tanh(1.0 - R) - lam
    effective_gamma = gamma_vec + adapt_curr
    dPsi = (1j*(W_eff @ Psi_curr)
            + alpha_curr*(Delta @ Psi_curr)
            + (g_vec * Psi_curr)
            - (effective_gamma * (np.abs(Psi_curr)**2) * Psi_curr))
    dPsi += noise_amp * noise_in
    dPsi += W_in * I_in * input_gain
    return dPsi


def run_sim(signal_fn, total_time=400.0, verbose=True):
    """
    Run simulation and extract INVARIANT FEATURES (energy variance + spectral).
    Returns (X_evar, X_spec, X_combined, Y, T)
    """
    steps = int(total_time / dt)
    W, W_in, Delta = build_network()

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

    # Rolling window buffer
    psi_buffer = np.zeros((window_steps, N), dtype=complex)
    buf_idx = 0
    buf_filled = False

    features_evar = []
    features_spec = []
    targets_Y = []
    harvest_T = []

    for t in range(steps):
        ct = t * dt
        noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
        I_val, Y_val = signal_fn(ct)
        Wc = W.tocsr()

        # RK4
        k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k2 = get_derivative(Psi+0.5*dt*k1, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k3 = get_derivative(Psi+0.5*dt*k2, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k4 = get_derivative(Psi+dt*k3, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

        # Ghost
        k1g = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        k2g = get_derivative(Psi_ghost+0.5*dt*k1g, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        k3g = get_derivative(Psi_ghost+0.5*dt*k2g, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        k4g = get_derivative(Psi_ghost+dt*k3g, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        Psi_ghost = Psi_ghost + (dt/6.0)*(k1g+2*k2g+2*k3g+k4g)

        # Homeostasis
        instant_energy = np.abs(Psi)**2
        E_avg_vec = 0.99*E_avg_vec + 0.01*instant_energy
        mean_energy = np.mean(E_avg_vec)

        if ct >= stabilization_time and not xi_frozen:
            xi_frozen = True
            xi_frozen_val = xi_vec.copy()
            if verbose:
                print(f"    Xi FROZEN at t={ct:.1f}s")

        if not xi_frozen:
            error_energy = target_energy - E_avg_vec
            if ct < 10.0:
                dXi = eta_xi_up * np.maximum(0, error_energy)
            else:
                rate = np.where(error_energy < 0, eta_xi_down, eta_xi_up)
                dXi = rate * error_energy
            xi_vec = np.clip(xi_vec + dXi, xi_min, xi_max)
        else:
            xi_vec = xi_frozen_val.copy()

        excess_energy = np.maximum(0, E_avg_vec - target_energy)
        A_vec = np.clip(A_vec + dt*((kappa_adapt*excess_energy - A_vec)/tau_adapt_vec), 0, adapt_max)

        # Chaos control
        current_dist = np.linalg.norm(Psi_ghost - Psi)
        if current_dist < 1e-7 or current_dist > 1.0:
            Psi_ghost = Psi + (np.random.randn(N)+1j*np.random.randn(N))*1e-4
            prev_dist = 1e-4
        else:
            Lyap_history.append(np.log(current_dist+1e-12) - np.log(prev_dist+1e-12))
            prev_dist = current_dist
        if len(Lyap_history) > lyap_window:
            Lyap_history.pop(0)
        lyap_smooth = np.mean(Lyap_history) if Lyap_history else 0.0
        alpha_global = np.clip(alpha_global + eta_alpha*(target_lyap - lyap_smooth), alpha_base, alpha_max)

        # Learning
        if ct < learning_end_time and (t % learn_interval == 0):
            rows, cols = W.nonzero()
            corr = Psi[rows] * np.conj(Psi[cols])
            update = eta_hebb * corr * np.abs(Psi[rows]) * np.abs(Psi[cols])
            W[rows, cols] += update - decay_hebb * W[rows, cols]
            try:
                eigenvals = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
                if np.abs(eigenvals[0]) > 0:
                    W = W * (0.9 / np.abs(eigenvals[0]))
            except:
                pass

        # Update rolling buffer
        psi_buffer[buf_idx] = Psi.copy()
        buf_idx = (buf_idx + 1) % window_steps
        if t >= window_steps:
            buf_filled = True

        # Harvest INVARIANT FEATURES
        if ct > stabilization_time and buf_filled and (t % feature_sample_interval == 0):
            if abs(mean_energy - target_energy) < energy_gate:
                time_in_block = ct % block_duration
                if time_in_block < transition_skip:
                    continue

                ordered = np.roll(psi_buffer, -buf_idx, axis=0)
                energy_series = np.abs(ordered)**2

                # Energy variance per neuron (N dims)
                features_evar.append(np.var(energy_series, axis=0))

                # Spectral power (4*N dims)
                energy_centered = energy_series - energy_series.mean(axis=0, keepdims=True)
                fft_result = np.fft.rfft(energy_centered, axis=0)
                power = np.abs(fft_result)**2
                freqs = np.fft.rfftfreq(window_steps, d=dt)
                bands = [(0.3, 0.7), (0.8, 1.5), (1.5, 2.5), (2.5, 5.0)]
                spec_feats = []
                for f_lo, f_hi in bands:
                    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
                    if np.any(band_mask):
                        spec_feats.append(np.mean(power[band_mask], axis=0))
                    else:
                        spec_feats.append(np.zeros(N))
                features_spec.append(np.concatenate(spec_feats))

                targets_Y.append(Y_val)
                harvest_T.append(ct)

        if verbose and t % 4000 == 0:
            print(f"    t={ct:6.1f}s  E={mean_energy:.3f}  samples={len(targets_Y)}")

    X_evar = np.array(features_evar)
    X_spec = np.array(features_spec)
    X_combined = np.hstack([X_evar, X_spec])
    Y = np.array(targets_Y)
    T = np.array(harvest_T)
    return X_evar, X_spec, X_combined, Y, T


# =============================================================
# TEMPORAL BLOCK SPLIT CLASSIFIER
# =============================================================

def classify_temporal(X, Y, T, block_dur, n_train_blocks=4, pca_dims=50):
    """Train on first n blocks, test on rest. No leakage."""
    block_idx = (T / block_dur).astype(int)
    first_block = int(stabilization_time / block_dur)
    rel_block = block_idx - first_block

    train_mask = rel_block < n_train_blocks
    test_mask = rel_block >= n_train_blocks

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]

    if len(X_train) < 5 or len(X_test) < 5:
        return {'test_acc': 0.0, 'train_acc': 0.0, 'per_class': {},
                'n_train': len(X_train), 'n_test': len(X_test)}

    classes = np.unique(Y_train)
    if len(classes) < 2:
        return {'test_acc': 0.0, 'train_acc': 0.0, 'per_class': {},
                'n_train': len(X_train), 'n_test': len(X_test)}

    # Balance
    min_c = min(np.sum(Y_train == c) for c in classes)
    bal_idx = []
    rng = np.random.default_rng(42)
    for c in classes:
        ci = np.where(Y_train == c)[0]
        if len(ci) > min_c:
            ci = rng.choice(ci, size=min_c, replace=False)
        bal_idx.extend(ci)
    X_tr = X_train[np.sort(bal_idx)]
    Y_tr = Y_train[np.sort(bal_idx)]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_test)

    n_pca = min(pca_dims, len(X_tr), X_tr_sc.shape[1])
    pca = PCA(n_components=n_pca)
    X_tr_p = pca.fit_transform(X_tr_sc)
    X_te_p = pca.transform(X_te_sc)

    n_classes = len(classes)
    if n_classes == 2:
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_tr_p, Y_tr)
        pred_tr = model.predict(X_tr_p)
        pred_te = model.predict(X_te_p)
        acc_tr = np.mean((pred_tr > 0) == (Y_tr > 0))
        acc_te = np.mean((pred_te > 0) == (Y_test > 0))
        per_class = {}
        for c in classes:
            m = Y_test == c
            per_class[c] = np.mean((pred_te[m] > 0) == (Y_test[m] > 0)) if np.any(m) else 0.0
    else:
        model = RidgeClassifier(alpha=ridge_alpha)
        model.fit(X_tr_p, Y_tr)
        pred_tr = model.predict(X_tr_p)
        pred_te = model.predict(X_te_p)
        acc_tr = np.mean(pred_tr == Y_tr)
        acc_te = np.mean(pred_te == Y_test)
        per_class = {}
        for c in classes:
            m = Y_test == c
            per_class[c] = np.mean(pred_te[m] == Y_test[m]) if np.any(m) else 0.0

    return {'test_acc': acc_te, 'train_acc': acc_tr, 'per_class': per_class,
            'n_train': len(X_tr), 'n_test': len(X_test)}


# =============================================================
# SIGNAL GENERATORS
# =============================================================

def make_binary_signal(fa, fb, block_dur=50.0, noise_level=0.0):
    def signal(t):
        block = int(t / block_dur) % 2
        if block == 0:
            return np.sin(2*np.pi*fa*t) + noise_level*np.random.randn(), -1
        else:
            return np.sin(2*np.pi*fb*t) + noise_level*np.random.randn(), 1
    return signal

def make_multi_signal(freqs, block_dur=50.0):
    n = len(freqs)
    def signal(t):
        block = int(t / block_dur) % n
        return np.sin(2*np.pi*freqs[block]*t), block
    return signal


# =============================================================
# RESULTS
# =============================================================
results_text = []
def log(msg):
    print(msg)
    results_text.append(msg)


# ---- TEST 0: BASELINE — Energy Variance, 0.5 vs 2.0 Hz ----
log("=" * 70)
log("  TEST 0: BASELINE — Energy Variance Features (0.5 vs 2.0 Hz)")
log("=" * 70)

sig = make_binary_signal(0.5, 2.0, block_dur=50.0)
t0 = clock.time()
X_ev, X_sp, X_cb, Y, T = run_sim(sig, total_time=400.0)
t1 = clock.time()
log(f"  Sim: {t1-t0:.1f}s, {len(Y)} samples")

for name, X_feat in [("EnergyVar", X_ev), ("Spectral", X_sp), ("Combined", X_cb)]:
    r = classify_temporal(X_feat, Y, T, block_dur=50.0)
    a = r['per_class'].get(-1, 0)
    b = r['per_class'].get(1, 0)
    log(f"  {name:>12}: test={r['test_acc']*100:.1f}%  A={a*100:.0f}% B={b*100:.0f}%  "
        f"(train={r['train_acc']*100:.0f}%, {X_feat.shape[1]}d, tr={r['n_train']} te={r['n_test']})")


# ---- TEST 1: SETTLING TIME ----
log("\n" + "=" * 70)
log("  TEST 1: SETTLING TIME (temporal split, energy variance)")
log("=" * 70)
# Need to re-run with different transition_skip — but that's baked into harvest.
# Instead, run multiple sims with different transition_skip values.
# Actually, we can post-hoc filter by time_in_block since T is available.
# But the features were already computed with transition_skip=15. 
# For settling test, we need sims with skip=0 and filter post-hoc.
log("  Running sim with transition_skip=0 for post-hoc analysis...")

# Temporarily patch skip to 0 for this sim
old_skip = transition_skip
transition_skip = 0.0

sig = make_binary_signal(0.5, 2.0, block_dur=50.0)
X_ev0, X_sp0, X_cb0, Y0, T0 = run_sim(sig, total_time=400.0, verbose=False)
transition_skip = old_skip

log(f"  Samples with skip=0: {len(Y0)}")
log(f"\n  {'Skip(s)':>8}  {'Test%':>8}  {'A%':>6}  {'B%':>6}  {'Samples':>8}")
log(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*8}")

for skip_val in [0, 1, 2, 3, 5, 7, 10, 15, 20]:
    time_in_block = T0 % 50.0
    mask = time_in_block >= skip_val
    if np.sum(mask) < 20:
        continue
    r = classify_temporal(X_cb0[mask], Y0[mask], T0[mask], block_dur=50.0)
    a = r['per_class'].get(-1, 0)
    b = r['per_class'].get(1, 0)
    log(f"  {skip_val:8d}  {r['test_acc']*100:7.1f}%  {a*100:5.1f}%  {b*100:5.1f}%  {np.sum(mask):8d}")


# ---- TEST 2: FREQUENCY RESOLUTION ----
log("\n" + "=" * 70)
log("  TEST 2: FREQUENCY RESOLUTION (temporal split, energy variance)")
log("=" * 70)

freq_pairs = [
    (0.5, 2.0),
    (0.5, 1.0),
    (0.5, 0.8),
    (0.5, 0.7),
    (0.5, 0.6),
    (0.5, 0.55),
    (0.5, 0.53),
    (0.5, 0.52),
    (0.5, 0.51),
]

log(f"\n  {'Pair':>16}  {'Δf':>6}  {'Test%':>8}  {'A%':>6}  {'B%':>6}  {'Time':>5}")
log(f"  {'─'*16}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*5}")

freq_results = []
for fa, fb in freq_pairs:
    sig = make_binary_signal(fa, fb, block_dur=50.0)
    t0 = clock.time()
    Xev, Xsp, Xcb, Yf, Tf = run_sim(sig, total_time=400.0, verbose=False)
    t1 = clock.time()
    r = classify_temporal(Xcb, Yf, Tf, block_dur=50.0)
    a = r['per_class'].get(-1, 0)
    b = r['per_class'].get(1, 0)
    freq_results.append((fa, fb, r))
    status = "✓" if r['test_acc'] >= 0.75 else ("~" if r['test_acc'] >= 0.6 else "✗")
    log(f"  {fa:.3f} vs {fb:.3f}  {fb-fa:5.3f}  {r['test_acc']*100:7.1f}%  "
        f"{a*100:5.1f}%  {b*100:5.1f}%  {t1-t0:4.0f}s  {status}")


# ---- TEST 3: MULTI-CLASS ----
log("\n" + "=" * 70)
log("  TEST 3: MULTI-CLASS (temporal split, energy variance)")
log("=" * 70)

for n_freq, total_t in [(4, 400), (8, 800)]:
    freqs = [0.5 + i * 0.5 for i in range(n_freq)]
    log(f"\n  {n_freq} classes: {[f'{f:.1f}' for f in freqs]}")
    sig = make_multi_signal(freqs, block_dur=50.0)
    t0 = clock.time()
    Xev, Xsp, Xcb, Ym, Tm = run_sim(sig, total_time=total_t, verbose=False)
    t1 = clock.time()
    log(f"  Sim: {t1-t0:.0f}s, {len(Ym)} samples")
    r = classify_temporal(Xcb, Ym, Tm, block_dur=50.0, n_train_blocks=n_freq)
    log(f"  Overall: {r['test_acc']*100:.1f}%  (chance={100/n_freq:.0f}%)")
    for cls in sorted(r['per_class'].keys()):
        log(f"    Class {cls} ({freqs[cls]:.1f}Hz): {r['per_class'][cls]*100:.1f}%")


# ---- TEST 4: NOISE ROBUSTNESS ----
log("\n" + "=" * 70)
log("  TEST 4: NOISE ROBUSTNESS (temporal split, energy variance)")
log("=" * 70)

log(f"\n  {'Noise':>8}  {'SNR':>8}  {'Test%':>8}  {'A%':>6}  {'B%':>6}")
log(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")

for nl in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    sig = make_binary_signal(0.5, 2.0, block_dur=50.0, noise_level=nl)
    Xev, Xsp, Xcb, Yn, Tn = run_sim(sig, total_time=400.0, verbose=False)
    r = classify_temporal(Xcb, Yn, Tn, block_dur=50.0)
    a = r['per_class'].get(-1, 0)
    b = r['per_class'].get(1, 0)
    snr = f"{10*np.log10(0.5/(nl**2+1e-12)):6.0f}dB" if nl > 0 else "     ∞"
    log(f"  {nl:8.2f}  {snr}  {r['test_acc']*100:7.1f}%  {a*100:5.1f}%  {b*100:5.1f}%")


# ---- TEST 5: SHORT BLOCKS ----
log("\n" + "=" * 70)
log("  TEST 5: SHORT BLOCKS (temporal split, energy variance)")
log("=" * 70)

log(f"\n  {'Block':>8}  {'Skip':>6}  {'Test%':>8}  {'A%':>6}  {'B%':>6}")
log(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*6}")

for bdur, skip in [(50, 15), (25, 7), (10, 3), (5, 2)]:
    old_skip = transition_skip
    transition_skip = skip
    sig = make_binary_signal(0.5, 2.0, block_dur=bdur)
    total_t = max(400.0, stabilization_time + bdur * 8 + 50)
    Xev, Xsp, Xcb, Yb, Tb = run_sim(sig, total_time=total_t, verbose=False)
    transition_skip = old_skip
    r = classify_temporal(Xcb, Yb, Tb, block_dur=bdur)
    a = r['per_class'].get(-1, 0)
    b = r['per_class'].get(1, 0)
    log(f"  {bdur:6d}s  {skip:4d}s  {r['test_acc']*100:7.1f}%  {a*100:5.1f}%  {b*100:5.1f}%  "
        f"(tr={r['n_train']} te={r['n_test']})")


# ---- TEST 6: COMBINED STRESS ----
log("\n" + "=" * 70)
log("  TEST 6: COMBINED (close freq + noise, temporal split, energy var)")
log("=" * 70)

combos = [
    (0.5, 0.6, 0.0),
    (0.5, 0.6, 0.1),
    (0.5, 0.6, 0.5),
    (0.5, 0.55, 0.0),
    (0.5, 0.55, 0.1),
    (0.5, 0.55, 0.5),
]

log(f"\n  {'Pair':>16}  {'Δf':>6}  {'Noise':>6}  {'Test%':>8}")
log(f"  {'─'*16}  {'─'*6}  {'─'*6}  {'─'*8}")

for fa, fb, nl in combos:
    sig = make_binary_signal(fa, fb, block_dur=50.0, noise_level=nl)
    Xev, Xsp, Xcb, Yc, Tc = run_sim(sig, total_time=400.0, verbose=False)
    r = classify_temporal(Xcb, Yc, Tc, block_dur=50.0)
    status = "✓" if r['test_acc'] >= 0.75 else ("~" if r['test_acc'] >= 0.6 else "✗")
    log(f"  {fa:.2f} vs {fb:.2f}Hz  {fb-fa:5.2f}  {nl:5.2f}  "
        f"{r['test_acc']*100:7.1f}%  {status}")


# =============================================================
# SUMMARY
# =============================================================
log("\n" + "=" * 70)
log("  FINAL SUMMARY — INVARIANT FEATURES + TEMPORAL SPLIT")
log("=" * 70)

# Save
with open('/Users/pranay./Documents/THEBRAIN/break_test_v2_results.txt', 'w') as f:
    f.write('\n'.join(results_text))
print("\nSaved to break_test_v2_results.txt")
