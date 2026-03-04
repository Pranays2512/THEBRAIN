"""
M36 BREAK TEST + FREQUENCY REGRESSION
=======================================
Tests the resonant reservoir with |Ψ|² snapshot readout.
No windowing needed — single snapshot IS a frequency spectrum.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time as clock

# =============================================================
# M36 PHYSICS (must match neuron.py)
# =============================================================
N = 500; lam = 0.8; eps = 1e-6; dt = 0.05
target_energy = 2.5; input_gain = 3.0

omega_hz = np.logspace(np.log10(0.3), np.log10(3.0), N)
omega_vec = 2.0 * np.pi * omega_hz
Q_factor = 15.0
gamma_vec = omega_hz / Q_factor

eta_xi = 0.002; xi_min = 0.1; xi_max = 3.0
tau_adapt_vec = np.linspace(0.2, 5.0, N)
kappa_adapt = 0.5; adapt_max = 2.0
alpha_base = 0.01; alpha_max = 0.1; target_lyap = 0.05; eta_alpha = 0.0005
lyap_window = 100; coupling_strength = 0.1
learning_end_time = 100.0; learn_interval = 20
eta_hebb = 0.001; decay_hebb = 0.0001; noise_amp = 0.02
stabilization_time = 120.0; energy_gate = 1.0
ridge_alpha = 1000.0; density = 0.02
block_duration = 50.0; transition_skip = 5.0
feature_sample_interval = 10


def build_network():
    W_real = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W_imag = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W = (W_real + 1j * W_imag)
    try:
        eigenvals = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
        if np.abs(eigenvals[0]) > 0: W = W * (0.9 / np.abs(eigenvals[0]))
    except: pass
    np.random.seed(42)
    W_in = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.5
    A_temp = sp.random(N, N, density=density, format='csr')
    A_temp = (A_temp + A_temp.T) * 0.5
    degrees = np.array(A_temp.sum(axis=1)).flatten()
    Delta = sp.diags(degrees) - A_temp
    return W, W_in, Delta


def get_derivative(Psi_curr, xi_val, adapt_curr, alpha_curr, noise_in, I_in, W_curr, W_in, Delta):
    hopf_rotation = 1j * omega_vec * Psi_curr
    num = np.real(Psi_curr.conj() * (W_curr @ Psi_curr))
    den = (np.abs(Psi_curr)**2) + eps
    R = num / den
    g_vec = xi_val * np.tanh(1.0 - R) - lam
    effective_gamma = gamma_vec + adapt_curr
    coupling = coupling_strength * (W_curr @ Psi_curr)
    diffusion = alpha_curr * (Delta @ Psi_curr)
    dPsi = (hopf_rotation + g_vec * Psi_curr
            - effective_gamma * (np.abs(Psi_curr)**2) * Psi_curr
            + coupling + diffusion + noise_amp * noise_in
            + W_in * I_in * input_gain)
    return dPsi


def run_sim(signal_func, total_time=400.0, verbose=True, t_skip=None, blk_dur=None):
    """Run M36 sim, return |Ψ|² snapshots, labels, times."""
    if t_skip is None: t_skip = transition_skip
    if blk_dur is None: blk_dur = block_duration
    steps = int(total_time / dt)
    W, W_in, Delta = build_network()

    Psi = (np.random.randn(N) + 1j*np.random.randn(N)) * 0.1
    xi_global = 0.5
    A_vec = np.zeros(N)
    E_avg_vec = np.ones(N) * 0.1
    alpha_global = alpha_base
    Psi_ghost = Psi + (np.random.randn(N)+1j*np.random.randn(N))*1e-5
    prev_dist = np.linalg.norm(Psi_ghost - Psi)
    Lyap_history = []
    xi_frozen = False; xi_frozen_val = None

    features_X = []
    targets_Y = []
    harvest_T = []

    for t in range(steps):
        ct = t * dt
        noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
        I_val, Y_val = signal_func(ct)
        xi_bcast = xi_global
        Wc = W.tocsr()

        k1 = get_derivative(Psi, xi_bcast, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k2 = get_derivative(Psi+0.5*dt*k1, xi_bcast, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k3 = get_derivative(Psi+0.5*dt*k2, xi_bcast, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k4 = get_derivative(Psi+dt*k3, xi_bcast, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

        k1g = get_derivative(Psi_ghost, xi_bcast, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        k2g = get_derivative(Psi_ghost+0.5*dt*k1g, xi_bcast, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        k3g = get_derivative(Psi_ghost+0.5*dt*k2g, xi_bcast, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        k4g = get_derivative(Psi_ghost+dt*k3g, xi_bcast, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        Psi_ghost = Psi_ghost + (dt/6.0)*(k1g+2*k2g+2*k3g+k4g)

        instant_energy = np.abs(Psi)**2
        E_avg_vec = 0.99*E_avg_vec + 0.01*instant_energy
        mean_energy = np.mean(E_avg_vec)

        if ct >= stabilization_time and not xi_frozen:
            xi_frozen = True; xi_frozen_val = xi_global
            if verbose: print(f"    Xi FROZEN at t={ct:.1f}s")

        if not xi_frozen:
            error = target_energy - np.mean(E_avg_vec)
            xi_global = np.clip(xi_global + eta_xi * error, xi_min, xi_max)
        else:
            xi_global = xi_frozen_val

        excess_energy = np.maximum(0, E_avg_vec - target_energy)
        A_vec = np.clip(A_vec + dt*((kappa_adapt*excess_energy - A_vec)/tau_adapt_vec), 0, adapt_max)

        current_dist = np.linalg.norm(Psi_ghost - Psi)
        if current_dist < 1e-7 or current_dist > 1.0:
            Psi_ghost = Psi + (np.random.randn(N)+1j*np.random.randn(N))*1e-4
            prev_dist = 1e-4
        else:
            Lyap_history.append(np.log(current_dist+1e-12) - np.log(prev_dist+1e-12))
            prev_dist = current_dist
        if len(Lyap_history) > lyap_window: Lyap_history.pop(0)
        lyap_smooth = np.mean(Lyap_history) if Lyap_history else 0.0
        alpha_global = np.clip(alpha_global + eta_alpha*(target_lyap - lyap_smooth), alpha_base, alpha_max)

        if ct < learning_end_time and (t % learn_interval == 0):
            rows, cols = W.nonzero()
            corr = Psi[rows] * np.conj(Psi[cols])
            update = eta_hebb * corr * np.abs(Psi[rows]) * np.abs(Psi[cols])
            W[rows, cols] += update - decay_hebb * W[rows, cols]
            try:
                eigenvals = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
                if np.abs(eigenvals[0]) > 0: W = W * (0.9 / np.abs(eigenvals[0]))
            except: pass

        # Snapshot harvest
        if ct > stabilization_time and (t % feature_sample_interval == 0):
            time_in_block = ct % blk_dur
            if time_in_block >= t_skip:
                features_X.append(np.abs(Psi)**2)
                targets_Y.append(Y_val)
                harvest_T.append(ct)

    return np.array(features_X), np.array(targets_Y), np.array(harvest_T)


def make_binary_signal(fa, fb, block_dur=50.0, noise_level=0.0):
    def sig(t):
        block = int(t / block_dur) % 2
        freq = fa if block == 0 else fb
        phase_noise = noise_level * np.random.randn() if noise_level > 0 else 0
        return np.sin(2*np.pi*(freq+phase_noise)*t), (-1 if block == 0 else 1)
    return sig


def classify_temporal(X, Y, T, block_dur=50.0, n_train_blocks=4):
    block_idx = (T / block_dur).astype(int)
    first_block = int(stabilization_time / block_dur)
    rel_block = block_idx - first_block
    train_mask = rel_block < n_train_blocks
    test_mask = rel_block >= n_train_blocks
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]
    if len(X_test) < 5 or len(X_train) < 5:
        return {'test_acc': 0.5, 'per_class': {}}

    classes = np.unique(Y_train)
    if len(classes) < 2: return {'test_acc': 0.5, 'per_class': {}}
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
        acc = np.mean((pred > 0) == (Y_test > 0))
        per_class = {}
        for c in classes:
            m = Y_test == c
            per_class[c] = np.mean((pred[m] > 0) == (Y_test[m] > 0)) if np.any(m) else 0
    else:
        pred_classes = classes[np.argmin(np.abs(pred[:, None] - classes[None, :]), axis=1)]
        acc = np.mean(pred_classes == Y_test)
        per_class = {}
        for c in classes:
            m = Y_test == c
            per_class[c] = np.mean(pred_classes[m] == c) if np.any(m) else 0

    return {'test_acc': acc, 'per_class': per_class, 'n_train': len(X_tr), 'n_test': len(X_test)}


def log(msg):
    print(msg)


# =============================================================
# RUN TESTS
# =============================================================
print("=" * 70)
print("  M36 BREAK TEST — RESONANT RESERVOIR + |Ψ|² SNAPSHOT")
print("=" * 70)

# --- TEST 0: BASELINE ---
print(f"\n{'='*70}")
print("  TEST 0: BASELINE (0.5 vs 2.0 Hz)")
print(f"{'='*70}")
sig = make_binary_signal(0.5, 2.0)
X, Y, T = run_sim(sig)
r = classify_temporal(X, Y, T)
a, b = r['per_class'].get(-1, 0), r['per_class'].get(1, 0)
log(f"  Test: {r['test_acc']*100:.1f}%  A={a*100:.0f}% B={b*100:.0f}%  ({len(Y)} samples)")


# --- TEST 1: SETTLING TIME ---
print(f"\n{'='*70}")
print("  TEST 1: SETTLING TIME")
print(f"{'='*70}")
sig0 = make_binary_signal(0.5, 2.0)
X0, Y0, T0 = run_sim(sig0, total_time=400.0, verbose=False, t_skip=0)
log(f"  Samples with skip=0: {len(Y0)}")
log(f"\n   {'Skip(s)':>8}  {'Test%':>8}  {'A%':>6}  {'B%':>6}")
log(f"   {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
for skip in [0, 1, 2, 3, 5, 7, 10, 15, 20]:
    time_in_block = T0 % block_duration
    mask = time_in_block >= skip
    if np.sum(mask) < 20: continue
    r = classify_temporal(X0[mask], Y0[mask], T0[mask])
    a, b = r['per_class'].get(-1, 0), r['per_class'].get(1, 0)
    log(f"   {skip:8d}  {r['test_acc']*100:7.1f}%  {a*100:5.1f}%  {b*100:5.1f}%")


# --- TEST 2: FREQUENCY RESOLUTION ---
print(f"\n{'='*70}")
print("  TEST 2: FREQUENCY RESOLUTION")
print(f"{'='*70}")
log(f"\n  {'Pair':>16}  {'Δf':>6}  {'Test%':>8}  {'A%':>6}  {'B%':>6}  {'Time':>5}")
log(f"  {'─'*16}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*5}")
freq_pairs = [(0.5,2.0),(0.5,1.0),(0.5,0.8),(0.5,0.7),(0.5,0.6),(0.5,0.55),(0.5,0.53),(0.5,0.52),(0.5,0.51)]
for fa, fb in freq_pairs:
    sig = make_binary_signal(fa, fb)
    t0 = clock.time()
    X, Y, T = run_sim(sig, verbose=False)
    t1 = clock.time()
    r = classify_temporal(X, Y, T)
    a, b = r['per_class'].get(-1, 0), r['per_class'].get(1, 0)
    status = "✓" if r['test_acc'] >= 0.75 else ("~" if r['test_acc'] >= 0.6 else "✗")
    log(f"  {fa:.3f} vs {fb:.3f}  {fb-fa:5.3f}  {r['test_acc']*100:7.1f}%  {a*100:5.1f}%  {b*100:5.1f}%  {t1-t0:4.0f}s  {status}")


# --- TEST 3: MULTI-CLASS ---
print(f"\n{'='*70}")
print("  TEST 3: MULTI-CLASS")
print(f"{'='*70}")
for n_freq in [4, 8]:
    freqs = [0.5 + i*1.5/(n_freq-1) for i in range(n_freq)]
    total_t = max(400.0, stabilization_time + block_duration * n_freq * 2 + 50)
    def make_multi_sig(freqs, bd):
        def sig(t):
            block = int(t / bd)
            idx = block % len(freqs)
            return np.sin(2*np.pi*freqs[idx]*t), idx
        return sig
    sig = make_multi_sig(freqs, block_duration)
    t0 = clock.time()
    X, Y, T = run_sim(sig, total_time=total_t, verbose=False)
    t1 = clock.time()
    log(f"\n  {n_freq} classes: {[f'{f:.1f}' for f in freqs]}")
    r = classify_temporal(X, Y, T, n_train_blocks=n_freq)
    log(f"  Overall: {r['test_acc']*100:.1f}%  (chance={100/n_freq:.0f}%)")
    for cls in sorted(r['per_class'].keys()):
        log(f"    Class {cls} ({freqs[cls]:.1f}Hz): {r['per_class'][cls]*100:.1f}%")


# --- TEST 4: NOISE ROBUSTNESS ---
print(f"\n{'='*70}")
print("  TEST 4: NOISE ROBUSTNESS")
print(f"{'='*70}")
log(f"\n  {'Noise':>8}  {'SNR':>8}  {'Test%':>8}  {'A%':>6}  {'B%':>6}")
log(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
for nl in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    sig = make_binary_signal(0.5, 2.0, noise_level=nl)
    X, Y, T = run_sim(sig, verbose=False)
    r = classify_temporal(X, Y, T)
    a, b = r['per_class'].get(-1, 0), r['per_class'].get(1, 0)
    snr = f"{10*np.log10(0.5/(nl**2+1e-12)):6.0f}dB" if nl > 0 else "     ∞"
    log(f"  {nl:8.2f}  {snr}  {r['test_acc']*100:7.1f}%  {a*100:5.1f}%  {b*100:5.1f}%")


# --- TEST 5: FREQUENCY REGRESSION ---
print(f"\n{'='*70}")
print("  TEST 5: FREQUENCY REGRESSION (continuous manifold)")
print(f"{'='*70}")

train_freqs = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]
interp_freqs = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

reg_block = 30.0

def make_multi_reg(freqs, bd):
    def sig(t):
        block = int(t / bd)
        idx = block % len(freqs)
        return np.sin(2*np.pi*freqs[idx]*t), freqs[idx]  # continuous label
    return sig

# Training
print(f"\n  Training on {train_freqs}")
sig_train = make_multi_reg(train_freqs, reg_block)
total_t = stabilization_time + reg_block * len(train_freqs) * 3 + 50
X_tr_all, Y_tr_all, T_tr_all = run_sim(sig_train, total_time=total_t, verbose=False, blk_dur=reg_block, t_skip=5.0)
print(f"  {len(Y_tr_all)} samples")

# Temporal split
n_cyc_train = 2 * len(train_freqs)
block_idx = (T_tr_all / reg_block).astype(int)
first_block = int(stabilization_time / reg_block)
rel_block = block_idx - first_block
tr_mask = rel_block < n_cyc_train
val_mask = rel_block >= n_cyc_train

X_trn, Y_trn = X_tr_all[tr_mask], Y_tr_all[tr_mask]
X_val, Y_val = X_tr_all[val_mask], Y_tr_all[val_mask]

scaler = StandardScaler()
X_trn_sc = scaler.fit_transform(X_trn)
n_pca = min(50, len(X_trn), X_trn_sc.shape[1])
pca = PCA(n_components=n_pca)
X_trn_p = pca.fit_transform(X_trn_sc)

model = Ridge(alpha=ridge_alpha)
model.fit(X_trn_p, Y_trn)

pred_val = model.predict(pca.transform(scaler.transform(X_val)))
mae_val = np.mean(np.abs(pred_val - Y_val))
print(f"  Val MAE (same freqs, later time): {mae_val:.4f} Hz")

for freq in sorted(set(Y_val)):
    m = Y_val == freq
    if np.any(m):
        pm = np.mean(pred_val[m])
        err = abs(pm - freq)
        print(f"    {freq:.2f} Hz → {pm:.3f} (err={err:.4f})")

# Interpolation
print(f"\n  Interpolation on {interp_freqs}")
sig_interp = make_multi_reg(interp_freqs, reg_block)
total_t = stabilization_time + reg_block * len(interp_freqs) * 2 + 50
X_int, Y_int, T_int = run_sim(sig_interp, total_time=total_t, verbose=False, blk_dur=reg_block, t_skip=5.0)
pred_int = model.predict(pca.transform(scaler.transform(X_int)))
mae_int = np.mean(np.abs(pred_int - Y_int))
print(f"  Interp MAE: {mae_int:.4f} Hz")

for freq in sorted(set(Y_int)):
    m = Y_int == freq
    if np.any(m):
        pm = np.mean(pred_int[m])
        err = abs(pm - freq)
        status = "✓" if err < 0.05 else ("~" if err < 0.1 else "✗")
        print(f"    {freq:.2f} Hz → {pm:.3f} (err={err:.4f})  {status}")


# --- SUMMARY ---
print(f"\n{'='*70}")
print("  M36 FINAL SUMMARY")
print(f"{'='*70}")
print(f"  Architecture: Hopf oscillator, ω=0.3-3.0Hz, Q={Q_factor}")
print(f"  Readout: |Ψ|² snapshot ({N} dims, no windowing)")
print(f"  Classification MAE: val={mae_val:.4f}, interp={mae_int:.4f}")
baseline = 1.0 / 5.0  # old T=5s window
print(f"  Linear baseline (1/T=5s): {baseline:.2f} Hz")
if mae_int > 0:
    print(f"  Resonant amplification: {baseline/mae_int:.1f}×")
print()
