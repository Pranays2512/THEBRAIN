"""
FREQUENCY REGRESSION — Does the reservoir encode a continuous frequency manifold?
==================================================================================
Train on 10 discrete frequencies, test on UNSEEN interpolation and extrapolation.
If regression works, the reservoir has built a geometric map of frequency space.

Design:
  Train freqs: 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3 Hz
  Interp test: 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4 Hz (unseen)
  Extrap test: 0.3, 0.4, 2.5, 3.0 Hz (outside training range)

Uses M35 multi-timescale reservoir + combined (evar+spectral) features.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as clock

# =============================================================
# M35 PHYSICS
# =============================================================
N = 500; lam = 0.8; eps = 1e-6; dt = 0.05
gamma_vec = np.linspace(0.1, 2.0, N)
target_energy = 2.5; input_gain = 1.5
eta_xi_up = 0.005; eta_xi_down = 0.002; xi_min = 0.1; xi_max = 3.0
tau_adapt_vec = np.linspace(0.2, 5.0, N)
kappa_adapt = 0.5; adapt_max = 2.0
alpha_base = 0.1; alpha_max = 0.3; target_lyap = 0.1; eta_alpha = 0.0005
lyap_window = 100; S_global = 1.0
learning_end_time = 100.0; learn_interval = 20
eta_hebb = 0.002; decay_hebb = 0.0001; noise_amp = 0.05
stabilization_time = 120.0; energy_gate = 0.5
ridge_alpha = 1000.0; density = 0.02

# Regression-specific
block_duration = 30.0     # shorter blocks → more frequency visits
transition_skip = 10.0    # 10s skip, 20s usable per block
window_seconds = 5.0
window_steps = int(window_seconds / dt)
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


def get_derivative(Psi_curr, xi_curr, adapt_curr, alpha_curr, noise_in, I_in, W_curr, W_in, Delta):
    W_eff = S_global * W_curr
    D = W_eff @ Psi_curr
    num = np.real(Psi_curr.conj() * D)
    den = (np.abs(Psi_curr)**2) + (np.abs(D)**2) + eps
    R = num / den
    g_vec = xi_curr * np.tanh(1.0 - R) - lam
    effective_gamma = gamma_vec + adapt_curr
    dPsi = (1j*(W_eff @ Psi_curr) + alpha_curr*(Delta @ Psi_curr)
            + (g_vec * Psi_curr) - (effective_gamma * (np.abs(Psi_curr)**2) * Psi_curr))
    dPsi += noise_amp * noise_in + W_in * I_in * input_gain
    return dPsi


def run_sim_regression(freqs, total_time, verbose=True):
    """
    Run sim cycling through multiple frequencies.
    Returns combined features (evar+spectral), frequency labels (continuous), and times.
    """
    steps = int(total_time / dt)
    n_freqs = len(freqs)
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

    psi_buffer = np.zeros((window_steps, N), dtype=complex)
    buf_idx = 0
    buf_filled = False

    features_evar = []
    features_spec = []
    targets_freq = []  # continuous frequency values
    harvest_T = []

    for t in range(steps):
        ct = t * dt
        noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))

        # Signal: cycle through frequencies
        block = int(ct / block_duration)
        freq_idx = block % n_freqs
        current_freq = freqs[freq_idx]
        I_val = np.sin(2 * np.pi * current_freq * ct)

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
            if verbose: print(f"    Xi FROZEN at t={ct:.1f}s")

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

        psi_buffer[buf_idx] = Psi.copy()
        buf_idx = (buf_idx + 1) % window_steps
        if t >= window_steps: buf_filled = True

        # Harvest combined features
        if ct > stabilization_time and buf_filled and (t % feature_sample_interval == 0):
            if abs(mean_energy - target_energy) < energy_gate:
                time_in_block = ct % block_duration
                if time_in_block < transition_skip:
                    continue

                ordered = np.roll(psi_buffer, -buf_idx, axis=0)
                energy_series = np.abs(ordered)**2

                evar = np.var(energy_series, axis=0)

                energy_centered = energy_series - energy_series.mean(axis=0, keepdims=True)
                fft_result = np.fft.rfft(energy_centered, axis=0)
                power = np.abs(fft_result)**2
                freqs_fft = np.fft.rfftfreq(window_steps, d=dt)
                bands = [(0.3, 0.7), (0.8, 1.5), (1.5, 2.5), (2.5, 5.0)]
                spec_feats = []
                for f_lo, f_hi in bands:
                    band_mask = (freqs_fft >= f_lo) & (freqs_fft <= f_hi)
                    if np.any(band_mask):
                        spec_feats.append(np.mean(power[band_mask], axis=0))
                    else:
                        spec_feats.append(np.zeros(N))
                spec = np.concatenate(spec_feats)

                features_evar.append(evar)
                features_spec.append(spec)
                targets_freq.append(current_freq)
                harvest_T.append(ct)

        if verbose and t % 4000 == 0:
            print(f"    t={ct:6.1f}s  E={mean_energy:.3f}  samples={len(targets_freq)}")

    X_cb = np.hstack([np.array(features_evar), np.array(features_spec)])
    return X_cb, np.array(targets_freq), np.array(harvest_T)


# =============================================================
# EXPERIMENT
# =============================================================
print("=" * 70)
print("  FREQUENCY REGRESSION — Continuous Manifold Test")
print("=" * 70)

# Training frequencies (10 evenly spaced)
train_freqs = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]

# Interpolation test frequencies (between training freqs)
interp_freqs = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

# Extrapolation test frequencies (outside training range)
extrap_freqs = [0.3, 0.4, 2.5, 3.0]

print(f"\n  Train:  {train_freqs}")
print(f"  Interp: {interp_freqs}")
print(f"  Extrap: {extrap_freqs}")
print(f"  Block={block_duration}s, skip={transition_skip}s, window={window_seconds}s")

# -- 1. Run training simulation (10 freqs × 3 cycles = 30 blocks = 900s + 120s) --
print(f"\n{'─'*70}")
print("  Phase 1: Training simulation (10 frequencies)")
print(f"{'─'*70}")

train_total = stabilization_time + block_duration * len(train_freqs) * 3 + 50
t0 = clock.time()
X_train_all, Y_train_all, T_train_all = run_sim_regression(train_freqs, total_time=train_total)
t1 = clock.time()
print(f"  Sim: {t1-t0:.1f}s, {len(Y_train_all)} samples, {X_train_all.shape[1]} features")

# Temporal split: first 2 cycles for train, last for validation
n_blocks_per_cycle = len(train_freqs)
train_blocks = 2 * n_blocks_per_cycle  # first 20 blocks = train
block_idx = (T_train_all / block_duration).astype(int)
first_block = int(stabilization_time / block_duration)
rel_block = block_idx - first_block

train_mask = rel_block < train_blocks
val_mask = rel_block >= train_blocks

X_train = X_train_all[train_mask]
Y_train = Y_train_all[train_mask]
X_val = X_train_all[val_mask]
Y_val = Y_train_all[val_mask]

print(f"  Train samples: {len(Y_train)} (blocks 0-{train_blocks-1})")
print(f"  Val samples:   {len(Y_val)} (blocks {train_blocks}+)")
print(f"  Train freqs in data: {sorted(set(Y_train))}")
print(f"  Val freqs in data:   {sorted(set(Y_val))}")

# Fit regression
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
n_pca = min(50, len(X_train), X_tr_sc.shape[1])
pca = PCA(n_components=n_pca)
X_tr_p = pca.fit_transform(X_tr_sc)

model = Ridge(alpha=ridge_alpha)
model.fit(X_tr_p, Y_train)

# Validation on same freqs (temporal split)
pred_train = model.predict(X_tr_p)
X_val_p = pca.transform(scaler.transform(X_val))
pred_val = model.predict(X_val_p)

mae_train = np.mean(np.abs(pred_train - Y_train))
mae_val = np.mean(np.abs(pred_val - Y_val))

print(f"\n  TRAINING RESULTS:")
print(f"  Train MAE: {mae_train:.4f} Hz")
print(f"  Val MAE:   {mae_val:.4f} Hz  (same freqs, later time blocks)")

# Per-frequency validation accuracy
print(f"\n  {'Freq':>8}  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}  {'N':>4}")
print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*4}")
for freq in sorted(set(Y_val)):
    mask = Y_val == freq
    if np.any(mask):
        pred_mean = np.mean(pred_val[mask])
        pred_std = np.std(pred_val[mask])
        error = abs(pred_mean - freq)
        print(f"  {freq:8.2f}  {freq:8.2f}  {pred_mean:9.3f}±{pred_std:.3f}  {error:7.4f}  {np.sum(mask):4d}")


# -- 2. Interpolation test (unseen frequencies) --
print(f"\n{'─'*70}")
print("  Phase 2: Interpolation test (10 UNSEEN frequencies)")
print(f"{'─'*70}")

interp_total = stabilization_time + block_duration * len(interp_freqs) * 2 + 50
t0 = clock.time()
X_interp, Y_interp, T_interp = run_sim_regression(interp_freqs, total_time=interp_total, verbose=False)
t1 = clock.time()
print(f"  Sim: {t1-t0:.1f}s, {len(Y_interp)} samples")

X_interp_p = pca.transform(scaler.transform(X_interp))
pred_interp = model.predict(X_interp_p)
mae_interp = np.mean(np.abs(pred_interp - Y_interp))

print(f"  Interpolation MAE: {mae_interp:.4f} Hz")
print(f"\n  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}  {'N':>4}  {'Nearest Train':>14}")
print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*4}  {'─'*14}")
for freq in sorted(set(Y_interp)):
    mask = Y_interp == freq
    if np.any(mask):
        pred_mean = np.mean(pred_interp[mask])
        pred_std = np.std(pred_interp[mask])
        error = abs(pred_mean - freq)
        nearest = min(train_freqs, key=lambda x: abs(x - freq))
        status = "✓" if error < 0.05 else ("~" if error < 0.1 else "✗")
        print(f"  {freq:8.2f}  {pred_mean:9.3f}±{pred_std:.3f}  {error:7.4f}  {np.sum(mask):4d}  "
              f"({nearest:.1f}Hz, Δ={abs(freq-nearest):.1f})  {status}")


# -- 3. Extrapolation test (outside training range) --
print(f"\n{'─'*70}")
print("  Phase 3: Extrapolation test (4 frequencies OUTSIDE training range)")
print(f"{'─'*70}")

extrap_total = stabilization_time + block_duration * len(extrap_freqs) * 2 + 50
t0 = clock.time()
X_extrap, Y_extrap, T_extrap = run_sim_regression(extrap_freqs, total_time=extrap_total, verbose=False)
t1 = clock.time()
print(f"  Sim: {t1-t0:.1f}s, {len(Y_extrap)} samples")

X_extrap_p = pca.transform(scaler.transform(X_extrap))
pred_extrap = model.predict(X_extrap_p)
mae_extrap = np.mean(np.abs(pred_extrap - Y_extrap))

print(f"  Extrapolation MAE: {mae_extrap:.4f} Hz")
print(f"\n  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}  {'N':>4}  {'Status':>8}")
print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*4}  {'─'*8}")
for freq in sorted(set(Y_extrap)):
    mask = Y_extrap == freq
    if np.any(mask):
        pred_mean = np.mean(pred_extrap[mask])
        pred_std = np.std(pred_extrap[mask])
        error = abs(pred_mean - freq)
        direction = "below" if freq < min(train_freqs) else "above"
        status = "✓" if error < 0.1 else ("~" if error < 0.2 else "✗")
        print(f"  {freq:8.2f}  {pred_mean:9.3f}±{pred_std:.3f}  {error:7.4f}  {np.sum(mask):4d}  "
              f"{direction:>6}  {status}")


# =============================================================
# SUMMARY & PLOT
# =============================================================
print(f"\n{'='*70}")
print("  SUMMARY — FREQUENCY REGRESSION")
print(f"{'='*70}")
print(f"  Train MAE:          {mae_train:.4f} Hz")
print(f"  Val MAE (same f):   {mae_val:.4f} Hz")
print(f"  Interp MAE (new f): {mae_interp:.4f} Hz")
print(f"  Extrap MAE (OOD f): {mae_extrap:.4f} Hz")
print()

if mae_interp < 0.05:
    print("  ✓ SMOOTH MANIFOLD: Interpolation error < 0.05 Hz")
    print("    → Reservoir builds continuous geometric encoding of frequency")
elif mae_interp < 0.1:
    print("  ~ ROUGH MANIFOLD: Interpolation error 0.05-0.1 Hz")
    print("    → Partial geometric encoding with gaps")
else:
    print("  ✗ NO MANIFOLD: Interpolation error > 0.1 Hz")
    print("    → Reservoir encodes frequency categorically, not geometrically")

if mae_extrap < 0.2:
    print("  ✓ EXTRAPOLATION WORKS: The manifold extends beyond training range")
else:
    print("  ✗ EXTRAPOLATION FAILS: No frequency structure beyond training data")

# Linear comparison
print(f"\n  Linear baseline (1/T_window): Δf = {1/window_seconds:.2f} Hz")
print(f"  Reservoir interpolation:      Δf = {mae_interp:.4f} Hz")
if mae_interp > 0:
    print(f"  Nonlinear amplification:      {(1/window_seconds)/mae_interp:.1f}×")

# Plot: Predicted vs Actual
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Validation (seen frequencies)
ax1 = axes[0]
for freq in sorted(set(Y_val)):
    mask = Y_val == freq
    ax1.scatter(np.full(np.sum(mask), freq), pred_val[mask], alpha=0.3, s=10, color='cyan')
ax1.plot([0, 3], [0, 3], 'r--', alpha=0.5, label='Perfect')
ax1.set_xlabel('Actual Frequency (Hz)')
ax1.set_ylabel('Predicted Frequency (Hz)')
ax1.set_title(f'Validation (seen freqs)\nMAE={mae_val:.4f} Hz')
ax1.legend()
ax1.set_xlim(0, 3)
ax1.set_ylim(0, 3)

# Panel 2: Interpolation (unseen frequencies)
ax2 = axes[1]
for freq in sorted(set(Y_interp)):
    mask = Y_interp == freq
    ax2.scatter(np.full(np.sum(mask), freq), pred_interp[mask], alpha=0.3, s=10, color='lime')
# Show training frequencies
for f in train_freqs:
    ax2.axvline(x=f, color='cyan', alpha=0.2, linewidth=0.5)
ax2.plot([0, 3], [0, 3], 'r--', alpha=0.5, label='Perfect')
ax2.set_xlabel('Actual Frequency (Hz)')
ax2.set_ylabel('Predicted Frequency (Hz)')
ax2.set_title(f'Interpolation (UNSEEN freqs)\nMAE={mae_interp:.4f} Hz')
ax2.legend()
ax2.set_xlim(0, 3)
ax2.set_ylim(0, 3)

# Panel 3: Extrapolation
ax3 = axes[2]
for freq in sorted(set(Y_extrap)):
    mask = Y_extrap == freq
    ax3.scatter(np.full(np.sum(mask), freq), pred_extrap[mask], alpha=0.5, s=20, color='orange')
# Training range band
ax3.axvspan(min(train_freqs), max(train_freqs), alpha=0.1, color='cyan', label='Training range')
ax3.plot([0, 4], [0, 4], 'r--', alpha=0.5, label='Perfect')
ax3.set_xlabel('Actual Frequency (Hz)')
ax3.set_ylabel('Predicted Frequency (Hz)')
ax3.set_title(f'Extrapolation (OUTSIDE range)\nMAE={mae_extrap:.4f} Hz')
ax3.legend()
ax3.set_xlim(0, 4)
ax3.set_ylim(0, 4)

plt.tight_layout()
plt.savefig('/Users/pranay./Documents/THEBRAIN/freq_regression.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to freq_regression.png")
