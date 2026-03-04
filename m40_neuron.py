"""
M40: BIOLOGICALLY REALISTIC FREQUENCY ENCODER
===============================================
Goal: Real-time frequency tracking, not attractor lookup.

Three changes from M38, each motivated by neuroscience:

1. SHORT WINDOW (5s → 0.2s)
   Auditory cortex resolves frequency in ~50-200ms.
   Long windows cause attractor stickiness — the system
   averages over changes instead of tracking them.
   Cost: less averaging → noisier features per sample.
   Gain: can track continuously changing frequency.

2. STRUCTURED LOCAL COUPLING (random → tonotopic nearest-neighbor)
   In the cochlea, hair cells couple to their immediate neighbors
   along the basilar membrane (frequency axis).
   Random coupling does nothing (confirmed by diagnostic).
   Local coupling creates traveling waves along the frequency axis —
   a frequency change at one neuron propagates to neighbors,
   sharpening the peak and improving discrimination.
   Implementation: W_local[i,j] = exp(-|i-j|^2 / (2*sigma^2))
   with sigma = 10 neurons (~0.06 octave bandwidth).

3. SWEEP TRAINING
   Train Ridge on continuously ramping frequency signals,
   not discrete blocks. This forces the readout to learn
   transient states, not just settled attractors.
   The model then generalizes to any continuous input.

Parameters adjusted for short window:
   - tau_adapt_vec: max 5.0s → 0.5s (faster adaptation)
   - stabilization_time: 120s → 60s (shorter warmup needed)
   - transition_skip: 10s → 2s (faster settling)
   - feature_sample_interval: every 10 steps → every 2 steps
     (need dense samples for sweep training)
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time as clock

# =============================================================
# M40 PARAMETERS
# =============================================================
N = 500
lam = 0.8
eps = 1e-6
dt = 0.05
target_energy = 2.5
input_gain = 1.5

omega_hz = np.logspace(np.log10(0.3), np.log10(3.0), N)
omega_vec = 2.0 * np.pi * omega_hz

# Coupling: local only, real positive
S_global = 0.0          # global random coupling OFF (confirmed useless)
S_local  = 0.15         # local tonotopic coupling ON
sigma_local = 10.0      # coupling range in neuron indices (~0.06 octave)

# Faster damping for quicker reset
gamma_vec = np.linspace(0.5, 3.0, N)   # raised floor: 0.1→0.5

# CRITICAL: Much faster adaptation time constants
# M38 had max 5.0s — that's why attractors were sticky
tau_adapt_vec = np.linspace(0.05, 0.5, N)   # 10x faster

kappa_adapt = 0.5; adapt_max = 2.0
xi_min, xi_max = 0.1, 3.0
alpha_base, alpha_max = 0.1, 0.3
target_lyap = 0.1; eta_alpha = 0.0005
lyap_window = 50                        # shorter window for faster chaos tracking

learning_end_time = 60.0               # shorter learning phase
learn_interval = 20
eta_hebb = 0.002; decay_hebb = 0.0001
noise_amp = 0.05

stabilization_time = 60.0             # 60s warmup (was 120s)
ridge_alpha = 1000.0
density = 0.02
block_duration = 50.0
transition_skip = 2.0                  # 2s skip (was 10s)

# SHORT WINDOW — the core change
window_seconds = 0.2                   # 200ms (was 5000ms)
window_steps = int(window_seconds / dt)  # = 4 steps
feature_sample_interval = 2            # harvest every 2 steps (was 10)


# =============================================================
# NETWORK BUILDER
# =============================================================

def build_network():
    """
    Local tonotopic coupling matrix.
    W_local[i,j] = Gaussian weight based on |i-j| along frequency axis.
    This creates cochlea-like nearest-neighbor coupling.
    """
    # Build dense Gaussian coupling, then sparsify
    idx = np.arange(N)
    ii, jj = np.meshgrid(idx, idx, indexing='ij')
    dist_sq = (ii - jj).astype(float)**2
    W_dense = np.exp(-dist_sq / (2.0 * sigma_local**2))
    np.fill_diagonal(W_dense, 0.0)  # no self-coupling

    # Normalize each row so spectral radius ≈ 1
    row_sums = W_dense.sum(axis=1, keepdims=True) + eps
    W_dense = W_dense / row_sums

    # Convert to sparse
    W_local = sp.csr_matrix(W_dense * (np.abs(W_dense) > 0.001))

    # Tonotopic input projection (same as M38)
    np.random.seed(42)
    W_in = np.zeros(N, dtype=complex)
    group_size = N // 5
    gains  = [2.0, 1.2, 0.5, 1.2, 0.8]
    phases = [0.0, 0.0, 0.0, np.pi, None]
    for g in range(5):
        sl = slice(g * group_size, (g + 1) * group_size)
        ph = (phases[g] if phases[g] is not None
              else np.random.uniform(0, 2*np.pi, group_size))
        base = (np.random.randn(group_size) + 1j*np.random.randn(group_size)) * 0.5
        W_in[sl] = base * gains[g] * np.exp(1j * ph)

    # Laplacian for diffusion term (built from same local structure)
    A_sym = (W_local + W_local.T) * 0.5
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    Delta = sp.diags(degrees) - A_sym

    return W_local, W_in, Delta


def energy_entropy(energy_series):
    E = energy_series - energy_series.min(axis=0, keepdims=True) + eps
    E_norm = E / (E.sum(axis=0, keepdims=True) + eps)
    H = -np.sum(E_norm * np.log(E_norm + eps), axis=0)
    return H / np.log(max(energy_series.shape[0], 2) + eps)


def get_derivative(Psi_curr, xi_vec, adapt_curr, alpha_curr,
                   noise_in, I_in, W_local, W_in, Delta):
    # Local tonotopic coupling
    W_eff = S_local * W_local
    D = W_eff @ Psi_curr

    num = np.real(Psi_curr.conj() * D)
    den = (np.abs(Psi_curr)**2) + (np.abs(D)**2) + eps
    R = num / den
    g_vec = xi_vec * np.tanh(1.0 - R) - lam
    effective_gamma = gamma_vec + adapt_curr

    dPsi = (1j * omega_vec * Psi_curr
            + (W_eff @ Psi_curr)
            + alpha_curr * (Delta @ Psi_curr)
            + (g_vec * Psi_curr)
            - (effective_gamma * (np.abs(Psi_curr)**2) * Psi_curr))

    dPsi += noise_amp * noise_in + W_in * I_in * input_gain
    return dPsi


# =============================================================
# SIMULATION ENGINE
# =============================================================

def run_sim_m40(signal_func, total_time=300.0, verbose=True,
                t_skip=None, blk_dur=None, sweep_mode=False):
    """
    sweep_mode=True: harvest every sample after stabilization,
                     no block gating (for sweep training/testing).
    sweep_mode=False: standard block-gated harvesting.
    """
    if t_skip is None: t_skip = transition_skip
    if blk_dur is None: blk_dur = block_duration

    steps = int(total_time / dt)
    W_local, W_in, Delta = build_network()

    Psi = (np.random.randn(N) + 1j*np.random.randn(N)) * 0.1
    xi_vec = np.ones(N) * 0.5
    A_vec = np.zeros(N)
    E_avg_vec = np.ones(N) * 0.1
    alpha_global = alpha_base
    Psi_ghost = Psi + (np.random.randn(N)+1j*np.random.randn(N)) * 1e-5
    prev_dist = np.linalg.norm(Psi_ghost - Psi)
    Lyap_history = []
    xi_frozen = False; xi_frozen_val = None

    psi_buffer   = np.zeros((window_steps, N), dtype=complex)
    phi_input_buffer = np.zeros((window_steps, 1))
    buf_idx = 0; buf_filled = False

    feats_plv = []; feats_ent = []; feats_spec = []
    targets_Y = []; harvest_T = []

    Wc = W_local.tocsr()

    for t in range(steps):
        ct = t * dt
        noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
        I_val, Y_val, freq = signal_func(ct)
        phi_in = (2*np.pi*freq*ct) % (2*np.pi) if freq > 0 else 0.0

        k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global,
                            noise_vec, I_val, Wc, W_in, Delta)
        k2 = get_derivative(Psi+0.5*dt*k1, xi_vec, A_vec, alpha_global,
                            noise_vec, I_val, Wc, W_in, Delta)
        k3 = get_derivative(Psi+0.5*dt*k2, xi_vec, A_vec, alpha_global,
                            noise_vec, I_val, Wc, W_in, Delta)
        k4 = get_derivative(Psi+dt*k3, xi_vec, A_vec, alpha_global,
                            noise_vec, I_val, Wc, W_in, Delta)
        Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

        k1g = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global,
                             noise_vec, 0, Wc, W_in, Delta)
        Psi_ghost = Psi_ghost + dt * k1g

        instant_energy = np.abs(Psi)**2
        E_avg_vec = 0.99*E_avg_vec + 0.01*instant_energy

        if ct >= stabilization_time and not xi_frozen:
            xi_frozen = True; xi_frozen_val = xi_vec.copy()
            if verbose: print(f"    Xi FROZEN at t={ct:.1f}s")

        if not xi_frozen:
            error_energy = target_energy - E_avg_vec
            rate = np.where(error_energy < 0, 0.002, 0.005)
            xi_vec = np.clip(xi_vec + rate * error_energy, xi_min, xi_max)
        else:
            xi_vec = xi_frozen_val.copy()

        excess_energy = np.maximum(0, E_avg_vec - target_energy)
        A_vec = np.clip(
            A_vec + dt*((kappa_adapt*excess_energy - A_vec)/tau_adapt_vec),
            0, adapt_max)

        current_dist = np.linalg.norm(Psi_ghost - Psi)
        if current_dist < 1e-7 or current_dist > 1.0:
            Psi_ghost = Psi + (np.random.randn(N)+1j*np.random.randn(N))*1e-4
            prev_dist = 1e-4
        else:
            Lyap_history.append(
                np.log(current_dist+1e-12) - np.log(prev_dist+1e-12))
            prev_dist = current_dist
        if len(Lyap_history) > lyap_window: Lyap_history.pop(0)
        lyap_smooth = np.mean(Lyap_history) if Lyap_history else 0.0
        alpha_global = np.clip(
            alpha_global + eta_alpha*(target_lyap - lyap_smooth),
            alpha_base, alpha_max)

        if ct < learning_end_time and (t % learn_interval == 0):
            rows, cols = W_local.nonzero()
            corr = Psi[rows] * np.conj(Psi[cols])
            update = np.real(eta_hebb * corr * np.abs(Psi[rows]) * np.abs(Psi[cols]))
            # Extract current weights as a plain numpy array
            current_weights = np.asarray(W_local[rows, cols]).flatten()
            new_weights = current_weights + update - decay_hebb * current_weights
            W_local = W_local.tolil()
            W_local[rows, cols] = np.abs(new_weights)  # keep positive
            W_local = W_local.tocsr()
            try:
                ev = sp.linalg.eigs(W_local, k=1, return_eigenvectors=False)
                if np.abs(ev[0]) > 0:
                    W_local = W_local * (0.9 / np.abs(ev[0]))
            except: pass
            Wc = W_local.tocsr()

        psi_buffer[buf_idx] = Psi.copy()
        phi_input_buffer[buf_idx] = phi_in
        buf_idx = (buf_idx + 1) % window_steps
        if t >= window_steps: buf_filled = True

        if ct > stabilization_time and buf_filled and (t % feature_sample_interval == 0):
            # Harvest condition
            if sweep_mode:
                should_harvest = True
            else:
                time_in_block = ct % blk_dur
                should_harvest = time_in_block >= t_skip

            if should_harvest:
                ordered_psi    = np.roll(psi_buffer, -buf_idx, axis=0)
                ordered_phi_in = np.roll(phi_input_buffer, -buf_idx, axis=0)

                # PLV
                phi_neuron = np.angle(ordered_psi)
                delta_phi  = np.angle(np.exp(1j*(phi_neuron - ordered_phi_in)))
                plv = np.abs(np.mean(np.exp(1j*delta_phi), axis=0))

                # Entropy
                energy_series = np.abs(ordered_psi)**2
                ent = energy_entropy(energy_series)

                # Spectral (adapted for short window — fewer freq bins)
                energy_centered = energy_series - energy_series.mean(axis=0, keepdims=True)
                fft_result = np.fft.rfft(energy_centered, axis=0)
                power = np.abs(fft_result)**2
                freqs_fft = np.fft.rfftfreq(window_steps, d=dt)
                # Broader bands since window is shorter
                bands = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)]
                spec_feats = []
                for f_lo, f_hi in bands:
                    mask = (freqs_fft >= f_lo) & (freqs_fft <= f_hi)
                    spec_feats.append(
                        np.mean(power[mask], axis=0) if np.any(mask)
                        else np.zeros(N))
                spec = np.concatenate(spec_feats)

                feats_plv.append(plv)
                feats_ent.append(ent)
                feats_spec.append(spec)
                targets_Y.append(Y_val)
                harvest_T.append(ct)

    return (np.array(feats_plv), np.array(feats_ent),
            np.array(feats_spec), np.array(targets_Y), np.array(harvest_T))


# =============================================================
# SIGNAL GENERATORS
# =============================================================

def make_block_signal(freqs, block_dur=50.0, noise_level=0.0):
    def sig(t):
        block = int(t / block_dur)
        idx = block % len(freqs)
        f = freqs[idx]
        I = np.sin(2*np.pi*f*t)
        if noise_level > 0: I += noise_level * np.random.randn()
        return I, idx, f
    return sig


def make_sweep_signal(f_start=0.5, f_end=2.0, sweep_dur=200.0,
                      warmup=None):
    """Continuous linear frequency ramp for training and testing."""
    if warmup is None: warmup = stabilization_time + 10.0
    def sig(t):
        if t < warmup:
            f = (f_start + f_end) / 2.0  # warmup at midpoint
        else:
            frac = min((t - warmup) / sweep_dur, 1.0)
            f = f_start + (f_end - f_start) * frac
        I = np.sin(2*np.pi*f*t)
        return I, f, f
    return sig


def make_multisweep_signal(f_start=0.5, f_end=2.0,
                            n_sweeps=4, sweep_dur=60.0, warmup=None):
    """
    Multiple back-and-forth sweeps for richer sweep training.
    Alternates: up, down, up, down...
    """
    if warmup is None: warmup = stabilization_time + 10.0
    def sig(t):
        if t < warmup:
            f = (f_start + f_end) / 2.0
        else:
            elapsed = t - warmup
            sweep_idx = int(elapsed / sweep_dur)
            frac = (elapsed % sweep_dur) / sweep_dur
            if sweep_idx % 2 == 0:  # up sweep
                f = f_start + (f_end - f_start) * frac
            else:  # down sweep
                f = f_end - (f_end - f_start) * frac
        I = np.sin(2*np.pi*f*t)
        return I, f, f
    return sig


# =============================================================
# CLASSIFIER / REGRESSOR
# =============================================================

def classify_temporal(X, Y, T, block_dur=50.0, n_train_blocks=4):
    block_idx  = (T / block_dur).astype(int)
    first_block = int(stabilization_time / block_dur)
    rel_block  = block_idx - first_block
    train_mask = rel_block < n_train_blocks
    test_mask  = rel_block >= n_train_blocks
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test,  Y_test  = X[test_mask],  Y[test_mask]
    if len(X_test) < 5 or len(X_train) < 5:
        return {'test_acc': 0, 'per_class': {}}
    classes = np.unique(Y_train)
    if len(classes) < 2: return {'test_acc': 0.5, 'per_class': {}}
    min_c = min(np.sum(Y_train == c) for c in classes)
    bal_idx = []
    rng = np.random.default_rng(42)
    for c in classes:
        ci = np.where(Y_train == c)[0]
        if len(ci) > min_c: ci = rng.choice(ci, size=min_c, replace=False)
        bal_idx.extend(ci)
    X_tr = X_train[np.sort(bal_idx)]; Y_tr = Y_train[np.sort(bal_idx)]
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_test)
    n_pca = min(50, len(X_tr), X_tr_sc.shape[1])
    pca   = PCA(n_components=n_pca)
    X_tr_p = pca.fit_transform(X_tr_sc)
    X_te_p = pca.transform(X_te_sc)
    model = Ridge(alpha=ridge_alpha)
    model.fit(X_tr_p, Y_tr)
    pred = model.predict(X_te_p)
    if len(classes) == 2:
        threshold = np.mean(classes)
        acc = np.mean((pred > threshold) == (Y_test > threshold))
        per_class = {c: np.mean((pred[Y_test==c] > threshold) == (c > threshold))
                     for c in classes if np.any(Y_test==c)}
    else:
        pred_classes = np.array(
            [classes[np.argmin(np.abs(p - classes))] for p in pred])
        acc = np.mean(pred_classes == Y_test)
        per_class = {c: np.mean(pred_classes[Y_test==c] == c)
                     for c in classes if np.any(Y_test==c)}
    return {'test_acc': acc, 'per_class': per_class}


def fit_regression(plv, ent, spec, Y):
    X = np.hstack([plv, ent, spec])
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    n_pca = min(50, X_sc.shape[0]-1, X_sc.shape[1])
    pca   = PCA(n_components=n_pca); X_p = pca.fit_transform(X_sc)
    model = Ridge(alpha=ridge_alpha); model.fit(X_p, Y)
    return model, scaler, pca


def predict_regression(plv, ent, spec, model, scaler, pca):
    X = np.hstack([plv, ent, spec])
    return model.predict(pca.transform(scaler.transform(X)))


if __name__ == "__main__":
    print("=" * 70)
    print("  M40: BIOLOGICALLY REALISTIC FREQUENCY ENCODER")
    print("  Short window (200ms) | Local coupling | Sweep training")
    print("=" * 70)

    # ----------------------------------------------------------
    # TEST 0: SANITY — does the short window still work for blocks?
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 0: BLOCK CLASSIFICATION (sanity check)")
    print(f"{'='*70}")

    np.random.seed(0)
    sig = make_block_signal([0.5, 2.0])
    plv, ent, spec, Y, T = run_sim_m40(sig, total_time=400.0)
    X = np.hstack([plv, ent, spec])
    r = classify_temporal(X, Y, T)
    print(f"  0.5 vs 2.0 Hz: {r['test_acc']*100:.1f}%  (M38 was 100%)")

    np.random.seed(1)
    sig = make_block_signal([0.5, 0.505])
    plv, ent, spec, Y, T = run_sim_m40(sig, total_time=400.0, verbose=False)
    r2 = classify_temporal(np.hstack([plv, ent, spec]), Y, T)
    print(f"  0.500 vs 0.505 Hz: {r2['test_acc']*100:.1f}%  (M38 was ~93%)")

    # ----------------------------------------------------------
    # TEST 1: SWEEP TRAINING + SWEEP TEST
    # The key test — can M40 track a continuously changing frequency?
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 1: SWEEP TRAINING + SWEEP TEST")
    print("  Train: 4 up/down sweeps 0.5→2.0→0.5 Hz")
    print("  Test:  fresh sweep — measures real-time tracking")
    print(f"{'='*70}")

    warmup = stabilization_time + 10.0
    sweep_dur = 60.0
    n_sweeps  = 4
    train_total = warmup + n_sweeps * sweep_dur + 10.0

    print("\n  Training sweep simulation...")
    np.random.seed(2)
    sig_train = make_multisweep_signal(n_sweeps=n_sweeps, sweep_dur=sweep_dur)
    plv_tr, ent_tr, spec_tr, Y_tr, T_tr = run_sim_m40(
        sig_train, total_time=train_total, sweep_mode=True)

    print(f"  Training samples: {len(Y_tr)}")
    model, scaler, pca = fit_regression(plv_tr, ent_tr, spec_tr, Y_tr)
    pred_tr = predict_regression(plv_tr, ent_tr, spec_tr, model, scaler, pca)
    mae_train = np.mean(np.abs(pred_tr - Y_tr))
    print(f"  Train MAE: {mae_train:.4f} Hz")

    print("\n  Test sweep simulation (fresh network, same signal)...")
    np.random.seed(3)
    sig_test = make_multisweep_signal(n_sweeps=2, sweep_dur=sweep_dur)
    test_total = warmup + 2 * sweep_dur + 10.0
    plv_te, ent_te, spec_te, Y_te, T_te = run_sim_m40(
        sig_test, total_time=test_total, sweep_mode=True, verbose=False)

    pred_te = predict_regression(plv_te, ent_te, spec_te, model, scaler, pca)
    mae_test = np.mean(np.abs(pred_te - Y_te))
    print(f"  Test samples: {len(Y_te)}")
    print(f"  Test MAE: {mae_test:.4f} Hz")
    print(f"  Fourier limit (200ms): {1/window_seconds:.2f} Hz")
    print(f"  Amplification: {(1/window_seconds)/mae_test:.1f}x")

    # Binned sweep accuracy
    print(f"\n  Binned MAE (test sweep):")
    print(f"  {'Freq range':>12}  {'MAE':>8}  {'Bias':>8}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}")
    bins = np.arange(0.5, 2.05, 0.15)
    for i in range(len(bins)-1):
        blo, bhi = bins[i], bins[i+1]
        m = (Y_te >= blo) & (Y_te < bhi)
        if np.sum(m) > 3:
            bin_mae  = np.mean(np.abs(pred_te[m] - Y_te[m]))
            bin_bias = np.mean(pred_te[m] - Y_te[m])
            print(f"  {blo:.2f}–{bhi:.2f} Hz   {bin_mae:8.4f}  {bin_bias:+8.4f}")

    # Compare to M38 sweep
    print(f"\n  M38 sweep MAE was: 0.7587 Hz (trained on blocks, tested on sweep)")
    print(f"  M40 sweep MAE is:  {mae_test:.4f} Hz (trained on sweeps, tested on sweeps)")

    # ----------------------------------------------------------
    # TEST 2: NOISE ROBUSTNESS (short window)
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 2: NOISE ROBUSTNESS")
    print(f"{'='*70}")
    print(f"  {'Noise σ':>8}  {'Acc%':>6}")
    print(f"  {'─'*8}  {'─'*6}")
    for noise_lvl in [0.0, 0.1, 0.3, 0.5, 1.0]:
        np.random.seed(4)
        sig = make_block_signal([0.5, 2.0], noise_level=noise_lvl)
        plv, ent, spec, Y, T = run_sim_m40(sig, total_time=400.0, verbose=False)
        r = classify_temporal(np.hstack([plv, ent, spec]), Y, T)
        print(f"  {noise_lvl:8.2f}  {r['test_acc']*100:5.1f}%")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Window:          {window_seconds*1000:.0f}ms  (was 5000ms)")
    print(f"  Coupling:        local tonotopic  (was random, useless)")
    print(f"  tau_adapt max:   {max(tau_adapt_vec):.2f}s  (was 5.0s)")
    print(f"  Sweep train MAE: {mae_train:.4f} Hz")
    print(f"  Sweep test MAE:  {mae_test:.4f} Hz")
    print(f"  Fourier limit:   {1/window_seconds:.2f} Hz")
    print(f"  Amplification:   {(1/window_seconds)/mae_test:.1f}x")
    print()
    if mae_test < 0.3:
        print("  ✓ Real-time tracking ACHIEVED")
        print("    System can follow continuous frequency changes.")
    elif mae_test < 0.7:
        print("  ~ Partial tracking — better than M38 sweep (0.76 Hz)")
        print("    Window or adaptation still too slow for full tracking.")
    else:
        print("  ✗ Tracking not yet achieved — further tuning needed.")