"""
WINDOW SWEEP — Confirm Time-Frequency Tradeoff
================================================
Sweep window_seconds = [2, 5, 10, 20] against frequency pairs.
If Δf ≈ 1/T_window, we confirm the resolution limit is physics, not architecture.

Each window size needs a separate simulation because the rolling buffer
and feature extraction depend on window_steps.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time as clock

# Physics params (same as neuron.py M34)
N = 500; lam = 0.8; gamma = 0.5; eps = 1e-6; dt = 0.05
target_energy = 2.5; input_gain = 1.5
eta_xi_up = 0.005; eta_xi_down = 0.002; xi_min = 0.1; xi_max = 3.0
tau_adapt = 1.0; kappa_adapt = 0.5; adapt_max = 2.0
alpha_base = 0.1; alpha_max = 0.3; target_lyap = 0.1; eta_alpha = 0.0005
lyap_window = 100; S_global = 1.0
learning_end_time = 100.0; learn_interval = 20
eta_hebb = 0.002; decay_hebb = 0.0001; noise_amp = 0.05
stabilization_time = 120.0; energy_gate = 0.5
ridge_alpha = 1000.0; density = 0.02
block_duration = 50.0; transition_skip = 15.0


def build_network():
    W_real = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W_imag = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W = (W_real + 1j * W_imag)
    try:
        eigenvals = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
        if np.abs(eigenvals[0]) > 0:
            W = W * (0.9 / np.abs(eigenvals[0]))
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
    effective_gamma = gamma + adapt_curr
    dPsi = (1j*(W_eff @ Psi_curr) + alpha_curr*(Delta @ Psi_curr)
            + (g_vec * Psi_curr) - (effective_gamma * (np.abs(Psi_curr)**2) * Psi_curr))
    dPsi += noise_amp * noise_in + W_in * I_in * input_gain
    return dPsi


def run_and_extract(signal_fn, win_sec, total_time=400.0):
    """Run sim and extract energy variance features with given window size."""
    win_steps = int(win_sec / dt)
    sample_interval = max(1, win_steps // 4)  # ~4 samples per window
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

    psi_buffer = np.zeros((win_steps, N), dtype=complex)
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

        k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k2 = get_derivative(Psi+0.5*dt*k1, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k3 = get_derivative(Psi+0.5*dt*k2, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k4 = get_derivative(Psi+dt*k3, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

        k1g = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        k2g = get_derivative(Psi_ghost+0.5*dt*k1g, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        k3g = get_derivative(Psi_ghost+0.5*dt*k2g, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        k4g = get_derivative(Psi_ghost+dt*k3g, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        Psi_ghost = Psi_ghost + (dt/6.0)*(k1g+2*k2g+2*k3g+k4g)

        instant_energy = np.abs(Psi)**2
        E_avg_vec = 0.99*E_avg_vec + 0.01*instant_energy
        mean_energy = np.mean(E_avg_vec)

        if ct >= stabilization_time and not xi_frozen:
            xi_frozen = True
            xi_frozen_val = xi_vec.copy()
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
        A_vec = np.clip(A_vec + dt*((kappa_adapt*excess_energy - A_vec)/tau_adapt), 0, adapt_max)

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
        buf_idx = (buf_idx + 1) % win_steps
        if t >= win_steps: buf_filled = True

        if ct > stabilization_time and buf_filled and (t % sample_interval == 0):
            if abs(mean_energy - target_energy) < energy_gate:
                time_in_block = ct % block_duration
                if time_in_block < transition_skip:
                    continue

                ordered = np.roll(psi_buffer, -buf_idx, axis=0)
                energy_series = np.abs(ordered)**2

                features_evar.append(np.var(energy_series, axis=0))

                energy_centered = energy_series - energy_series.mean(axis=0, keepdims=True)
                fft_result = np.fft.rfft(energy_centered, axis=0)
                power = np.abs(fft_result)**2
                freqs_fft = np.fft.rfftfreq(win_steps, d=dt)
                bands = [(0.3, 0.7), (0.8, 1.5), (1.5, 2.5), (2.5, 5.0)]
                spec_feats = []
                for f_lo, f_hi in bands:
                    band_mask = (freqs_fft >= f_lo) & (freqs_fft <= f_hi)
                    if np.any(band_mask):
                        spec_feats.append(np.mean(power[band_mask], axis=0))
                    else:
                        spec_feats.append(np.zeros(N))
                features_spec.append(np.concatenate(spec_feats))

                targets_Y.append(Y_val)
                harvest_T.append(ct)

    X_ev = np.array(features_evar) if features_evar else np.zeros((0, N))
    X_sp = np.array(features_spec) if features_spec else np.zeros((0, 4*N))
    X_cb = np.hstack([X_ev, X_sp]) if len(X_ev) > 0 else np.zeros((0, 5*N))
    return X_ev, X_sp, X_cb, np.array(targets_Y), np.array(harvest_T)


def classify_temporal(X, Y, T, block_dur=50.0, n_train_blocks=4):
    block_idx = (T / block_dur).astype(int)
    first_block = int(stabilization_time / block_dur)
    rel_block = block_idx - first_block
    train_mask = rel_block < n_train_blocks
    test_mask = rel_block >= n_train_blocks
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]
    if len(X_train) < 5 or len(X_test) < 5:
        return 0.0, 0.0, {}
    classes = np.unique(Y_train)
    if len(classes) < 2:
        return 0.0, 0.0, {}
    min_c = min(np.sum(Y_train == c) for c in classes)
    bal_idx = []
    rng = np.random.default_rng(42)
    for c in classes:
        ci = np.where(Y_train == c)[0]
        if len(ci) > min_c: ci = rng.choice(ci, min_c, replace=False)
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
    pred_te = model.predict(X_te_p)
    acc_te = np.mean((pred_te > 0) == (Y_test > 0))
    per_class = {}
    for c in classes:
        m = Y_test == c
        per_class[c] = np.mean((pred_te[m] > 0) == (Y_test[m] > 0)) if np.any(m) else 0.0
    acc_tr = np.mean((model.predict(X_tr_p) > 0) == (Y_tr > 0))
    return acc_te, acc_tr, per_class


# =============================================================
# WINDOW SWEEP × FREQUENCY RESOLUTION
# =============================================================

freq_pairs = [
    (0.5, 2.0),    # easy
    (0.5, 1.0),    # moderate
    (0.5, 0.7),    # at M34 limit
    (0.5, 0.6),    # failed at 2s
    (0.5, 0.55),   # hard
    (0.5, 0.53),   # very hard
    (0.5, 0.52),   # extreme
    (0.5, 0.51),   # near-impossible
]

window_sizes = [2, 5, 10, 20]

print("=" * 80)
print("  WINDOW SWEEP × FREQUENCY RESOLUTION")
print("  Confirming: does Δf_min ≈ 1/T_window?")
print("=" * 80)

# Header
header = f"  {'Pair':>16}  {'Δf':>6}"
for ws in window_sizes:
    header += f"  {'T='+str(ws)+'s':>8}"
header += f"  {'Theory':>8}"
print(header)
print(f"  {'─'*16}  {'─'*6}" + f"  {'─'*8}" * len(window_sizes) + f"  {'─'*8}")

all_results = {}

for fa, fb in freq_pairs:
    row = f"  {fa:.2f} vs {fb:.2f}  {fb-fa:5.2f}"
    for ws in window_sizes:
        key = (fa, fb, ws)
        if key not in all_results:
            print(f"    Running {fa} vs {fb} Hz, window={ws}s...", end="", flush=True)

            def make_sig(t, _fa=fa, _fb=fb):
                block = int(t / block_duration) % 2
                if block == 0:
                    return np.sin(2*np.pi*_fa*t), -1
                else:
                    return np.sin(2*np.pi*_fb*t), 1

            t0 = clock.time()
            Xev, Xsp, Xcb, Y, T = run_and_extract(make_sig, win_sec=ws, total_time=400.0)
            t1 = clock.time()

            if len(Y) >= 20:
                # Try both evar and spectral, take best
                acc_ev, _, pc_ev = classify_temporal(Xev, Y, T)
                acc_sp, _, pc_sp = classify_temporal(Xsp, Y, T)
                acc_cb, _, pc_cb = classify_temporal(Xcb, Y, T)
                best_acc = max(acc_ev, acc_sp, acc_cb)
                best_name = ["evar", "spec", "comb"][[acc_ev, acc_sp, acc_cb].index(best_acc)]
            else:
                best_acc = 0.0
                best_name = "?"

            all_results[key] = (best_acc, best_name, len(Y))
            print(f" {best_acc*100:.0f}% ({best_name}, {len(Y)} samp, {t1-t0:.0f}s)")

        acc, name, n = all_results[key]
        mark = "✓" if acc >= 0.75 else ("~" if acc >= 0.6 else "✗")
        row += f"  {acc*100:5.0f}%{mark:>2}"

    # Theoretical minimum Δf for each window (1/T)
    row += f"  1/T→{1.0/window_sizes[-1]:.2f}"
    print(row)


# =============================================================
# ANALYSIS
# =============================================================
print("\n" + "=" * 80)
print("  ANALYSIS: Resolution Limit vs Window Size")
print("=" * 80)

for ws in window_sizes:
    theoretical = 1.0 / ws
    # Find smallest Δf with ≥75% accuracy
    best_df = None
    for fa, fb in freq_pairs:
        key = (fa, fb, ws)
        if key in all_results and all_results[key][0] >= 0.75:
            df = fb - fa
            if best_df is None or df < best_df:
                best_df = df

    if best_df is not None:
        ratio = best_df / theoretical
        print(f"  T={ws:2d}s:  theory Δf={theoretical:.3f}Hz  |  measured Δf={best_df:.3f}Hz  |  ratio={ratio:.1f}×")
    else:
        print(f"  T={ws:2d}s:  theory Δf={theoretical:.3f}Hz  |  measured: no pair ≥75%")

print("\n  If ratio stays ~constant across windows → physics-limited (confirmed)")
print("  If ratio decreases with larger windows → reservoir is adding value")

# Also test: does longer window help noise and multi-class?
print("\n" + "=" * 80)
print("  BONUS: Noise robustness with T=10s window")
print("=" * 80)

for nl in [0.0, 0.3, 0.5, 1.0, 2.0]:
    def make_noisy_sig(t, _nl=nl):
        block = int(t / block_duration) % 2
        if block == 0:
            return np.sin(2*np.pi*0.5*t) + _nl*np.random.randn(), -1
        else:
            return np.sin(2*np.pi*2.0*t) + _nl*np.random.randn(), 1

    Xev, Xsp, Xcb, Y, T = run_and_extract(make_noisy_sig, win_sec=10, total_time=400.0)
    if len(Y) >= 20:
        acc, _, pc = classify_temporal(Xev, Y, T)
        a = pc.get(-1, 0)
        b = pc.get(1, 0)
        snr = f"{10*np.log10(0.5/(nl**2+1e-12)):5.0f}dB" if nl > 0 else "    ∞"
        print(f"  noise={nl:.1f}  SNR={snr}  test={acc*100:.1f}%  A={a*100:.0f}% B={b*100:.0f}%")
    else:
        print(f"  noise={nl:.1f}  too few samples")

print("\nDone.")
