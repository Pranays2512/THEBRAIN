"""
M39: KURAMOTO ANALYSIS + UPGRADE
==================================
Purpose: Prove M38 is accidentally Kuramoto, then exploit it.

The Kuramoto model is:
  dθ_i/dt = ω_i + (K/N) * Σ_j sin(θ_j - θ_i)

Your M38 phase equation (derived from dPsi):
  dθ_i/dt = ω_i + S * Σ_j Re[W_ij * (Psi_j/Psi_i)]
           ≈ ω_i + S * Σ_j |W_ij| * sin(θ_j - θ_i + φ_ij)

where φ_ij = arg(W_ij) is a random phase offset from the complex W.

The problem: random φ_ij partially cancels the coupling.
             Real Kuramoto needs φ_ij = 0.

The fix (M39): Make W purely real-positive for coupling.
               This maximizes coherent phase interaction.

This script:
  PART 1 — Diagnose: measure order parameter r(t) for M38 vs M39
  PART 2 — Compare: resolution and robustness head-to-head
  PART 3 — Sweep: find optimal K (coupling strength) for M39
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time as clock

# =============================================================
# SHARED PARAMETERS (identical to M38)
# =============================================================
N = 500
lam = 0.8
eps = 1e-6
dt = 0.05
target_energy = 2.5
input_gain = 1.5
omega_hz = np.logspace(np.log10(0.3), np.log10(3.0), N)
omega_vec = 2.0 * np.pi * omega_hz
gamma_vec = np.linspace(0.1, 2.0, N)
tau_adapt_vec = np.linspace(0.2, 5.0, N)
kappa_adapt = 0.5; adapt_max = 2.0
xi_min, xi_max = 0.1, 3.0
alpha_base, alpha_max = 0.1, 0.3; target_lyap = 0.1; eta_alpha = 0.0005
lyap_window = 100
learning_end_time = 100.0; learn_interval = 20; eta_hebb = 0.002; decay_hebb = 0.0001
noise_amp = 0.05
stabilization_time = 120.0; ridge_alpha = 1000.0; density = 0.02
block_duration = 50.0; transition_skip = 10.0
window_seconds = 5.0; window_steps = int(window_seconds / dt)
feature_sample_interval = 10


def energy_entropy(energy_series):
    E = energy_series - energy_series.min(axis=0, keepdims=True) + eps
    E_norm = E / (E.sum(axis=0, keepdims=True) + eps)
    H = -np.sum(E_norm * np.log(E_norm + eps), axis=0)
    return H / np.log(energy_series.shape[0] + eps)


# =============================================================
# TWO NETWORK VARIANTS
# =============================================================

def build_network_m38():
    """Original M38: complex W (incoherent Kuramoto)"""
    W_real = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W_imag = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W = W_real + 1j * W_imag
    try:
        ev = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
        if np.abs(ev[0]) > 0: W = W * (0.9 / np.abs(ev[0]))
    except: pass
    np.random.seed(42)
    W_in = _make_tonotopic_Win()
    Delta = _make_laplacian()
    return W, W_in, Delta


def build_network_m39(S_global):
    """
    M39: REAL POSITIVE W (coherent Kuramoto).
    W_ij >= 0 everywhere → φ_ij = 0 → no phase cancellation.
    Coupling is now purely sin(θ_j - θ_i), true Kuramoto form.
    """
    W = sp.random(N, N, density=density, format='lil',
                  data_rvs=lambda s: np.abs(np.random.randn(s)))  # real positive
    # Symmetrize for undirected coupling (true Kuramoto)
    W = (W + W.T) * 0.5
    try:
        ev = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
        if np.abs(ev[0]) > 0: W = W * (0.9 / np.abs(ev[0]))
    except: pass
    np.random.seed(42)
    W_in = _make_tonotopic_Win()
    Delta = _make_laplacian()
    return W, W_in, Delta


def _make_tonotopic_Win():
    W_in = np.zeros(N, dtype=complex)
    group_size = N // 5
    gains  = [2.0, 1.2, 0.5, 1.2, 0.8]
    phases = [0.0, 0.0, 0.0, np.pi, None]
    for g in range(5):
        sl = slice(g * group_size, (g + 1) * group_size)
        ph = phases[g] if phases[g] is not None else np.random.uniform(0, 2*np.pi, group_size)
        base = (np.random.randn(group_size) + 1j * np.random.randn(group_size)) * 0.5
        W_in[sl] = base * gains[g] * np.exp(1j * ph)
    return W_in


def _make_laplacian():
    A_temp = sp.random(N, N, density=density, format='csr')
    A_temp = (A_temp + A_temp.T) * 0.5
    degrees = np.array(A_temp.sum(axis=1)).flatten()
    return sp.diags(degrees) - A_temp


def get_derivative(Psi_curr, xi_vec, adapt_curr, alpha_curr,
                   noise_in, I_in, W_curr, W_in, Delta, S_global):
    W_eff = S_global * W_curr
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


def kuramoto_order_parameter(Psi):
    """
    r = |mean(exp(iθ))| over all neurons.
    r=1: full synchronization
    r=0: incoherent
    r~0.3-0.6: partial sync (ideal for encoding)
    """
    phases = np.angle(Psi)
    return np.abs(np.mean(np.exp(1j * phases)))


# =============================================================
# SIMULATION ENGINE (variant-aware)
# =============================================================

def run_sim(signal_func, W, W_in, Delta, S_global,
            total_time=400.0, verbose=False,
            t_skip=transition_skip, blk_dur=block_duration,
            record_order_param=False):

    steps = int(total_time / dt)
    Psi = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1
    xi_vec = np.ones(N) * 0.5
    A_vec = np.zeros(N)
    E_avg_vec = np.ones(N) * 0.1
    alpha_global = alpha_base
    Psi_ghost = Psi + (np.random.randn(N) + 1j*np.random.randn(N)) * 1e-5
    prev_dist = np.linalg.norm(Psi_ghost - Psi)
    Lyap_history = []
    xi_frozen = False; xi_frozen_val = None

    psi_buffer = np.zeros((window_steps, N), dtype=complex)
    phi_input_buffer = np.zeros((window_steps, 1))
    buf_idx = 0; buf_filled = False

    feats_plv = []; feats_entropy = []; feats_spec = []
    targets_Y = []; harvest_T = []
    order_params = []  # Kuramoto order parameter over time

    Wc = W.tocsr()

    for t in range(steps):
        ct = t * dt
        noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
        I_val, Y_val, freq = signal_func(ct)
        phi_in = (2 * np.pi * freq * ct) % (2 * np.pi) if freq > 0 else 0.0

        k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta, S_global)
        k2 = get_derivative(Psi+0.5*dt*k1, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta, S_global)
        k3 = get_derivative(Psi+0.5*dt*k2, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta, S_global)
        k4 = get_derivative(Psi+dt*k3, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta, S_global)
        Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

        k1g = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta, S_global)
        Psi_ghost = Psi_ghost + dt * k1g

        if record_order_param and t % 20 == 0:
            order_params.append((ct, kuramoto_order_parameter(Psi)))

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
            # For real W (M39): take real part only to preserve dtype
            if W.dtype == np.float64:
                update = np.real(update)
            W[rows, cols] += update - decay_hebb * W[rows, cols]
            try:
                ev = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
                if np.abs(ev[0]) > 0: W = W * (0.9 / np.abs(ev[0]))
                Wc = W.tocsr()
            except: pass

        psi_buffer[buf_idx] = Psi.copy()
        phi_input_buffer[buf_idx] = phi_in
        buf_idx = (buf_idx + 1) % window_steps
        if t >= window_steps: buf_filled = True

        if ct > stabilization_time and buf_filled and (t % feature_sample_interval == 0):
            time_in_block = ct % blk_dur
            if time_in_block >= t_skip:
                ordered_psi = np.roll(psi_buffer, -buf_idx, axis=0)
                ordered_phi_in = np.roll(phi_input_buffer, -buf_idx, axis=0)

                phi_neuron = np.angle(ordered_psi)
                delta_phi = np.angle(np.exp(1j * (phi_neuron - ordered_phi_in)))
                plv = np.abs(np.mean(np.exp(1j * delta_phi), axis=0))

                energy_series = np.abs(ordered_psi)**2
                ent = energy_entropy(energy_series)

                energy_centered = energy_series - energy_series.mean(axis=0, keepdims=True)
                fft_result = np.fft.rfft(energy_centered, axis=0)
                power = np.abs(fft_result)**2
                freqs_fft = np.fft.rfftfreq(window_steps, d=dt)
                bands = [(0.3, 0.7), (0.8, 1.5), (1.5, 2.5), (2.5, 5.0)]
                spec_feats = []
                for f_lo, f_hi in bands:
                    mask = (freqs_fft >= f_lo) & (freqs_fft <= f_hi)
                    spec_feats.append(np.mean(power[mask], axis=0) if np.any(mask) else np.zeros(N))
                spec = np.concatenate(spec_feats)

                feats_plv.append(plv); feats_entropy.append(ent)
                feats_spec.append(spec); targets_Y.append(Y_val)
                harvest_T.append(ct)

    result = (np.array(feats_plv), np.array(feats_entropy),
              np.array(feats_spec), np.array(targets_Y), np.array(harvest_T))
    if record_order_param:
        return result, np.array(order_params)
    return result


def make_signal(freqs, block_dur=50.0):
    def sig(t):
        block = int(t / block_dur)
        idx = block % len(freqs)
        f = freqs[idx]
        return np.sin(2 * np.pi * f * t), idx, f
    return sig


def make_regression_signal(freqs, block_dur=30.0):
    def sig(t):
        block = int(t / block_dur)
        idx = block % len(freqs)
        f = freqs[idx]
        return np.sin(2 * np.pi * f * t), f, f
    return sig


def classify_temporal(X, Y, T, block_dur=50.0, n_train_blocks=4):
    block_idx = (T / block_dur).astype(int)
    first_block = int(stabilization_time / block_dur)
    rel_block = block_idx - first_block
    train_mask = rel_block < n_train_blocks
    test_mask  = rel_block >= n_train_blocks
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test,  Y_test  = X[test_mask],  Y[test_mask]
    if len(X_test) < 5 or len(X_train) < 5: return {'test_acc': 0, 'per_class': {}}
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
    X_tr_sc = scaler.fit_transform(X_tr); X_te_sc = scaler.transform(X_test)
    n_pca = min(50, len(X_tr), X_tr_sc.shape[1])
    pca = PCA(n_components=n_pca)
    X_tr_p = pca.fit_transform(X_tr_sc); X_te_p = pca.transform(X_te_sc)
    model = Ridge(alpha=ridge_alpha)
    model.fit(X_tr_p, Y_tr)
    pred = model.predict(X_te_p)
    if len(classes) == 2:
        threshold = np.mean(classes)
        acc = np.mean((pred > threshold) == (Y_test > threshold))
        per_class = {c: np.mean((pred[Y_test==c] > threshold) == (c > threshold))
                     for c in classes if np.any(Y_test==c)}
    else:
        pred_classes = np.array([classes[np.argmin(np.abs(p - classes))] for p in pred])
        acc = np.mean(pred_classes == Y_test)
        per_class = {c: np.mean(pred_classes[Y_test==c] == c)
                     for c in classes if np.any(Y_test==c)}
    return {'test_acc': acc, 'per_class': per_class}


def run_regression(plv, ent, spec, Y):
    X = np.hstack([plv, ent, spec])
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    pca = PCA(n_components=50); X_p = pca.fit_transform(X_sc)
    model = Ridge(alpha=ridge_alpha); model.fit(X_p, Y)
    pred = model.predict(X_p)
    return np.mean(np.abs(pred - Y)), scaler, pca, model


# =============================================================

if __name__ == "__main__":
    pass

if __name__ == "__main__":
    # PART 1: DIAGNOSE — ORDER PARAMETER COMPARISON
    # =============================================================
    print("=" * 70)
    print("  M39 KURAMOTO ANALYSIS")
    print("=" * 70)

    print("\n--- PART 1: ORDER PARAMETER DIAGNOSIS ---")
    print("  Measuring Kuramoto r(t) for M38 (complex W) vs M39 (real W)")
    print("  Target: r ~ 0.3-0.6 (partial sync = best encoding regime)")
    print()

    sig_base = make_signal([0.5, 2.0])

    # M38: complex W
    np.random.seed(0)
    W38, W_in38, Delta38 = build_network_m38()
    _, op38 = run_sim(sig_base, W38, W_in38, Delta38, S_global=0.12,
                      record_order_param=True, verbose=True)

    r38_stable = np.mean(op38[op38[:,0] > stabilization_time, 1])
    r38_all    = op38[:, 1]
    print(f"  M38 (complex W):  mean r = {r38_stable:.3f}  "
          f"[min={r38_all.min():.3f}, max={r38_all.max():.3f}]")

    # M39: real positive W
    np.random.seed(0)
    W39, W_in39, Delta39 = build_network_m39(S_global=0.12)
    _, op39 = run_sim(sig_base, W39, W_in39, Delta39, S_global=0.12,
                      record_order_param=True, verbose=True)

    r39_stable = np.mean(op39[op39[:,0] > stabilization_time, 1])
    r39_all    = op39[:, 1]
    print(f"  M39 (real W):     mean r = {r39_stable:.3f}  "
          f"[min={r39_all.min():.3f}, max={r39_all.max():.3f}]")

    print()
    if 0.25 <= r39_stable <= 0.65:
        print("  ✓ M39 is in the partial-sync regime (ideal for encoding)")
    elif r39_stable > 0.65:
        print("  ⚠ M39 over-synchronizing — reduce S_global")
    else:
        print("  ⚠ M39 under-synchronizing — increase S_global")


    # =============================================================
    # PART 2: HEAD-TO-HEAD COMPARISON
    # =============================================================
    print("\n--- PART 2: M38 vs M39 HEAD-TO-HEAD ---")
    print(f"  {'Test':30s}  {'M38':>8}  {'M39':>8}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}")

    # Baseline classification
    np.random.seed(1)
    W38, W_in38, Delta38 = build_network_m38()
    plv38, ent38, spec38, Y38, T38 = run_sim(sig_base, W38, W_in38, Delta38, S_global=0.12)
    r38 = classify_temporal(np.hstack([plv38, ent38, spec38]), Y38, T38)

    np.random.seed(1)
    W39, W_in39, Delta39 = build_network_m39(S_global=0.12)
    plv39, ent39, spec39, Y39, T39 = run_sim(sig_base, W39, W_in39, Delta39, S_global=0.12)
    r39 = classify_temporal(np.hstack([plv39, ent39, spec39]), Y39, T39)

    print(f"  {'Baseline 0.5 vs 2.0 Hz':30s}  {r38['test_acc']*100:7.1f}%  {r39['test_acc']*100:7.1f}%")

    # Resolution floor
    for fa, fb in [(0.5, 0.51), (0.5, 0.505), (0.5, 0.502)]:
        sig_res = make_signal([fa, fb])
        np.random.seed(2)
        W38, W_in38, Delta38 = build_network_m38()
        plv, ent, spec, Y, T = run_sim(sig_res, W38, W_in38, Delta38, S_global=0.12)
        r38_res = classify_temporal(np.hstack([plv, ent, spec]), Y, T)

        np.random.seed(2)
        W39, W_in39, Delta39 = build_network_m39(S_global=0.12)
        plv, ent, spec, Y, T = run_sim(sig_res, W39, W_in39, Delta39, S_global=0.12)
        r39_res = classify_temporal(np.hstack([plv, ent, spec]), Y, T)

        label = f"Resolution {fa:.3f} vs {fb:.3f}"
        print(f"  {label:30s}  {r38_res['test_acc']*100:7.1f}%  {r39_res['test_acc']*100:7.1f}%")

    # Regression MAE
    train_f  = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]
    interp_f = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

    sig_tr = make_regression_signal(train_f)
    np.random.seed(3)
    W38, W_in38, Delta38 = build_network_m38()
    plv, ent, spec, Y, _ = run_sim(sig_tr, W38, W_in38, Delta38, S_global=0.12, blk_dur=30.0)
    mae38, sc38, pc38, md38 = run_regression(plv, ent, spec, Y)

    sig_in = make_regression_signal(interp_f)
    np.random.seed(3)
    W38b, W_in38b, Delta38b = build_network_m38()
    plv_i, ent_i, spec_i, Y_i, _ = run_sim(sig_in, W38b, W_in38b, Delta38b, S_global=0.12, blk_dur=30.0)
    X_i38 = np.hstack([plv_i, ent_i, spec_i])
    pred_i38 = md38.predict(pc38.transform(sc38.transform(X_i38)))
    mae_interp38 = np.mean(np.abs(pred_i38 - Y_i))

    np.random.seed(3)
    W39, W_in39, Delta39 = build_network_m39(S_global=0.12)
    plv, ent, spec, Y, _ = run_sim(sig_tr, W39, W_in39, Delta39, S_global=0.12, blk_dur=30.0)
    mae39, sc39, pc39, md39 = run_regression(plv, ent, spec, Y)

    np.random.seed(3)
    W39b, W_in39b, Delta39b = build_network_m39(S_global=0.12)
    plv_i, ent_i, spec_i, Y_i, _ = run_sim(sig_in, W39b, W_in39b, Delta39b, S_global=0.12, blk_dur=30.0)
    X_i39 = np.hstack([plv_i, ent_i, spec_i])
    pred_i39 = md39.predict(pc39.transform(sc39.transform(X_i39)))
    mae_interp39 = np.mean(np.abs(pred_i39 - Y_i))

    print(f"  {'Regression Interp MAE':30s}  {mae_interp38:7.4f}Hz  {mae_interp39:7.4f}Hz")
    print(f"  {'Amplification vs Fourier':30s}  {(1/window_seconds)/mae_interp38:7.1f}x   {(1/window_seconds)/mae_interp39:7.1f}x")


    # =============================================================
    # PART 3: COUPLING STRENGTH SWEEP FOR M39
    # =============================================================
    print("\n--- PART 3: OPTIMAL COUPLING SWEEP (M39 real W) ---")
    print("  Finding K that maximizes resolution without over-sync")
    print(f"  {'K':>6}  {'r(order)':>10}  {'0.505 acc':>10}  {'Interp MAE':>12}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}")

    sig_res = make_signal([0.5, 0.505])
    sig_tr  = make_regression_signal(train_f)
    sig_in  = make_regression_signal(interp_f)

    for K in [0.06, 0.10, 0.12, 0.16, 0.20, 0.25]:
        np.random.seed(5)
        W, W_in, Delta = build_network_m39(K)

        # Order parameter
        _, op = run_sim(sig_base, W, W_in, Delta, S_global=K, record_order_param=True)
        r_val = np.mean(op[op[:,0] > stabilization_time, 1])

        # Resolution
        np.random.seed(5)
        W, W_in, Delta = build_network_m39(K)
        plv, ent, spec, Y, T = run_sim(sig_res, W, W_in, Delta, S_global=K)
        acc_res = classify_temporal(np.hstack([plv, ent, spec]), Y, T)['test_acc']

        # Regression
        np.random.seed(5)
        W, W_in, Delta = build_network_m39(K)
        plv, ent, spec, Y, _ = run_sim(sig_tr, W, W_in, Delta, S_global=K, blk_dur=30.0)
        mae_tr, sc, pc, md = run_regression(plv, ent, spec, Y)

        np.random.seed(5)
        W, W_in, Delta = build_network_m39(K)
        plv_i, ent_i, spec_i, Y_i, _ = run_sim(sig_in, W, W_in, Delta, S_global=K, blk_dur=30.0)
        X_i = np.hstack([plv_i, ent_i, spec_i])
        mae_i = np.mean(np.abs(md.predict(pc.transform(sc.transform(X_i))) - Y_i))

        flag = " ← sweet spot?" if 0.25 <= r_val <= 0.55 and acc_res > 0.85 else ""
        print(f"  {K:6.2f}  {r_val:10.3f}  {acc_res*100:9.1f}%  {mae_i:10.4f}Hz{flag}")


    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  M38 (complex W, S=0.12): Interp MAE={mae_interp38:.4f} Hz")
    print(f"  M39 (real W,    S=0.12): Interp MAE={mae_interp39:.4f} Hz")
    print(f"  Fourier limit: {1/window_seconds:.2f} Hz")
    print()
    print("  Kuramoto insight:")
    print("  Complex W has random phase offsets φ_ij that partially cancel coupling.")
    print("  Real positive W sets φ_ij=0 → coherent sin(θ_j-θ_i) interaction.")
    print("  This is the true Kuramoto regime and should sharpen the frequency manifold.")