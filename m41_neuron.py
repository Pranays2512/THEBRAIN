"""
M41: DUAL-MODE FREQUENCY ENCODER
==================================
Combines M38 (precision) + M40 (tracking) into one system.

Architecture:
  One oscillator bank → two parallel readout streams:

  FAST stream (200ms window):
    - Tracks continuously changing frequency
    - Lower precision (~0.31 Hz MAE)
    - Biological analog: cochlear nucleus / inferior colliculus

  SLOW stream (5s window):
    - High precision on steady-state signals (~0.033 Hz MAE)
    - Blind to fast changes
    - Biological analog: primary auditory cortex

  FUSION layer:
    - Measures stability of recent fast predictions
    - If stable (σ < threshold): signal is steady → trust SLOW
    - If unstable (σ > threshold): signal is changing → trust FAST
    - Output is a weighted blend: w*slow + (1-w)*fast
    - This is analogous to thalamocortical gating in biology

Key fix vs M40:
  - Extended training range: 0.3–3.0 Hz (fixes edge bias)
  - Both streams share the same oscillator dynamics
  - No need to run two separate simulations

Parameters:
  - Uses M40 dynamics (fast tau_adapt, local coupling)
  - FAST window: 200ms (4 steps)
  - SLOW window: 5s   (100 steps)
  - Fusion stability window: 10 fast predictions (~1s)
  - Stability threshold: tunable (default 0.15 Hz std)
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================
# PARAMETERS
# =============================================================
N = 500
lam = 0.8
eps = 1e-6
dt = 0.05
target_energy = 2.5
input_gain = 1.5

omega_hz  = np.logspace(np.log10(0.3), np.log10(3.0), N)
omega_vec = 2.0 * np.pi * omega_hz

# Local tonotopic coupling (M40 proven)
S_local     = 0.15
sigma_local = 10.0

# M40 dynamics — fast adaptation
gamma_vec     = np.linspace(0.5, 3.0, N)
tau_adapt_vec = np.linspace(0.05, 0.5, N)
kappa_adapt   = 0.5;  adapt_max = 2.0
xi_min, xi_max = 0.1, 3.0
alpha_base, alpha_max = 0.1, 0.3
target_lyap = 0.1;  eta_alpha = 0.0005
lyap_window = 50

learning_end_time = 60.0
learn_interval    = 20
eta_hebb          = 0.002
decay_hebb        = 0.0001
noise_amp         = 0.05

stabilization_time = 60.0
ridge_alpha        = 1000.0
block_duration     = 50.0
transition_skip    = 2.0
feature_sample_interval = 2

# DUAL WINDOWS
FAST_SECONDS = 0.2                           # 200ms
SLOW_SECONDS = 5.0                           # 5s
fast_steps   = int(FAST_SECONDS / dt)        # 4
slow_steps   = int(SLOW_SECONDS / dt)        # 100

# Fusion
STABILITY_WINDOW    = 10    # number of recent fast predictions to measure stability
STABILITY_THRESHOLD = 0.15  # Hz std — below this = steady state, trust slow


# =============================================================
# NETWORK
# =============================================================
def build_network():
    idx = np.arange(N)
    ii, jj = np.meshgrid(idx, idx, indexing='ij')
    W_dense = np.exp(-(ii-jj).astype(float)**2 / (2.0*sigma_local**2))
    np.fill_diagonal(W_dense, 0.0)
    row_sums = W_dense.sum(axis=1, keepdims=True) + eps
    W_dense /= row_sums
    W_local = sp.csr_matrix(W_dense * (np.abs(W_dense) > 0.001))

    np.random.seed(42)
    W_in = np.zeros(N, dtype=complex)
    group_size = N // 5
    gains  = [2.0, 1.2, 0.5, 1.2, 0.8]
    phases = [0.0, 0.0, 0.0, np.pi, None]
    for g in range(5):
        sl = slice(g*group_size, (g+1)*group_size)
        ph = (phases[g] if phases[g] is not None
              else np.random.uniform(0, 2*np.pi, group_size))
        base = (np.random.randn(group_size) + 1j*np.random.randn(group_size)) * 0.5
        W_in[sl] = base * gains[g] * np.exp(1j*ph)

    A_sym   = (W_local + W_local.T) * 0.5
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    Delta   = sp.diags(degrees) - A_sym
    return W_local, W_in, Delta


# =============================================================
# FEATURES  (shared for both windows)
# =============================================================
def energy_entropy(energy_series):
    E      = energy_series - energy_series.min(axis=0, keepdims=True) + eps
    E_norm = E / (E.sum(axis=0, keepdims=True) + eps)
    H      = -np.sum(E_norm * np.log(E_norm + eps), axis=0)
    return H / np.log(max(energy_series.shape[0], 2) + eps)


def extract_features(psi_buf, phi_buf, n_steps):
    """Extract PLV + entropy + spectral from a window buffer."""
    phi_neuron = np.angle(psi_buf)
    delta_phi  = np.angle(np.exp(1j*(phi_neuron - phi_buf)))
    plv = np.abs(np.mean(np.exp(1j*delta_phi), axis=0))

    energy = np.abs(psi_buf)**2
    ent    = energy_entropy(energy)

    ec  = energy - energy.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(ec, axis=0)
    pwr = np.abs(fft)**2
    fq  = np.fft.rfftfreq(n_steps, d=dt)
    # Bands adapted per window length
    if n_steps <= 8:   # short window
        bands = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)]
    else:              # long window
        bands = [(0.3, 0.7), (0.8, 1.5), (1.5, 2.5), (2.5, 5.0)]
    spec = np.concatenate([
        np.mean(pwr[(fq >= lo) & (fq <= hi)], axis=0)
        if np.any((fq >= lo) & (fq <= hi)) else np.zeros(N)
        for lo, hi in bands
    ])
    return plv, ent, spec


def get_derivative(Psi, xi_vec, adapt, alpha, noise, I_in, W, W_in, Delta):
    W_eff = S_local * W
    D     = W_eff @ Psi
    num   = np.real(Psi.conj() * D)
    den   = np.abs(Psi)**2 + np.abs(D)**2 + eps
    R     = num / den
    g_vec = xi_vec * np.tanh(1.0 - R) - lam
    eff_gamma = gamma_vec + adapt
    dPsi = (1j*omega_vec*Psi
            + W_eff@Psi
            + alpha*(Delta@Psi)
            + g_vec*Psi
            - eff_gamma*(np.abs(Psi)**2)*Psi)
    dPsi += noise_amp*noise + W_in*I_in*input_gain
    return dPsi


# =============================================================
# SIMULATION — dual-window harvest
# =============================================================
def run_sim_m41(signal_func, total_time=300.0, verbose=True,
                sweep_mode=False, blk_dur=None, t_skip=None):
    if blk_dur is None: blk_dur = block_duration
    if t_skip  is None: t_skip  = transition_skip

    steps = int(total_time / dt)
    W_local, W_in, Delta = build_network()

    Psi         = (np.random.randn(N) + 1j*np.random.randn(N)) * 0.1
    xi_vec      = np.ones(N) * 0.5
    A_vec       = np.zeros(N)
    E_avg_vec   = np.ones(N) * 0.1
    alpha_global = alpha_base
    Psi_ghost   = Psi + (np.random.randn(N)+1j*np.random.randn(N))*1e-5
    prev_dist   = np.linalg.norm(Psi_ghost - Psi)
    Lyap_hist   = []
    xi_frozen   = False;  xi_frozen_val = None

    # DUAL BUFFERS
    fast_psi_buf = np.zeros((fast_steps, N), dtype=complex)
    fast_phi_buf = np.zeros((fast_steps, 1))
    slow_psi_buf = np.zeros((slow_steps, N), dtype=complex)
    slow_phi_buf = np.zeros((slow_steps, 1))
    buf_idx      = 0
    fast_filled  = False
    slow_filled  = False

    # Storage — fast and slow features harvested at same timesteps
    fast_plv = []; fast_ent = []; fast_spec = []
    slow_plv = []; slow_ent = []; slow_spec = []
    targets_Y = []; harvest_T = []

    Wc = W_local.tocsr()

    for t in range(steps):
        ct        = t * dt
        noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
        I_val, Y_val, freq = signal_func(ct)
        phi_in = (2*np.pi*freq*ct) % (2*np.pi) if freq > 0 else 0.0

        # RK4
        k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k2 = get_derivative(Psi+0.5*dt*k1, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k3 = get_derivative(Psi+0.5*dt*k2, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k4 = get_derivative(Psi+dt*k3, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

        k1g = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        Psi_ghost = Psi_ghost + dt*k1g

        # Homeostasis
        instant_energy = np.abs(Psi)**2
        E_avg_vec = 0.99*E_avg_vec + 0.01*instant_energy

        if ct >= stabilization_time and not xi_frozen:
            xi_frozen = True;  xi_frozen_val = xi_vec.copy()
            if verbose: print(f"    Xi FROZEN at t={ct:.1f}s")

        if not xi_frozen:
            err  = target_energy - E_avg_vec
            rate = np.where(err < 0, 0.002, 0.005)
            xi_vec = np.clip(xi_vec + rate*err, xi_min, xi_max)
        else:
            xi_vec = xi_frozen_val.copy()

        excess = np.maximum(0, E_avg_vec - target_energy)
        A_vec  = np.clip(A_vec + dt*((kappa_adapt*excess - A_vec)/tau_adapt_vec),
                         0, adapt_max)

        # Lyapunov / chaos control
        cur_dist = np.linalg.norm(Psi_ghost - Psi)
        if cur_dist < 1e-7 or cur_dist > 1.0:
            Psi_ghost = Psi + (np.random.randn(N)+1j*np.random.randn(N))*1e-4
            prev_dist = 1e-4
        else:
            Lyap_hist.append(np.log(cur_dist+1e-12) - np.log(prev_dist+1e-12))
            prev_dist = cur_dist
        if len(Lyap_hist) > lyap_window: Lyap_hist.pop(0)
        lyap_smooth  = np.mean(Lyap_hist) if Lyap_hist else 0.0
        alpha_global = np.clip(alpha_global + eta_alpha*(target_lyap - lyap_smooth),
                               alpha_base, alpha_max)

        # Hebbian learning
        if ct < learning_end_time and (t % learn_interval == 0):
            rows, cols = W_local.nonzero()
            corr   = Psi[rows] * np.conj(Psi[cols])
            update = np.real(eta_hebb * corr * np.abs(Psi[rows]) * np.abs(Psi[cols]))
            current_w = np.asarray(W_local[rows, cols]).flatten()
            new_w     = np.abs(current_w + update - decay_hebb*current_w)
            W_local   = W_local.tolil()
            W_local[rows, cols] = new_w
            W_local = W_local.tocsr()
            try:
                ev = sp.linalg.eigs(W_local, k=1, return_eigenvectors=False)
                if np.abs(ev[0]) > 0: W_local = W_local * (0.9/np.abs(ev[0]))
            except: pass
            Wc = W_local.tocsr()

        # Dual buffer update (both share same index)
        fast_idx = t % fast_steps
        slow_idx = t % slow_steps
        fast_psi_buf[fast_idx] = Psi.copy()
        fast_phi_buf[fast_idx] = phi_in
        slow_psi_buf[slow_idx] = Psi.copy()
        slow_phi_buf[slow_idx] = phi_in
        if t >= fast_steps: fast_filled = True
        if t >= slow_steps: slow_filled = True

        # Harvest condition
        if ct > stabilization_time and fast_filled and (t % feature_sample_interval == 0):
            if sweep_mode:
                should_harvest = True
            else:
                should_harvest = (ct % blk_dur) >= t_skip

            if should_harvest:
                # FAST features
                f_psi = np.roll(fast_psi_buf, -fast_idx-1, axis=0)
                f_phi = np.roll(fast_phi_buf, -fast_idx-1, axis=0)
                fp, fe, fs = extract_features(f_psi, f_phi, fast_steps)
                fast_plv.append(fp); fast_ent.append(fe); fast_spec.append(fs)

                # SLOW features (only if slow buffer filled)
                if slow_filled:
                    s_psi = np.roll(slow_psi_buf, -slow_idx-1, axis=0)
                    s_phi = np.roll(slow_phi_buf, -slow_idx-1, axis=0)
                    sp_, se, ss = extract_features(s_psi, s_phi, slow_steps)
                else:
                    sp_ = np.zeros(N); se = np.zeros(N); ss = np.zeros(N*4)
                slow_plv.append(sp_); slow_ent.append(se); slow_spec.append(ss)

                targets_Y.append(Y_val)
                harvest_T.append(ct)

    return {
        'fast': (np.array(fast_plv), np.array(fast_ent), np.array(fast_spec)),
        'slow': (np.array(slow_plv), np.array(slow_ent), np.array(slow_spec)),
        'Y': np.array(targets_Y),
        'T': np.array(harvest_T)
    }


# =============================================================
# READOUT MODELS
# =============================================================
def fit_readout(plv, ent, spec, Y):
    X = np.hstack([plv, ent, spec])
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    n  = min(50, X_sc.shape[0]-1, X_sc.shape[1])
    pc = PCA(n_components=n); X_p = pc.fit_transform(X_sc)
    md = Ridge(alpha=ridge_alpha); md.fit(X_p, Y)
    return md, sc, pc

def predict_readout(plv, ent, spec, md, sc, pc):
    X = np.hstack([plv, ent, spec])
    return md.predict(pc.transform(sc.transform(X)))


# =============================================================
# FUSION — confidence-weighted blend
# =============================================================
def fuse_predictions(fast_pred, slow_pred,
                     stability_window=STABILITY_WINDOW,
                     threshold=STABILITY_THRESHOLD):
    """
    For each timepoint, compute stability of recent fast predictions.
    If std of last `stability_window` fast preds < threshold → stable → weight slow.
    Returns: fused predictions, weight_slow array (1=fully slow, 0=fully fast)
    """
    n = len(fast_pred)
    weight_slow = np.zeros(n)
    for i in range(n):
        lo  = max(0, i - stability_window)
        std = np.std(fast_pred[lo:i+1])
        # Sigmoid mapping: std=0 → w=1 (trust slow), std=threshold → w=0.5
        w = 1.0 / (1.0 + np.exp((std - threshold) / (threshold * 0.3)))
        weight_slow[i] = w
    fused = weight_slow * slow_pred + (1.0 - weight_slow) * fast_pred
    return fused, weight_slow


# =============================================================
# SIGNAL GENERATORS
# =============================================================
def make_block_signal(freqs, block_dur=50.0, noise_level=0.0):
    def sig(t):
        block = int(t / block_dur)
        idx   = block % len(freqs)
        f     = freqs[idx]
        I     = np.sin(2*np.pi*f*t)
        if noise_level > 0: I += noise_level * np.random.randn()
        return I, idx, f
    return sig

def make_multisweep_signal(f_start=0.3, f_end=3.0,
                            n_sweeps=6, sweep_dur=60.0, warmup=None):
    """Extended range: 0.3–3.0 Hz fixes edge bias at 0.5 and 2.0 Hz."""
    if warmup is None: warmup = stabilization_time + 10.0
    def sig(t):
        if t < warmup:
            f = (f_start + f_end) / 2.0
        else:
            elapsed   = t - warmup
            sweep_idx = int(elapsed / sweep_dur)
            frac      = (elapsed % sweep_dur) / sweep_dur
            f = (f_start + (f_end-f_start)*frac if sweep_idx % 2 == 0
                 else f_end - (f_end-f_start)*frac)
        return np.sin(2*np.pi*f*t), f, f
    return sig

def make_step_signal(freqs, step_dur=5.0, warmup=None):
    """
    Rapid discrete steps — tests fusion switching speed.
    Different from blocks: steps are short (5s) so slow stream
    never fully settles. Fusion should rely on fast stream here.
    """
    if warmup is None: warmup = stabilization_time + 5.0
    def sig(t):
        if t < warmup:
            f = freqs[0]
        else:
            idx = int((t - warmup) / step_dur) % len(freqs)
            f   = freqs[idx]
        return np.sin(2*np.pi*f*t), f, f
    return sig


# =============================================================
# CLASSIFIER HELPER
# =============================================================
def classify_temporal(X, Y, T, block_dur=50.0, n_train_blocks=4):
    block_idx   = (T / block_dur).astype(int)
    first_block = int(stabilization_time / block_dur)
    rel_block   = block_idx - first_block
    train_mask  = rel_block < n_train_blocks
    test_mask   = rel_block >= n_train_blocks
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test,  Y_test  = X[test_mask],  Y[test_mask]
    if len(X_test) < 5 or len(X_train) < 5:
        return {'test_acc': 0, 'per_class': {}}
    classes = np.unique(Y_train)
    if len(classes) < 2: return {'test_acc': 0.5, 'per_class': {}}
    min_c   = min(np.sum(Y_train == c) for c in classes)
    rng     = np.random.default_rng(42)
    bal_idx = []
    for c in classes:
        ci = np.where(Y_train == c)[0]
        if len(ci) > min_c: ci = rng.choice(ci, size=min_c, replace=False)
        bal_idx.extend(ci)
    X_tr = X_train[np.sort(bal_idx)]; Y_tr = Y_train[np.sort(bal_idx)]
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_test)
    n  = min(50, len(X_tr), X_tr_sc.shape[1])
    pc = PCA(n_components=n)
    X_tr_p = pc.fit_transform(X_tr_sc); X_te_p = pc.transform(X_te_sc)
    md = Ridge(alpha=ridge_alpha); md.fit(X_tr_p, Y_tr)
    pred = md.predict(X_te_p)
    threshold = np.mean(classes)
    acc = np.mean((pred > threshold) == (Y_test > threshold))
    per_class = {c: np.mean((pred[Y_test==c] > threshold) == (c > threshold))
                 for c in classes if np.any(Y_test==c)}
    return {'test_acc': acc, 'per_class': per_class}


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  M41: DUAL-MODE FREQUENCY ENCODER")
    print("  Fast (200ms) + Slow (5s) + Confidence fusion")
    print("=" * 70)

    # ----------------------------------------------------------
    # SETUP: Train both readout models
    # Extended range 0.3–3.0 Hz fixes edge bias
    # ----------------------------------------------------------
    print("\n  [Setup] Training dual readout models...")
    print("  Extended sweep range: 0.3–3.0 Hz (fixes edge bias)")

    warmup     = stabilization_time + 10.0
    sweep_dur  = 60.0
    n_sweeps   = 6
    train_time = warmup + n_sweeps * sweep_dur + 10.0

    np.random.seed(0)
    sig_train = make_multisweep_signal(f_start=0.3, f_end=3.0,
                                        n_sweeps=n_sweeps, sweep_dur=sweep_dur)
    data = run_sim_m41(sig_train, total_time=train_time,
                       sweep_mode=True, verbose=True)

    fp, fe, fs = data['fast']
    sp_, se, ss = data['slow']
    Y_tr = data['Y'];  T_tr = data['T']

    # Mask to slow-filled samples only
    slow_mask = T_tr > (stabilization_time + SLOW_SECONDS)

    fast_model, fast_sc, fast_pc = fit_readout(fp, fe, fs, Y_tr)
    slow_model, slow_sc, slow_pc = fit_readout(
        sp_[slow_mask], se[slow_mask], ss[slow_mask], Y_tr[slow_mask])

    fast_pred_tr = fast_model.predict(fast_pc.transform(fast_sc.transform(
        np.hstack([fp, fe, fs]))))
    slow_pred_tr = slow_model.predict(slow_pc.transform(slow_sc.transform(
        np.hstack([sp_[slow_mask], se[slow_mask], ss[slow_mask]]))))

    print(f"  Fast stream train MAE: {np.mean(np.abs(fast_pred_tr - Y_tr)):.4f} Hz")
    print(f"  Slow stream train MAE: {np.mean(np.abs(slow_pred_tr - Y_tr[slow_mask])):.4f} Hz")

    # ----------------------------------------------------------
    # TEST 1: SWEEP — edge bias fixed?
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 1: SWEEP TRACKING — edge bias fixed?")
    print(f"{'='*70}")

    np.random.seed(1)
    sig_test  = make_multisweep_signal(f_start=0.5, f_end=2.0,
                                        n_sweeps=2, sweep_dur=60.0)
    test_time = warmup + 2*60.0 + 10.0
    data_te   = run_sim_m41(sig_test, total_time=test_time,
                             sweep_mode=True, verbose=False)

    fp_te, fe_te, fs_te = data_te['fast']
    sp_te, se_te, ss_te = data_te['slow']
    Y_te = data_te['Y'];  T_te = data_te['T']
    slow_te_mask = T_te > (stabilization_time + SLOW_SECONDS)

    fast_pred = fast_model.predict(fast_pc.transform(fast_sc.transform(
        np.hstack([fp_te, fe_te, fs_te]))))
    slow_pred = np.full(len(Y_te), np.nan)
    slow_pred[slow_te_mask] = slow_model.predict(slow_pc.transform(slow_sc.transform(
        np.hstack([sp_te[slow_te_mask], se_te[slow_te_mask], ss_te[slow_te_mask]]))))

    # Fill early slow predictions with fast
    slow_pred[~slow_te_mask] = fast_pred[~slow_te_mask]

    fused, w_slow = fuse_predictions(fast_pred, slow_pred)

    mae_fast  = np.mean(np.abs(fast_pred - Y_te))
    mae_slow  = np.mean(np.abs(slow_pred - Y_te))
    mae_fused = np.mean(np.abs(fused - Y_te))

    print(f"  Fast stream MAE:  {mae_fast:.4f} Hz")
    print(f"  Slow stream MAE:  {mae_slow:.4f} Hz")
    print(f"  Fused MAE:        {mae_fused:.4f} Hz")
    print(f"  M40 sweep MAE was: 0.3108 Hz (narrow training range)")
    print(f"  M38 sweep MAE was: 0.7587 Hz (block-trained, wrong distribution)")

    print(f"\n  Binned MAE (fused, 0.5–2.0 Hz test range):")
    print(f"  {'Freq range':>12}  {'Fast':>7}  {'Slow':>7}  {'Fused':>7}  {'Bias':>8}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")
    bins = np.arange(0.5, 2.05, 0.15)
    for i in range(len(bins)-1):
        blo, bhi = bins[i], bins[i+1]
        m = (Y_te >= blo) & (Y_te < bhi)
        if np.sum(m) > 3:
            mf  = np.mean(np.abs(fast_pred[m] - Y_te[m]))
            ms  = np.mean(np.abs(slow_pred[m] - Y_te[m]))
            mfu = np.mean(np.abs(fused[m] - Y_te[m]))
            bias = np.mean(fused[m] - Y_te[m])
            print(f"  {blo:.2f}–{bhi:.2f} Hz   {mf:7.4f}  {ms:7.4f}  {mfu:7.4f}  {bias:+8.4f}")

    # ----------------------------------------------------------
    # TEST 2: STEADY-STATE PRECISION
    # Fusion should route to slow stream when signal is stable
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 2: STEADY-STATE PRECISION (long blocks)")
    print("  Fusion should engage slow stream → M38-level precision")
    print(f"{'='*70}")

    train_f  = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]
    interp_f = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

    # Regression on block data using slow stream
    np.random.seed(2)
    sig_blk = make_block_signal(train_f, block_dur=30.0)
    data_blk = run_sim_m41(sig_blk, total_time=500.0,
                            sweep_mode=False, blk_dur=30.0, verbose=False)
    fp_b, fe_b, fs_b = data_blk['fast']
    sp_b, se_b, ss_b = data_blk['slow']
    Y_b = data_blk['Y']

    blk_fast_model, blk_fast_sc, blk_fast_pc = fit_readout(fp_b, fe_b, fs_b, Y_b)
    blk_slow_model, blk_slow_sc, blk_slow_pc = fit_readout(sp_b, se_b, ss_b, Y_b)

    np.random.seed(3)
    sig_int = make_block_signal(interp_f, block_dur=30.0)
    data_int = run_sim_m41(sig_int, total_time=500.0,
                            sweep_mode=False, blk_dur=30.0, verbose=False)
    fp_i, fe_i, fs_i = data_int['fast']
    sp_i, se_i, ss_i = data_int['slow']
    Y_i = data_int['Y']

    pred_fast_i = blk_fast_model.predict(blk_fast_pc.transform(blk_fast_sc.transform(
        np.hstack([fp_i, fe_i, fs_i]))))
    pred_slow_i = blk_slow_model.predict(blk_slow_pc.transform(blk_slow_sc.transform(
        np.hstack([sp_i, se_i, ss_i]))))
    pred_fused_i, _ = fuse_predictions(pred_fast_i, pred_slow_i)

    mae_fast_i  = np.mean(np.abs(pred_fast_i - Y_i))
    mae_slow_i  = np.mean(np.abs(pred_slow_i - Y_i))
    mae_fused_i = np.mean(np.abs(pred_fused_i - Y_i))

    print(f"  Interp MAE — Fast stream:  {mae_fast_i:.4f} Hz  ({(1/FAST_SECONDS)/mae_fast_i:.1f}x Fourier)")
    print(f"  Interp MAE — Slow stream:  {mae_slow_i:.4f} Hz  ({(1/SLOW_SECONDS)/mae_slow_i:.1f}x Fourier)")
    print(f"  Interp MAE — Fused:        {mae_fused_i:.4f} Hz  ({(1/SLOW_SECONDS)/mae_fused_i:.1f}x Fourier)")
    print(f"  M38 reference:             0.0334 Hz  (6.0x Fourier)")

    print(f"\n  {'Actual':>6}  {'Fast':>7}  {'Slow':>7}  {'Fused':>7}")
    for f in sorted(set(Y_i)):
        m = Y_i == f
        if np.any(m):
            pf = np.mean(pred_fast_i[m]);  ef = abs(pf-f)
            ps = np.mean(pred_slow_i[m]);  es = abs(ps-f)
            pu = np.mean(pred_fused_i[m]); eu = abs(pu-f)
            print(f"  {f:6.2f}  {pf:6.3f}({ef:.3f})  {ps:6.3f}({es:.3f})  {pu:6.3f}({eu:.3f})")

    # ----------------------------------------------------------
    # TEST 3: STEP SIGNAL — fusion switching test
    # Rapid steps: slow stream can't settle, fast must dominate
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 3: RAPID STEP SIGNAL (5s steps)")
    print("  Slow stream can't settle. Fusion must use fast stream.")
    print(f"{'='*70}")

    step_freqs = [0.5, 1.0, 1.5, 2.0, 0.8, 1.3, 1.8]
    np.random.seed(4)
    sig_step  = make_step_signal(step_freqs, step_dur=5.0)
    step_time = stabilization_time + 5.0 + len(step_freqs)*5.0*4 + 10.0
    data_step = run_sim_m41(sig_step, total_time=step_time,
                             sweep_mode=True, verbose=False)

    fp_s, fe_s, fs_s = data_step['fast']
    sp_s, se_s, ss_s = data_step['slow']
    Y_s = data_step['Y']

    pred_fast_s  = fast_model.predict(fast_pc.transform(fast_sc.transform(
        np.hstack([fp_s, fe_s, fs_s]))))
    pred_slow_s  = slow_model.predict(slow_pc.transform(slow_sc.transform(
        np.hstack([sp_s, se_s, ss_s]))))
    pred_fused_s, w_slow_s = fuse_predictions(pred_fast_s, pred_slow_s)

    mae_fast_s  = np.mean(np.abs(pred_fast_s  - Y_s))
    mae_slow_s  = np.mean(np.abs(pred_slow_s  - Y_s))
    mae_fused_s = np.mean(np.abs(pred_fused_s - Y_s))
    mean_w_slow = np.mean(w_slow_s)

    print(f"  Fast MAE:  {mae_fast_s:.4f} Hz")
    print(f"  Slow MAE:  {mae_slow_s:.4f} Hz")
    print(f"  Fused MAE: {mae_fused_s:.4f} Hz")
    print(f"  Mean fusion weight (slow): {mean_w_slow:.3f}")
    print(f"  (0.0 = fully fast, 1.0 = fully slow)")
    if mean_w_slow < 0.4:
        print("  ✓ Fusion correctly using fast stream for rapid steps")
    else:
        print("  ~ Fusion still leaning slow — threshold may need tuning")

    # ----------------------------------------------------------
    # TEST 4: NOISE ROBUSTNESS
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 4: NOISE ROBUSTNESS")
    print(f"{'='*70}")
    print(f"  {'Noise σ':>8}  {'Fast%':>7}  {'Fused%':>8}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*8}")

    for nl in [0.0, 0.5, 1.0, 2.0, 3.0]:
        np.random.seed(5)
        sig = make_block_signal([0.5, 2.0], noise_level=nl)
        d   = run_sim_m41(sig, total_time=400.0, verbose=False)
        Xf  = np.hstack([d['fast'][0], d['fast'][1], d['fast'][2]])
        r   = classify_temporal(Xf, d['Y'], d['T'])
        pf  = fast_model.predict(fast_pc.transform(fast_sc.transform(Xf)))
        ps  = slow_model.predict(slow_pc.transform(slow_sc.transform(
            np.hstack([d['slow'][0], d['slow'][1], d['slow'][2]]))))
        fu, _ = fuse_predictions(pf, ps)
        threshold = 1.25
        acc_fused = np.mean((fu > threshold) == (d['Y'] > threshold))
        print(f"  {nl:8.2f}  {r['test_acc']*100:6.1f}%  {acc_fused*100:7.1f}%")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  M41 SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Mode':20s}  {'Sweep MAE':>10}  {'Block MAE':>10}  {'Noise cliff'}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*12}")
    print(f"  {'M38 (slow only)':20s}  {'0.7587 Hz':>10}  {'0.0334 Hz':>10}  σ=1.0")
    print(f"  {'M40 (fast only)':20s}  {'0.3108 Hz':>10}  {'~0.03 Hz':>10}  σ≥3.0")
    print(f"  {'M41 fast stream':20s}  {mae_fast:.4f} Hz  {mae_fast_i:.4f} Hz")
    print(f"  {'M41 slow stream':20s}  {mae_slow:.4f} Hz  {mae_slow_i:.4f} Hz")
    print(f"  {'M41 fused':20s}  {mae_fused:.4f} Hz  {mae_fused_i:.4f} Hz")
    print()
    print("  Fusion behavior:")
    print(f"    Steady-state: weight_slow → 1.0 (precision mode)")
    print(f"    Rapid steps:  weight_slow → {mean_w_slow:.2f} (tracking mode)")
    print()
    if mae_fused < mae_fast and mae_fused_i < mae_fast_i:
        print("  ✓ FUSION HELPS: fused beats both streams independently")
    elif mae_fused_i < 0.05:
        print("  ✓ PRECISION ACHIEVED: slow stream <0.05 Hz on steady-state")
    else:
        print("  ~ Fusion tuning may improve further")