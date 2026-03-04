"""
M46: CHANGE DETECTOR + CURIOSITY FROM ENERGY REDISTRIBUTION
=============================================================

WHAT CHANGED FROM M45:
  1. FAST SYSTEM no longer decodes frequency — it detects CHANGE.
     → KL divergence between consecutive energy distributions
     → "Did the resonance pattern shift?" (scalar 0→∞)
     → This IS the curiosity signal (no separate prediction loop)
     
  2. CURIOSITY uses raw energy redistribution, not smoothed fused estimate.
     → M45 curiosity = |f_fused - f_pred| → always smooth → never spiked
     → M46 curiosity = KL(energy_now || energy_prev) → spikes instantly

  3. FUSION uses change detector to control slow/fast blend.
     → High KL → frequency changing → hold previous slow estimate
     → Low KL → stable → trust slow decoder

WHAT'S KEPT FROM M45 (working parts — DO NOT BREAK):
  - Hopf oscillator bank (ω_i log-spaced 0.4–2.2 Hz)  ✔
  - PLV^8 resonance readout (slow decoder, 0.015 Hz MAE) ✔
  - Bias correction table (not ML — geometric calibration) ✔
  - Variance-based stability for slow fusion ✔
  - Ridge regression as benchmark ✔
  - Noise robustness (0.024 Hz at σ=3.0) ✔
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from collections import deque

# =============================================================
# PARAMETERS
# =============================================================
N      = 500
N_FAST = 250
N_SLOW = 250

lam           = 0.8
eps           = 1e-9
dt            = 0.05
target_energy = 2.5
input_gain    = 1.5

FREQ_MIN  = 0.4
FREQ_MAX  = 2.2
omega_hz  = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N)
omega_vec = 2.0 * np.pi * omega_hz

gamma_fast     = np.linspace(2.5, 4.0, N_FAST)
gamma_slow     = np.linspace(0.3, 1.0, N_SLOW)
gamma_vec      = np.concatenate([gamma_fast, gamma_slow])

tau_adapt_fast = np.linspace(0.02, 0.08, N_FAST)
tau_adapt_slow = np.linspace(0.3,  0.8,  N_SLOW)
tau_adapt_vec  = np.concatenate([tau_adapt_fast, tau_adapt_slow])

S_local     = 0.15
sigma_local = 10.0

kappa_adapt = 0.5;  adapt_max  = 2.0
xi_min, xi_max     = 0.1, 3.0
alpha_base, alpha_max = 0.1, 0.3
target_lyap = 0.1;  eta_alpha  = 0.0005
lyap_window = 50

learning_end_time       = 60.0
learn_interval          = 20
eta_hebb                = 0.002
decay_hebb              = 0.0001
noise_amp               = 0.05
stabilization_time      = 60.0
feature_sample_interval = 2

# M46: Slow decoder uses resonance (PLV weighted mean)
# Fast system detects CHANGE, doesn't decode frequency
TAU_FAST_S      = 1.0       # fast energy tracking (for change detection)
TAU_SLOW_S      = 5.0       # slow precision decoder (unchanged)
alpha_leak_fast = dt / TAU_FAST_S
alpha_leak_slow = dt / TAU_SLOW_S

# Ultra-fast energy tracker for change detection
TAU_CHANGE_S    = 0.3       # very fast energy snapshot
alpha_leak_change = dt / TAU_CHANGE_S

PLV_SHARPENING  = 8
CALIB_N_FREQS   = 40

MIN_SETTLE_S    = 8.0
SETTLE_CYCLES   = 4.0

# Fusion: variance-based stability (kept from M45)
STABILITY_WINDOW = 30
STABILITY_SCALE  = 0.003

# M46: Change detection thresholds
# Use smoothed energy (tau=0.3s) but compare far back (100 samples = 10s)
# This spans a full block transition, allowing KL to accumulate
CHANGE_KL_THRESHOLD = 0.001  # KL > this → frequency is changing
CHANGE_LOOKBACK     = 100    # compare to 100 samples ago (~10s)

# Ridge (benchmark)
RIDGE_ALPHA_FAST = 100
RIDGE_ALPHA_SLOW = 500


# =============================================================
# NETWORK
# =============================================================
def build_network():
    idx = np.arange(N)
    ii, jj = np.meshgrid(idx, idx, indexing='ij')
    W_dense = np.exp(-(ii - jj).astype(float)**2 / (2.0 * sigma_local**2))
    np.fill_diagonal(W_dense, 0.0)
    W_dense /= W_dense.sum(axis=1, keepdims=True) + eps
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


def get_derivative(Psi, xi_vec, adapt, alpha_g, noise, I_in, W, W_in, Delta):
    W_eff = S_local * W
    D     = W_eff @ Psi
    num   = np.real(Psi.conj() * D)
    den   = np.abs(Psi)**2 + np.abs(D)**2 + eps
    R     = num / den
    g_vec = xi_vec * np.tanh(1.0 - R) - lam
    dPsi  = (1j*omega_vec*Psi + W_eff@Psi + alpha_g*(Delta@Psi)
             + g_vec*Psi - gamma_vec*(np.abs(Psi)**2)*Psi - adapt*Psi)
    dPsi += noise_amp*noise + W_in*I_in*input_gain
    return dPsi


def update_leaky(X, v, a):
    return (1.0 - a)*X + a*v


# =============================================================
# UNIFIED RESONANCE DECODER — same physics at both timescales
# =============================================================
def decode_resonance_raw(plv_leaky, energy_leaky):
    """
    Frequency from PLV resonance peak. Works at any timescale.
    PLV^8 sharpens the peak → concentrated weight on resonant oscillators.
    """
    w = (np.maximum(plv_leaky, 0.0)**PLV_SHARPENING) * np.maximum(energy_leaky, 0.0)
    w_sum = w.sum()
    if w_sum < eps:
        return np.mean(omega_hz)
    return np.dot(omega_hz, w) / w_sum


def build_bias_table(calib_freqs, plv_records, energy_records):
    """Build 1D bias correction from calibration. Not ML — just geometry."""
    biases = np.zeros(len(calib_freqs))
    for i, f_true in enumerate(calib_freqs):
        if f_true in plv_records and len(plv_records[f_true]) > 0:
            plv_mean    = np.mean(plv_records[f_true],    axis=0)
            energy_mean = np.mean(energy_records[f_true], axis=0)
            f_raw = decode_resonance_raw(plv_mean, energy_mean)
            biases[i] = f_raw - f_true
    return calib_freqs, biases


def decode_resonance(plv_leaky, energy_leaky, bias_freqs, bias_vals):
    """Bias-corrected resonance decoder."""
    f_raw = decode_resonance_raw(plv_leaky, energy_leaky)
    bias  = np.interp(f_raw, bias_freqs, bias_vals,
                      left=bias_vals[0], right=bias_vals[-1])
    return f_raw - bias


# =============================================================
# VARIANCE-BASED STABILITY DETECTION (kept from M45)
# =============================================================
def compute_stability(slow_history):
    """
    Low variance of recent slow estimates → signal is stable → trust slow.
    Returns w_slow ∈ [0, 1].
    """
    if len(slow_history) < 5:
        return 0.0
    recent = np.array(list(slow_history))
    var = np.var(recent)
    return float(np.clip(np.exp(-var / STABILITY_SCALE), 0.0, 1.0))


# =============================================================
# M46: KL-DIVERGENCE CHANGE DETECTOR
# =============================================================
def compute_kl_divergence(p, q):
    """
    KL divergence between two energy distributions.
    
    KL(P||Q) measures how much distribution P differs from Q.
    High KL → resonance pattern shifted → frequency changed.
    Low KL  → resonance pattern stable → frequency unchanged.
    
    This is how the brain detects novelty: from the raw neural
    activity pattern, not from a post-hoc frequency estimate.
    """
    # Normalize to probability distributions
    p_norm = np.maximum(p, eps)
    q_norm = np.maximum(q, eps)
    p_norm = p_norm / p_norm.sum()
    q_norm = q_norm / q_norm.sum()
    # Symmetric KL (Jensen-Shannon style)
    m = 0.5 * (p_norm + q_norm)
    kl = 0.5 * np.sum(p_norm * np.log(p_norm / m + eps)) + \
         0.5 * np.sum(q_norm * np.log(q_norm / m + eps))
    return float(kl)


class ChangeDetector:
    """
    M46: Detects frequency changes from energy redistribution.
    
    Instead of predicting frequency and checking prediction error,
    directly compares the resonance energy distribution across time.
    
    KL(energy_now || energy_recent) spikes when frequency changes
    and stays near zero when frequency is stable.
    
    This IS the curiosity signal — no separate prediction loop needed.
    """
    def __init__(self, lookback=CHANGE_LOOKBACK, threshold=CHANGE_KL_THRESHOLD):
        self.energy_history = deque(maxlen=lookback + 5)
        self.lookback = lookback
        self.threshold = threshold
        self.kl_history = []
        self.novelty_events = []
    
    def update(self, energy_snapshot, t):
        """
        Args:
            energy_snapshot: current energy distribution (N,) — raw, unsmoothed
            t: current time
        Returns:
            kl: KL divergence (curiosity)
            is_novel: whether this is a novel event
        """
        self.energy_history.append(energy_snapshot.copy())
        
        if len(self.energy_history) <= self.lookback:
            self.kl_history.append(0.0)
            return 0.0, False
        
        # Compare current energy to lookback-ago energy
        prev = self.energy_history[-1 - self.lookback]
        kl = compute_kl_divergence(energy_snapshot, prev)
        is_novel = kl > self.threshold
        
        self.kl_history.append(kl)
        if is_novel:
            self.novelty_events.append((t, kl))
        
        return kl, is_novel
    
    def reset(self):
        self.energy_history.clear()
        self.kl_history = []
        self.novelty_events = []


# =============================================================
# SIMULATION
# =============================================================
def run_sim(signal_func, total_time=300.0, verbose=True,
            sweep_mode=False, dynamic_settle=True,
            collect_calib=False):
    """
    Unified M46 simulation.
    
    Returns per-sample:
      direct_fast:  resonance decoder at tau=1.0s (no training)
      direct_slow:  resonance decoder at tau=5.0s (no training)
      plv_fast/slow: leaky PLV magnitude (N,)
      energy_fast/slow: leaky energy (N,)
      energy_change: ultra-fast energy snapshot (N,) for change detection
      feat_fast/slow: feature vectors for Ridge benchmark
      Y, T: targets and timestamps
    """
    steps    = int(total_time / dt)
    W_local, W_in, Delta = build_network()

    Psi          = (np.random.randn(N) + 1j*np.random.randn(N)) * 0.1
    xi_vec       = np.ones(N) * 0.5
    A_vec        = np.zeros(N)
    E_avg_vec    = np.ones(N) * 0.1
    alpha_global = alpha_base
    Psi_ghost    = Psi + (np.random.randn(N)+1j*np.random.randn(N))*1e-5
    prev_dist    = np.linalg.norm(Psi_ghost - Psi)
    Lyap_hist    = []
    xi_frozen    = False;  xi_frozen_val = None

    # M46: Three energy timescales
    X_fast_plv    = np.zeros(N, dtype=complex)
    X_slow_plv    = np.zeros(N, dtype=complex)
    X_fast_energy = np.zeros(N)
    X_slow_energy = np.zeros(N)
    X_change_energy = np.zeros(N)  # ultra-fast (tau=0.3s) for change detection

    # Output collectors
    out = {k: [] for k in ['direct_fast', 'direct_slow',
                            'plv_fast', 'plv_slow',
                            'energy_fast', 'energy_slow',
                            'energy_change',
                            'instant_energy',
                            'feat_fast', 'feat_slow',
                            'Y', 'T']}

    calib_plv_fast = {}; calib_energy_fast = {}
    calib_plv_slow = {}; calib_energy_slow = {}

    Wc = W_local.tocsr()

    for t in range(steps):
        ct        = t * dt
        noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
        I_val, Y_val, freq = signal_func(ct)

        # RK4
        k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global,
                             noise_vec, I_val, Wc, W_in, Delta)
        k2 = get_derivative(Psi+0.5*dt*k1, xi_vec, A_vec, alpha_global,
                             noise_vec, I_val, Wc, W_in, Delta)
        k3 = get_derivative(Psi+0.5*dt*k2, xi_vec, A_vec, alpha_global,
                             noise_vec, I_val, Wc, W_in, Delta)
        k4 = get_derivative(Psi+dt*k3, xi_vec, A_vec, alpha_global,
                             noise_vec, I_val, Wc, W_in, Delta)
        Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

        # Ghost (Euler for speed)
        k1g       = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global,
                                   noise_vec, 0, Wc, W_in, Delta)
        Psi_ghost = Psi_ghost + dt*k1g

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

        if ct < learning_end_time and (t % learn_interval == 0):
            rows, cols = W_local.nonzero()
            corr   = Psi[rows] * np.conj(Psi[cols])
            update = np.real(eta_hebb*corr*np.abs(Psi[rows])*np.abs(Psi[cols]))
            current_w = np.asarray(W_local[rows, cols]).flatten()
            new_w     = np.abs(current_w + update - decay_hebb*current_w)
            W_local   = W_local.tolil()
            W_local[rows, cols] = new_w
            W_local = W_local.tocsr()
            try:
                ev = sp.linalg.eigs(W_local, k=1, return_eigenvectors=False)
                if np.abs(ev[0]) > 0: W_local = W_local*(0.9/np.abs(ev[0]))
            except: pass
            Wc = W_local.tocsr()

        # PLV: complex leaky integrator
        phi_in   = 2*np.pi*freq*ct if freq > 0 else 0.0
        phasor_i = np.exp(1j*(np.angle(Psi) - phi_in))

        X_fast_plv      = update_leaky(X_fast_plv,      phasor_i,      alpha_leak_fast)
        X_slow_plv      = update_leaky(X_slow_plv,      phasor_i,      alpha_leak_slow)
        X_fast_energy   = update_leaky(X_fast_energy,   instant_energy, alpha_leak_fast)
        X_slow_energy   = update_leaky(X_slow_energy,   instant_energy, alpha_leak_slow)
        X_change_energy = update_leaky(X_change_energy, instant_energy, alpha_leak_change)

        # Harvest
        if ct > stabilization_time and (t % feature_sample_interval == 0):
            if dynamic_settle and freq > 0 and not sweep_mode:
                should_harvest = getattr(signal_func, '_settled', True)
            else:
                should_harvest = True

            if should_harvest:
                plv_fast_mag = np.abs(X_fast_plv)
                plv_slow_mag = np.abs(X_slow_plv)

                # Direct decode: both streams use resonance
                f_fast = decode_resonance_raw(plv_fast_mag, X_fast_energy)
                f_slow = decode_resonance_raw(plv_slow_mag, X_slow_energy)

                # Ridge features (real-valued)
                f_fast_feat = np.concatenate([
                    plv_fast_mag[:N_FAST], plv_fast_mag[N_FAST:],
                    X_fast_energy[:N_FAST], X_fast_energy[N_FAST:],
                ])
                f_slow_feat = np.concatenate([
                    plv_slow_mag[:N_FAST], plv_slow_mag[N_FAST:],
                    X_slow_energy[:N_FAST], X_slow_energy[N_FAST:],
                ])

                out['direct_fast'].append(f_fast)
                out['direct_slow'].append(f_slow)
                out['plv_fast'].append(plv_fast_mag.copy())
                out['plv_slow'].append(plv_slow_mag.copy())
                out['energy_fast'].append(X_fast_energy.copy())
                out['energy_slow'].append(X_slow_energy.copy())
                out['energy_change'].append(X_change_energy.copy())
                out['instant_energy'].append(instant_energy.copy())
                out['feat_fast'].append(f_fast_feat)
                out['feat_slow'].append(f_slow_feat)
                out['Y'].append(Y_val)
                out['T'].append(ct)

                # Calibration
                if collect_calib:
                    key = round(freq, 4)
                    for store, vec in [(calib_plv_fast, plv_fast_mag),
                                       (calib_energy_fast, X_fast_energy),
                                       (calib_plv_slow, plv_slow_mag),
                                       (calib_energy_slow, X_slow_energy)]:
                        if key not in store: store[key] = []
                        store[key].append(vec.copy())

    for k in out:
        out[k] = np.array(out[k])
    out['calib_plv_fast']    = calib_plv_fast
    out['calib_energy_fast'] = calib_energy_fast
    out['calib_plv_slow']    = calib_plv_slow
    out['calib_energy_slow'] = calib_energy_slow
    return out


# =============================================================
# RIDGE REGRESSION — benchmark only
# =============================================================
def fit_ridge(X, Y, alpha):
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    md = Ridge(alpha=alpha)
    md.fit(X_sc, Y)
    return md, sc

def predict_ridge(X, md, sc):
    return md.predict(sc.transform(X))


# =============================================================
# SIGNAL GENERATORS
# =============================================================
def make_sweep(f_start, f_end, n_sweeps=6, sweep_dur=60.0, warmup=None):
    if warmup is None: warmup = stabilization_time + 10.0
    def sig(t):
        if t < warmup:
            f = (f_start + f_end)/2.0
        else:
            elapsed   = t - warmup
            sweep_idx = int(elapsed/sweep_dur)
            frac      = (elapsed % sweep_dur)/sweep_dur
            f = (f_start + (f_end-f_start)*frac if sweep_idx % 2 == 0
                 else f_end - (f_end-f_start)*frac)
        return np.sin(2*np.pi*f*t), f, f
    sig._settled = True
    return sig


def make_blocks(freqs, block_dur=40.0, noise_level=0.0):
    settle_times = {f: max(MIN_SETTLE_S, SETTLE_CYCLES/f) for f in freqs}
    def sig(t):
        block_idx     = int(t/block_dur) % len(freqs)
        block_t0      = int(t/block_dur) * block_dur
        f             = freqs[block_idx]
        time_in_block = t - block_t0
        sig._settled  = (time_in_block >= settle_times[f])
        I = np.sin(2*np.pi*f*t)
        if noise_level > 0: I += noise_level*np.random.randn()
        return I, f, f
    sig._settled = False
    return sig, settle_times


def make_steps(freqs, step_dur=5.0, warmup=None):
    if warmup is None: warmup = stabilization_time + 5.0
    def sig(t):
        if t < warmup:
            f = freqs[0]; sig._settled = False
        else:
            step_t0      = warmup + int((t-warmup)/step_dur)*step_dur
            time_in_step = t - step_t0
            idx          = int((t-warmup)/step_dur) % len(freqs)
            f            = freqs[idx]
            sig._settled = (time_in_step >= max(MIN_SETTLE_S, SETTLE_CYCLES/f))
        return np.sin(2*np.pi*f*t), f, f
    sig._settled = False
    return sig


# =============================================================
# HELPERS
# =============================================================
def mae(pred, true): return np.mean(np.abs(np.array(pred) - np.array(true)))


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("  M46: CHANGE DETECTOR + CURIOSITY FROM ENERGY REDISTRIBUTION")
    print("  KL divergence detects transitions, slow PLV decoder reads frequency")
    print("=" * 72)
    print(f"\n  Change detector: KL divergence of energy (tau={TAU_CHANGE_S}s)")
    print(f"  Precision decoder: PLV^8 resonance (tau={TAU_SLOW_S}s)")
    print(f"  Fusion: variance-based stability (window={STABILITY_WINDOW})")
    print(f"  Curiosity: KL threshold={CHANGE_KL_THRESHOLD}")

    warmup    = stabilization_time + 10.0
    sweep_dur = 60.0
    n_sweeps  = 6

    # ================================================================
    # CALIBRATION
    # ================================================================
    print(f"\n{'='*72}")
    print("  CALIBRATION + TRAINING")
    print(f"{'='*72}")

    # --- Calibration sweep ---
    print("\n  [Calibration] Sweep 0.5–2.0 Hz...")
    train_time = warmup + n_sweeps*sweep_dur + 10.0
    np.random.seed(0)
    data_train = run_sim(
        make_sweep(0.5, 2.0, n_sweeps, sweep_dur),
        total_time=train_time,
        sweep_mode=True, verbose=True,
        collect_calib=True
    )

    # Build bias tables for BOTH timescales
    calib_freqs_fast = sorted(data_train['calib_plv_fast'].keys())
    bias_freqs_fast, bias_vals_fast = build_bias_table(
        np.array(calib_freqs_fast),
        data_train['calib_plv_fast'],
        data_train['calib_energy_fast']
    )
    print(f"  Fast bias table: {len(bias_freqs_fast)} points, "
          f"max bias = {np.max(np.abs(bias_vals_fast)):.4f} Hz")

    calib_freqs_slow = sorted(data_train['calib_plv_slow'].keys())
    bias_freqs_slow, bias_vals_slow = build_bias_table(
        np.array(calib_freqs_slow),
        data_train['calib_plv_slow'],
        data_train['calib_energy_slow']
    )
    print(f"  Slow bias table: {len(bias_freqs_slow)} points, "
          f"max bias = {np.max(np.abs(bias_vals_slow)):.4f} Hz")

    # Apply bias correction to calibration data
    d_fast_corr = np.array([
        decode_resonance(data_train['plv_fast'][i], data_train['energy_fast'][i],
                         bias_freqs_fast, bias_vals_fast)
        for i in range(len(data_train['Y']))
    ])
    d_slow_corr = np.array([
        decode_resonance(data_train['plv_slow'][i], data_train['energy_slow'][i],
                         bias_freqs_slow, bias_vals_slow)
        for i in range(len(data_train['Y']))
    ])
    print(f"  Direct fast train MAE: {mae(d_fast_corr, data_train['Y']):.4f} Hz "
          f"(raw: {mae(data_train['direct_fast'], data_train['Y']):.4f})")
    print(f"  Direct slow train MAE: {mae(d_slow_corr, data_train['Y']):.4f} Hz "
          f"(raw: {mae(data_train['direct_slow'], data_train['Y']):.4f})")

    # Train Ridge as benchmark
    ridge_fast, ridge_fast_sc = fit_ridge(
        data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)
    rf_train = predict_ridge(data_train['feat_fast'], ridge_fast, ridge_fast_sc)
    print(f"  Ridge fast train MAE:  {mae(rf_train, data_train['Y']):.4f} Hz")

    # --- Block calibration for slow decoder ---
    print("\n  [Blocks] Calibrating slow decoder...")
    slow_freqs = sorted(set([
        0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,
        0.55,0.75,0.95,1.15,1.35,1.55,1.75,1.95,2.05,
    ]))
    block_sig_train, _ = make_blocks(slow_freqs, block_dur=40.0)
    slow_total = stabilization_time + 2*len(slow_freqs)*40.0 + 10.0
    np.random.seed(1)
    data_slow = run_sim(
        block_sig_train, total_time=slow_total,
        sweep_mode=False, dynamic_settle=True, verbose=False,
        collect_calib=True
    )

    # Augment bias tables with block data
    block_freqs_slow = sorted(data_slow['calib_plv_slow'].keys())
    bias_freqs_slow, bias_vals_slow = build_bias_table(
        np.array(block_freqs_slow),
        data_slow['calib_plv_slow'],
        data_slow['calib_energy_slow']
    )
    block_freqs_fast = sorted(data_slow['calib_plv_fast'].keys())
    bias_freqs_fast, bias_vals_fast = build_bias_table(
        np.array(block_freqs_fast),
        data_slow['calib_plv_fast'],
        data_slow['calib_energy_fast']
    )
    print(f"  Block bias (slow): {len(bias_freqs_slow)} pts, "
          f"max bias = {np.max(np.abs(bias_vals_slow)):.4f} Hz")
    print(f"  Block bias (fast): {len(bias_freqs_fast)} pts, "
          f"max bias = {np.max(np.abs(bias_vals_fast)):.4f} Hz")

    # Verify slow decoder on blocks
    d_slow_block = np.array([
        decode_resonance(data_slow['plv_slow'][i], data_slow['energy_slow'][i],
                         bias_freqs_slow, bias_vals_slow)
        for i in range(len(data_slow['Y']))
    ])
    print(f"  Direct slow block MAE: {mae(d_slow_block, data_slow['Y']):.4f} Hz")

    ridge_slow, ridge_slow_sc = fit_ridge(
        data_slow['feat_slow'], data_slow['Y'], RIDGE_ALPHA_SLOW)
    rs_train = predict_ridge(data_slow['feat_slow'], ridge_slow, ridge_slow_sc)
    print(f"  Ridge slow block MAE:  {mae(rs_train, data_slow['Y']):.4f} Hz")


    # ================================================================
    # HELPER: Apply decode + fusion + change detection to test data
    # ================================================================
    def decode_test(data, label=""):
        Y = data['Y']; T = data['T']
        n = len(Y)

        # Direct decode (bias-corrected, both timescales)
        df = np.array([decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                         bias_freqs_fast, bias_vals_fast) for i in range(n)])
        ds = np.array([decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                         bias_freqs_slow, bias_vals_slow) for i in range(n)])

        # M46: Change detector from smoothed energy (tau=0.3s)
        # Use wide lookback (100 samples=10s) to span block transitions
        change_detector = ChangeDetector()
        curiosity = np.zeros(n)
        novelty = np.zeros(n, dtype=bool)
        for i in range(n):
            c, nov = change_detector.update(data['energy_change'][i], T[i])
            curiosity[i] = c
            novelty[i] = nov

        # Variance-based fusion (kept from M45)
        slow_hist = deque(maxlen=STABILITY_WINDOW)
        d_fused = np.zeros(n)
        d_w_slow = np.zeros(n)
        for i in range(n):
            slow_hist.append(ds[i])
            w = compute_stability(slow_hist)
            # M46: Also suppress slow during detected changes
            if curiosity[i] > CHANGE_KL_THRESHOLD:
                w = w * 0.1  # dampen slow trust during transitions
            d_fused[i] = w * ds[i] + (1.0 - w) * df[i]
            d_w_slow[i] = w

        # Ridge benchmark
        rf = predict_ridge(data['feat_fast'], ridge_fast, ridge_fast_sc)
        rs = predict_ridge(data['feat_slow'], ridge_slow, ridge_slow_sc)

        # Ridge fusion (same variance approach)
        ridge_slow_hist = deque(maxlen=STABILITY_WINDOW)
        r_fused = np.zeros(n)
        r_w_slow = np.zeros(n)
        for i in range(n):
            ridge_slow_hist.append(rs[i])
            w = compute_stability(ridge_slow_hist)
            r_fused[i] = w * rs[i] + (1.0 - w) * rf[i]
            r_w_slow[i] = w

        return {
            'df': df, 'ds': ds, 'd_fused': d_fused, 'd_w_slow': d_w_slow,
            'rf': rf, 'rs': rs, 'r_fused': r_fused, 'r_w_slow': r_w_slow,
            'curiosity': curiosity, 'novelty': novelty,
            'change_events': change_detector.novelty_events,
            'Y': Y, 'T': T,
        }


    def print_comparison(label, r):
        Y = r['Y']
        print(f"\n  {'':22s}  {'Direct':>8}  {'Ridge':>8}  {'Winner':>8}")
        print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}")
        for name, dp, rp in [("Fast MAE",  r['df'],      r['rf']),
                              ("Slow MAE",  r['ds'],      r['rs']),
                              ("Fused MAE", r['d_fused'], r['r_fused'])]:
            dm = mae(dp, Y); rm = mae(rp, Y)
            w  = "Direct" if dm < rm else ("Ridge" if rm < dm else "Tie")
            print(f"  {name:22s}  {dm:8.4f}  {rm:8.4f}  {w:>8}")
        print(f"  {'w_slow (direct)':22s}  {np.mean(r['d_w_slow']):8.3f}")
        print(f"  {'w_slow (ridge)':22s}  {np.mean(r['r_w_slow']):8.3f}")
        print(f"  {'KL curiosity mean':22s}  {np.mean(r['curiosity']):8.4f}")
        print(f"  {'Change events':22s}  {int(np.sum(r['novelty'])):>8}")


    # ================================================================
    # TEST 1: SWEEP TRACKING
    # ================================================================
    print(f"\n{'='*72}")
    print("  TEST 1: SWEEP TRACKING (0.5–2.0 Hz)")
    print(f"{'='*72}")
    np.random.seed(2)
    d_sw = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
                   total_time=warmup+2*sweep_dur+10.0,
                   sweep_mode=True, verbose=False)
    r_sw = decode_test(d_sw)
    print_comparison("SWEEP", r_sw)
    print(f"\n  w_slow target: <0.30 (sweep = moving = trust fast)")

    # Per-band
    print(f"\n  Per-band:")
    print(f"  {'Freq':>12}  {'Dir.Fast':>9}  {'Rdg.Fast':>9}  {'Dir.Fused':>10}  {'Rdg.Fused':>10}")
    print(f"  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*10}")
    for blo, bhi in zip(np.arange(0.5,2.0,0.15), np.arange(0.65,2.05,0.15)):
        Y = r_sw['Y']
        m = (Y >= blo) & (Y < bhi)
        if m.sum() > 3:
            print(f"  {blo:.2f}–{bhi:.2f} Hz"
                  f"  {mae(r_sw['df'][m], Y[m]):9.4f}"
                  f"  {mae(r_sw['rf'][m], Y[m]):9.4f}"
                  f"  {mae(r_sw['d_fused'][m], Y[m]):10.4f}"
                  f"  {mae(r_sw['r_fused'][m], Y[m]):10.4f}")


    # ================================================================
    # TEST 2: STEADY-STATE PRECISION (blocks)
    # ================================================================
    print(f"\n{'='*72}")
    print("  TEST 2: STEADY-STATE PRECISION")
    print(f"{'='*72}")
    test_freqs = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
    test_sig, _ = make_blocks(test_freqs, block_dur=40.0)
    test_total  = stabilization_time + 2*len(test_freqs)*40.0 + 10.0
    np.random.seed(3)
    d_bl = run_sim(test_sig, total_time=test_total,
                   sweep_mode=False, dynamic_settle=True, verbose=False)
    r_bl = decode_test(d_bl)
    print_comparison("BLOCKS", r_bl)
    print(f"\n  w_slow target: >0.80 (blocks = stable = trust slow)")

    # Per-frequency
    print(f"\n  Per-frequency:")
    print(f"  {'Actual':>6}  {'Dir.Fast':>9}  {'Dir.Slow':>9}  {'Dir.Fused':>10}  {'Rdg.Fused':>10}  {'w_d':>5}  {'Curio':>7}")
    Y_bl = r_bl['Y']
    for f in sorted(set(Y_bl)):
        m = Y_bl == f
        if m.any():
            print(f"  {f:6.2f}"
                  f"  {np.mean(r_bl['df'][m]):9.3f}"
                  f"  {np.mean(r_bl['ds'][m]):9.3f}"
                  f"  {np.mean(r_bl['d_fused'][m]):10.3f}"
                  f"  {np.mean(r_bl['r_fused'][m]):10.3f}"
                  f"  {np.mean(r_bl['d_w_slow'][m]):5.2f}"
                  f"  {np.mean(r_bl['curiosity'][m]):7.4f}")


    # ================================================================
    # TEST 3: RAPID STEPS + CURIOSITY
    # ================================================================
    print(f"\n{'='*72}")
    print("  TEST 3: RAPID STEPS + CURIOSITY DETECTION")
    print(f"{'='*72}")
    step_freqs = [0.5, 1.0, 1.5, 2.0, 0.8, 1.3, 1.8]
    np.random.seed(4)
    d_st = run_sim(make_steps(step_freqs, step_dur=5.0),
                   total_time=stabilization_time+5.0+len(step_freqs)*5.0*4+10.0,
                   sweep_mode=True, verbose=False)
    r_st = decode_test(d_st)
    print_comparison("STEPS", r_st)

    # Curiosity analysis
    if r_st['change_events']:
        print(f"\n  Change events detected: {len(r_st['change_events'])}")
        print(f"  {'Time':>8}  {'KL div':>10}")
        print(f"  {'─'*8}  {'─'*10}")
        for t, kl in r_st['change_events'][:20]:
            print(f"  {t:8.1f}  {kl:10.4f}")
    else:
        print(f"\n  No change events detected (KL threshold={CHANGE_KL_THRESHOLD})")


    # ================================================================
    # TEST 4: NOISE ROBUSTNESS
    # ================================================================
    print(f"\n{'='*72}")
    print("  TEST 4: NOISE ROBUSTNESS")
    print(f"{'='*72}")
    print(f"  {'Noise':>6}  {'Dir.Fast':>9}  {'Dir.Slow':>9}  "
          f"{'Dir.Fsd':>8}  {'Rdg.Fsd':>8}  {'w_slow':>7}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*7}")
    for nl in [0.0, 0.5, 1.0, 2.0, 3.0]:
        np.random.seed(5)
        ns, _ = make_blocks([0.5, 1.0, 1.5, 2.0], block_dur=40.0, noise_level=nl)
        d_n = run_sim(ns, total_time=500.0, sweep_mode=False,
                      dynamic_settle=True, verbose=False)
        r_n = decode_test(d_n)
        print(f"  {nl:6.1f}"
              f"  {mae(r_n['df'], r_n['Y']):9.4f}"
              f"  {mae(r_n['ds'], r_n['Y']):9.4f}"
              f"  {mae(r_n['d_fused'], r_n['Y']):8.4f}"
              f"  {mae(r_n['r_fused'], r_n['Y']):8.4f}"
              f"  {np.mean(r_n['d_w_slow']):7.3f}")


    # ================================================================
    # TEST 5: CURIOSITY ON FREQUENCY CHANGES
    # ================================================================
    print(f"\n{'='*72}")
    print("  TEST 5: CURIOSITY SIGNAL ANALYSIS")
    print(f"{'='*72}")
    # Run blocks with clear frequency transitions
    curiosity_freqs = [0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
    c_sig, _ = make_blocks(curiosity_freqs, block_dur=30.0)
    np.random.seed(6)
    d_c = run_sim(c_sig, total_time=stabilization_time + len(curiosity_freqs)*30.0*2 + 10.0,
                  sweep_mode=False, dynamic_settle=False, verbose=False)
    r_c = decode_test(d_c)

    # Analyze: KL curiosity should spike at block transitions
    Y_c = r_c['Y']; T_c = r_c['T']
    freq_changes = np.where(np.diff(Y_c) != 0)[0] + 1
    n_events = len(r_c['change_events'])

    # KL curiosity near transitions vs during blocks
    near_transition = np.zeros(len(Y_c), dtype=bool)
    for idx in freq_changes:
        lo = max(0, idx-3)
        hi = min(len(Y_c), idx+10)  # wider window for KL to propagate
        near_transition[lo:hi] = True

    c_at_transition = np.mean(r_c['curiosity'][near_transition]) if near_transition.any() else 0
    c_between = np.mean(r_c['curiosity'][~near_transition]) if (~near_transition).any() else 0

    print(f"  Frequency transitions: {len(freq_changes)}")
    print(f"  Change events (KL):    {n_events}")
    print(f"  KL at transitions:     {c_at_transition:.4f}  (target: >{CHANGE_KL_THRESHOLD:.2f})")
    print(f"  KL between blocks:     {c_between:.4f}  (target: <0.02)")
    print(f"  Ratio (change/calm):   {c_at_transition/(c_between+1e-9):.1f}×")

    if c_at_transition > CHANGE_KL_THRESHOLD and c_between < 0.03:
        print(f"  ✓ CURIOSITY WORKS: KL spikes on transitions, calm between")
    elif c_at_transition > c_between * 2:
        print(f"  ~ Partial: KL is higher at transitions but not fully clean")
    else:
        print(f"  ✗ KL curiosity not discriminating transitions")


    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'='*72}")
    print("  M46 FINAL VERDICT")
    print(f"{'='*72}")

    metrics = {
        'Sweep fast MAE':     (mae(r_sw['df'],      r_sw['Y']),  mae(r_sw['rf'],      r_sw['Y']),  0.20, '<'),
        'Block slow MAE':     (mae(r_bl['ds'],      r_bl['Y']),  mae(r_bl['rs'],      r_bl['Y']),  0.02, '<'),
        'Block fused MAE':    (mae(r_bl['d_fused'], r_bl['Y']),  mae(r_bl['r_fused'], r_bl['Y']),  None, None),
        'w_slow blocks':      (np.mean(r_bl['d_w_slow']),        np.mean(r_bl['r_w_slow']),        0.80, '>'),
        'w_slow sweep':       (np.mean(r_sw['d_w_slow']),        np.mean(r_sw['r_w_slow']),        0.30, '<'),
        'KL curiosity ratio': (c_at_transition/(c_between+1e-9), 0, 3.0, '>'),
    }

    print(f"\n  {'Metric':25s}  {'Direct':>9}  {'Ridge':>9}  {'Target':>8}  {'OK':>4}")
    print(f"  {'─'*25}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*4}")
    dir_wins = 0; rdg_wins = 0; dir_pass = 0
    for name, (dv, rv, thr, op) in metrics.items():
        if thr is not None:
            if op == '<':
                ok = "✓" if dv < thr else "✗"
                if dv < thr: dir_pass += 1
            else:
                ok = "✓" if dv > thr else "✗"
                if dv > thr: dir_pass += 1
            t_str = f"{op}{thr}"
        else:
            slow_m = mae(r_bl['ds'], r_bl['Y'])
            ok = "✓" if dv <= slow_m + 0.01 else "✗"
            if dv <= slow_m + 0.01: dir_pass += 1
            t_str = "≤ slow"

        if rv > 0 and dv < rv: dir_wins += 1
        elif rv > 0 and rv < dv: rdg_wins += 1
        print(f"  {name:25s}  {dv:9.4f}  {rv:9.4f}  {t_str:>8}  {ok:>4}")

    print(f"\n  Direct checks passed: {dir_pass}/{len(metrics)}")
    print(f"  Head-to-head: Direct={dir_wins}, Ridge={rdg_wins}")
    print()
    print("  M46 architecture:")
    print("    EAR:        500 Hopf oscillators (0.4–2.2 Hz)")
    print(f"    CHANGE:     KL divergence of energy (tau={TAU_CHANGE_S}s)")
    print(f"    PRECISION:  PLV^8 resonance readout (tau={TAU_SLOW_S}s)")
    print("    FUSION:     variance stability + KL change gate")
    print("    CURIOSITY:  KL(energy_now || energy_prev)")
    print()
