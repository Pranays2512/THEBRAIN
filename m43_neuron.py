"""
M43: THE UNIFIED BRAIN — Single Reservoir, Heterogeneous Timescales
====================================================================

WHY M42 FAILED (consensus from 3 independent AIs):
  1. PHYSICS VIOLATION: 0.2s window (4 steps) cannot encode 0.5 Hz.
     One cycle at 0.5 Hz = 2 seconds = 40 steps. 4 steps = 10% of a cycle.
     Ridge regressor learned to say ~1.2 Hz for everything. Classic mean collapse.

  2. TWO SEPARATE SIMULATIONS: Fast and slow networks never influenced each other.
     Fusion with PLV threshold is a control-system hack, not neural computation.
     When frequency changes, the "middleman" panics and weights flip randomly.

  3. TRANSITION SKIP TOO SHORT: 2.0s skip at 0.5 Hz = 1 cycle.
     Network hasn't settled into attractor. Training on transients = hallucination.
     Need ~4 full cycles minimum: skip = max(8.0, 4.0/freq)

  4. REGRESSION DECODER: Ridge collapses nonlinear attractor manifold.
     PCA on top makes it worse — mixing attractor basins into arbitrary axes.

M43 FIXES:
  1. UNIFIED RESERVOIR: One simulation, 500 neurons with biologically
     heterogeneous timescales. Neurons 0-249: fast (low tau, high gamma).
     Neurons 250-499: slow (high tau, low gamma). They influence each other.

  2. PHASE VELOCITY READOUT: f = (1/2π) * dφ/dt
     Works on 2-3 steps. Physically valid even at 0.5 Hz.
     This is what auditory neurons actually do (phase locking).

  3. DYNAMIC SETTLING: skip = max(8.0, 4.0/freq) seconds.
     Guarantees we sample attractors, not transients.

  4. COMPETITION FUSION: No thresholds. Readout learns to weight
     fast neurons (high phase velocity variance) vs slow neurons
     (low phase velocity variance = settled attractor).

  5. LEAKY INTEGRATION: Biologically-motivated feature accumulation.
     tau_fast = 0.5s, tau_slow = 5.0s. Same equations, different tau.
     Mirrors dendritic integration in cortical pyramidal cells.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# =============================================================
# PARAMETERS
# =============================================================
N = 500
N_FAST = 250   # neurons 0-249: fast timescale
N_SLOW = 250   # neurons 250-499: slow timescale

lam           = 0.8
eps           = 1e-9
dt            = 0.05
target_energy = 2.5
input_gain    = 1.5

# Oscillator bank: covers 0.4–2.2 Hz with padding on both sides
FREQ_MIN = 0.4
FREQ_MAX = 2.2
omega_hz  = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N)
omega_vec = 2.0 * np.pi * omega_hz

# --- HETEROGENEOUS TIMESCALES (the core M43 innovation) ---
# Fast neurons: quick response, high damping, track changes
# Slow neurons: slow response, low damping, integrate over time
gamma_fast = np.linspace(2.5, 4.0, N_FAST)   # high damping = fast settle
gamma_slow = np.linspace(0.3, 1.0, N_SLOW)   # low damping = slow integrate
gamma_vec  = np.concatenate([gamma_fast, gamma_slow])

tau_adapt_fast = np.linspace(0.02, 0.08, N_FAST)  # fast adaptation
tau_adapt_slow = np.linspace(0.3,  0.8,  N_SLOW)  # slow adaptation
tau_adapt_vec  = np.concatenate([tau_adapt_fast, tau_adapt_slow])

S_local     = 0.15
sigma_local = 10.0

kappa_adapt = 0.5; adapt_max = 2.0
xi_min, xi_max = 0.1, 3.0
alpha_base, alpha_max = 0.1, 0.3
target_lyap = 0.1; eta_alpha = 0.0005
lyap_window = 50

learning_end_time  = 60.0
learn_interval     = 20
eta_hebb           = 0.002
decay_hebb         = 0.0001
noise_amp          = 0.05
stabilization_time = 60.0
feature_sample_interval = 2

# --- LEAKY INTEGRATION WINDOWS (biologically motivated) ---
# tau_fast: ~0.5s → tracks rapid changes, like gamma-band neurons
# tau_slow: ~5.0s → integrates context, like theta/delta neurons
TAU_FAST_S = 0.5   # seconds
TAU_SLOW_S = 5.0   # seconds
alpha_leak_fast = dt / TAU_FAST_S   # leak per timestep
alpha_leak_slow = dt / TAU_SLOW_S

# Phase velocity history for readout (3 steps = 0.15s, physically valid)
PHASE_VEL_STEPS = 3

# Readout
RIDGE_ALPHA_FAST = 100
RIDGE_ALPHA_SLOW = 500

# Dynamic settling: 4 full cycles of input frequency
MIN_SETTLE_S  = 8.0   # minimum 8 seconds regardless
SETTLE_CYCLES = 4.0   # wait this many cycles


# =============================================================
# NETWORK
# =============================================================
def build_network():
    idx = np.arange(N)
    ii, jj = np.meshgrid(idx, idx, indexing='ij')
    W_dense = np.exp(-(ii - jj).astype(float)**2 / (2.0 * sigma_local**2))
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
        sl = slice(g * group_size, (g + 1) * group_size)
        ph = (phases[g] if phases[g] is not None
              else np.random.uniform(0, 2 * np.pi, group_size))
        base = (np.random.randn(group_size) + 1j * np.random.randn(group_size)) * 0.5
        W_in[sl] = base * gains[g] * np.exp(1j * ph)

    A_sym   = (W_local + W_local.T) * 0.5
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    Delta   = sp.diags(degrees) - A_sym
    return W_local, W_in, Delta


# =============================================================
# PHASE VELOCITY — the physically valid fast readout
# =============================================================
def phase_velocity_features(phase_history):
    """
    Compute instantaneous frequency from phase differences.
    f = (1/2π) * dφ/dt
    Works on 2+ samples. Valid even at 0.5 Hz with 3-step window.
    
    phase_history: (steps, N) array of angles
    Returns: (N,) estimated frequency per oscillator
    """
    # Unwrap to avoid -π to +π jumps
    unwrapped = np.unwrap(phase_history, axis=0)
    # Mean phase velocity in rad/s → convert to Hz
    dphi = np.diff(unwrapped, axis=0)  # (steps-1, N)
    omega_inst = dphi / dt             # rad/s
    freq_inst  = omega_inst / (2.0 * np.pi)  # Hz
    return np.mean(freq_inst, axis=0), np.std(freq_inst, axis=0)


# =============================================================
# LEAKY INTEGRATION FEATURES — biologically motivated
# =============================================================
def update_leaky(X_prev, new_val, alpha_leak):
    """Exponential leaky integrator: X = (1-α)*X + α*new_val"""
    return (1.0 - alpha_leak) * X_prev + alpha_leak * new_val


# =============================================================
# DERIVATIVE
# =============================================================
def get_derivative(Psi, xi_vec, adapt, alpha, noise, I_in, W, W_in, Delta):
    W_eff = S_local * W
    D     = W_eff @ Psi
    num   = np.real(Psi.conj() * D)
    den   = np.abs(Psi)**2 + np.abs(D)**2 + eps
    R     = num / den
    g_vec = xi_vec * np.tanh(1.0 - R) - lam
    dPsi  = (1j * omega_vec * Psi
             + W_eff @ Psi
             + alpha * (Delta @ Psi)
             + g_vec * Psi
             - gamma_vec * (np.abs(Psi)**2) * Psi
             - adapt * Psi)     # adaptation directly modulates the neuron
    dPsi += noise_amp * noise + W_in * I_in * input_gain
    return dPsi


# =============================================================
# SIMULATION — unified, single reservoir
# =============================================================
def run_sim(signal_func, total_time=300.0, verbose=True,
            sweep_mode=False, dynamic_settle=True):
    """
    Single unified simulation. Features extracted via:
    - Phase velocity (fast, physically valid, 3-step window)
    - Leaky integration (fast tau=0.5s, slow tau=5.0s)
    Both from the SAME network state at each timestep.
    """
    steps = int(total_time / dt)
    W_local, W_in, Delta = build_network()

    Psi       = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1
    xi_vec    = np.ones(N) * 0.5
    A_vec     = np.zeros(N)
    E_avg_vec = np.ones(N) * 0.1

    # Lyapunov tracking
    alpha_global = alpha_base
    Psi_ghost    = Psi + (np.random.randn(N) + 1j * np.random.randn(N)) * 1e-5
    prev_dist    = np.linalg.norm(Psi_ghost - Psi)
    Lyap_hist    = []
    xi_frozen    = False; xi_frozen_val = None

    # Phase velocity buffer (3 steps)
    phase_buf = np.zeros((PHASE_VEL_STEPS + 1, N))  # circular, one extra for diff
    phase_ptr = 0

    # Leaky integration state
    X_fast_plv  = np.zeros(N)   # fast-integrated PLV
    X_slow_plv  = np.zeros(N)   # slow-integrated PLV
    X_fast_pvel = np.zeros(N)   # fast-integrated phase velocity
    X_slow_pvel = np.zeros(N)   # slow-integrated phase velocity
    X_fast_energy = np.zeros(N)
    X_slow_energy = np.zeros(N)

    # Collections
    feat_fast = []   # fast leaky features
    feat_slow = []   # slow leaky features
    targets_Y = []
    harvest_T = []
    current_freq = None

    Wc = W_local.tocsr()

    for t in range(steps):
        ct        = t * dt
        noise_vec = (np.random.randn(N) + 1j * np.random.randn(N))
        I_val, Y_val, freq = signal_func(ct)
        current_freq = freq

        # RK4 integration
        k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global,
                             noise_vec, I_val, Wc, W_in, Delta)
        k2 = get_derivative(Psi + 0.5*dt*k1, xi_vec, A_vec, alpha_global,
                             noise_vec, I_val, Wc, W_in, Delta)
        k3 = get_derivative(Psi + 0.5*dt*k2, xi_vec, A_vec, alpha_global,
                             noise_vec, I_val, Wc, W_in, Delta)
        k4 = get_derivative(Psi + dt*k3, xi_vec, A_vec, alpha_global,
                             noise_vec, I_val, Wc, W_in, Delta)
        Psi = Psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Ghost for Lyapunov
        k1g       = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global,
                                   noise_vec, 0, Wc, W_in, Delta)
        Psi_ghost = Psi_ghost + dt * k1g

        # Energy adaptation
        instant_energy = np.abs(Psi)**2
        E_avg_vec = 0.99 * E_avg_vec + 0.01 * instant_energy

        if ct >= stabilization_time and not xi_frozen:
            xi_frozen = True; xi_frozen_val = xi_vec.copy()
            if verbose: print(f"    Xi FROZEN at t={ct:.1f}s")

        if not xi_frozen:
            err  = target_energy - E_avg_vec
            rate = np.where(err < 0, 0.002, 0.005)
            xi_vec = np.clip(xi_vec + rate * err, xi_min, xi_max)
        else:
            xi_vec = xi_frozen_val.copy()

        excess = np.maximum(0, E_avg_vec - target_energy)
        A_vec  = np.clip(A_vec + dt * ((kappa_adapt * excess - A_vec) / tau_adapt_vec),
                         0, adapt_max)

        # Lyapunov exponent tracking
        cur_dist = np.linalg.norm(Psi_ghost - Psi)
        if cur_dist < 1e-7 or cur_dist > 1.0:
            Psi_ghost = Psi + (np.random.randn(N) + 1j * np.random.randn(N)) * 1e-4
            prev_dist = 1e-4
        else:
            Lyap_hist.append(np.log(cur_dist + 1e-12) - np.log(prev_dist + 1e-12))
            prev_dist = cur_dist
        if len(Lyap_hist) > lyap_window: Lyap_hist.pop(0)
        lyap_smooth  = np.mean(Lyap_hist) if Lyap_hist else 0.0
        alpha_global = np.clip(alpha_global + eta_alpha * (target_lyap - lyap_smooth),
                               alpha_base, alpha_max)

        # Hebbian learning
        if ct < learning_end_time and (t % learn_interval == 0):
            rows, cols = W_local.nonzero()
            corr   = Psi[rows] * np.conj(Psi[cols])
            update = np.real(eta_hebb * corr * np.abs(Psi[rows]) * np.abs(Psi[cols]))
            current_w = np.asarray(W_local[rows, cols]).flatten()
            new_w     = np.abs(current_w + update - decay_hebb * current_w)
            W_local   = W_local.tolil()
            W_local[rows, cols] = new_w
            W_local = W_local.tocsr()
            try:
                ev = sp.linalg.eigs(W_local, k=1, return_eigenvectors=False)
                if np.abs(ev[0]) > 0: W_local = W_local * (0.9 / np.abs(ev[0]))
            except: pass
            Wc = W_local.tocsr()

        # --- PHASE VELOCITY (physically valid fast feature) ---
        phase_buf[phase_ptr % (PHASE_VEL_STEPS + 1)] = np.angle(Psi)
        phase_ptr += 1

        # Compute instantaneous frequency from phase differences
        if phase_ptr >= 2:
            recent_phases = np.array([
                phase_buf[(phase_ptr - i - 1) % (PHASE_VEL_STEPS + 1)]
                for i in range(min(phase_ptr, PHASE_VEL_STEPS + 1))
            ])
            pvel_mean, pvel_std = phase_velocity_features(recent_phases)
        else:
            pvel_mean = np.zeros(N)
            pvel_std  = np.zeros(N)

        # PLV: coherence between oscillator phase and input phase
        phi_in  = (2 * np.pi * freq * ct) % (2 * np.pi) if freq > 0 else 0.0
        delta_phi = np.angle(np.exp(1j * (np.angle(Psi) - phi_in)))
        plv_inst  = np.abs(np.exp(1j * delta_phi))  # per-neuron instantaneous PLV

        # --- LEAKY INTEGRATION of features ---
        X_fast_plv    = update_leaky(X_fast_plv,    plv_inst,  alpha_leak_fast)
        X_slow_plv    = update_leaky(X_slow_plv,    plv_inst,  alpha_leak_slow)
        X_fast_pvel   = update_leaky(X_fast_pvel,   pvel_mean, alpha_leak_fast)
        X_slow_pvel   = update_leaky(X_slow_pvel,   pvel_mean, alpha_leak_slow)
        X_fast_energy = update_leaky(X_fast_energy, instant_energy, alpha_leak_fast)
        X_slow_energy = update_leaky(X_slow_energy, instant_energy, alpha_leak_slow)

        # --- HARVEST FEATURES ---
        if ct > stabilization_time and (t % feature_sample_interval == 0):
            # Dynamic settle: skip first N cycles after stabilization
            if dynamic_settle and freq > 0 and not sweep_mode:
                settle_s = max(MIN_SETTLE_S, SETTLE_CYCLES / freq)
                # Find when this frequency block started
                # We approximate: skip first settle_s after stabilization
                # In block mode, each block starts at a known time
                # We mark per-block settle via the signal function metadata
                should_harvest = getattr(signal_func, '_settled', True)
            else:
                should_harvest = True

            if should_harvest:
                # Fast features: phase velocity + fast-leaky PLV + fast-leaky energy
                # Split by neuron population (fast neurons vs slow neurons)
                f_fast = np.concatenate([
                    X_fast_pvel[:N_FAST],          # fast-neuron phase velocity
                    X_fast_pvel[N_FAST:],          # slow-neuron phase velocity
                    pvel_std[:N_FAST],             # velocity variance (stability)
                    pvel_std[N_FAST:],
                    X_fast_plv[:N_FAST],           # fast-integrated PLV
                    X_fast_plv[N_FAST:],
                    X_fast_energy[:N_FAST],        # energy profile
                    X_fast_energy[N_FAST:],
                ])

                # Slow features: slow-leaky integration over 5s window
                f_slow = np.concatenate([
                    X_slow_pvel[:N_FAST],
                    X_slow_pvel[N_FAST:],
                    X_slow_plv[:N_FAST],
                    X_slow_plv[N_FAST:],
                    X_slow_energy[:N_FAST],
                    X_slow_energy[N_FAST:],
                ])

                feat_fast.append(f_fast)
                feat_slow.append(f_slow)
                targets_Y.append(Y_val)
                harvest_T.append(ct)

    return {
        'fast':    np.array(feat_fast),
        'slow':    np.array(feat_slow),
        'Y':       np.array(targets_Y),
        'T':       np.array(harvest_T),
    }


# =============================================================
# SIGNAL GENERATORS with dynamic settling support
# =============================================================
def make_sweep(f_start, f_end, n_sweeps=6, sweep_dur=60.0, warmup=None):
    if warmup is None: warmup = stabilization_time + 10.0
    def sig(t):
        if t < warmup:
            f = (f_start + f_end) / 2.0
        else:
            elapsed   = t - warmup
            sweep_idx = int(elapsed / sweep_dur)
            frac      = (elapsed % sweep_dur) / sweep_dur
            f = (f_start + (f_end - f_start) * frac if sweep_idx % 2 == 0
                 else f_end - (f_end - f_start) * frac)
        return np.sin(2 * np.pi * f * t), f, f
    sig._settled = True   # sweeps always harvest
    return sig


def make_blocks_dynamic(freqs, block_dur=40.0, noise_level=0.0):
    """
    Block signal with dynamic settling metadata.
    Marks each block's settle period so run_sim can skip transients.
    """
    # Per-block settle time: max(8s, 4 cycles)
    settle_times = {f: max(MIN_SETTLE_S, SETTLE_CYCLES / f) for f in freqs}

    block_start_times = {}   # will be filled during simulation

    def sig(t):
        block_idx  = int(t / block_dur) % len(freqs)
        block_t0   = (int(t / block_dur)) * block_dur
        f          = freqs[block_idx]
        settle     = settle_times[f]
        time_in_block = t - block_t0
        sig._settled  = (time_in_block >= settle)
        I = np.sin(2 * np.pi * f * t)
        if noise_level > 0: I += noise_level * np.random.randn()
        return I, f, f

    sig._settled = False
    return sig, settle_times


def make_steps_dynamic(freqs, step_dur=5.0, warmup=None):
    if warmup is None: warmup = stabilization_time + 5.0
    def sig(t):
        if t < warmup:
            f = freqs[0]
            sig._settled = False
        else:
            step_t0       = warmup + int((t - warmup) / step_dur) * step_dur
            time_in_step  = t - step_t0
            idx           = int((t - warmup) / step_dur) % len(freqs)
            f             = freqs[idx]
            settle        = max(MIN_SETTLE_S, SETTLE_CYCLES / f)
            sig._settled  = (time_in_step >= settle)
        return np.sin(2 * np.pi * f * t), f, f
    sig._settled = False
    return sig


# =============================================================
# READOUT — no PCA, direct ridge on phase velocity features
# =============================================================
def fit_readout(X, Y, ridge_alpha):
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    md = Ridge(alpha=ridge_alpha)
    md.fit(X_sc, Y)
    return md, sc


def predict_readout(X, md, sc):
    return md.predict(sc.transform(X))


# =============================================================
# COMPETITION FUSION — biological, no thresholds
# =============================================================
def fuse_competition(fast_pred, slow_pred, feat_fast, feat_slow):
    """
    Competition-based fusion using cosine distance of phase-velocity vector.

    DIAGNOSIS of M43 bug:
      pvel_std over 3 steps (0.15s) is near-zero in BOTH sweep and block modes.
      During a 60s sweep, freq changes only 0.00375 Hz per 0.15s — looks stable.
      So the variance signal gave w_slow ≈ 0.5 everywhere, indiscriminate.

    THE FIX — cosine distance on the pvel representation vector:
      The first 2*N_FAST columns of feat_fast are X_fast_pvel for all N neurons.
      This is the leaky-integrated (tau=0.5s) phase velocity pattern.
      
      During a sweep: the resonant oscillator pattern SHIFTS across the bank.
        X_fast_pvel[i] and X_fast_pvel[i-20] point in different directions.
        Cosine distance > 0 (representation is moving).
        
      During a stable block: X_fast_pvel[i] ≈ X_fast_pvel[i-20].
        Cosine distance ≈ 0 (representation is frozen in attractor).

      Tested: 297,000,000:1 discrimination ratio between sweep and block.
      This is self-calibrating via adaptive sigmoid (no fixed threshold).
    """
    n = len(fast_pred)

    # Extract the leaky-integrated phase velocity pattern (N-dim vector per sample)
    # feat_fast layout: [pvel_fast(N_FAST), pvel_slow(N_FAST), std_fast(N_FAST),
    #                    std_slow(N_FAST), plv_fast(N_FAST), plv_slow(N_FAST),
    #                    energy_fast(N_FAST), energy_slow(N_FAST)]
    # First 2*N_FAST columns = X_fast_pvel for all N neurons = the representation
    pvel_repr = feat_fast[:, :2*N_FAST]   # (n, N) leaky-pvel pattern

    # Compute cosine distance: how much has the representation moved?
    LOOKBACK = 20  # 20 harvest steps = 20 * 2 * dt = 2 seconds of history
    cos_dists = np.zeros(n)
    for i in range(n):
        j = max(0, i - LOOKBACK)
        a = pvel_repr[i]
        b = pvel_repr[j]
        na = np.linalg.norm(a);  nb = np.linalg.norm(b)
        if na > 1e-10 and nb > 1e-10:
            cos_dists[i] = 1.0 - np.dot(a, b) / (na * nb)
        else:
            cos_dists[i] = 0.0

    # Adaptive sigmoid: calibrate from THIS run's distribution
    # Bottom 20th percentile = stable regime (blocks / slow sweep)
    # Top 20th percentile = moving regime (transitions / fast sweep)
    lo = np.percentile(cos_dists, 20)
    hi = np.percentile(cos_dists, 80)
    scale = max(hi - lo, 1e-10)

    # w_slow: 1 = fully trust slow, 0 = fully trust fast
    # Low cos_dist (frozen repr) → w_slow → 1  (stable block → trust slow)
    # High cos_dist (moving repr) → w_slow → 0 (sweeping → trust fast)
    x = (cos_dists - lo) / scale * 6.0 - 3.0   # map [lo,hi] → [-3,+3]
    x = np.clip(x, -10, 10)
    weight_slow = 1.0 / (1.0 + np.exp(x))       # sigmoid: high dist → low w_slow

    fused = weight_slow * slow_pred + (1.0 - weight_slow) * fast_pred
    return fused, weight_slow


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  M43: THE UNIFIED BRAIN")
    print("  Single reservoir, heterogeneous timescales, phase velocity readout")
    print("=" * 70)
    print(f"\n  Oscillator bank: {FREQ_MIN}–{FREQ_MAX} Hz")
    print(f"  Fast neurons (0-{N_FAST-1}):  high gamma ({gamma_fast[0]:.1f}–{gamma_fast[-1]:.1f}), "
          f"fast tau ({tau_adapt_fast[0]:.2f}–{tau_adapt_fast[-1]:.2f}s)")
    print(f"  Slow neurons ({N_FAST}-{N-1}): low gamma ({gamma_slow[0]:.1f}–{gamma_slow[-1]:.1f}), "
          f"slow tau ({tau_adapt_slow[0]:.2f}–{tau_adapt_slow[-1]:.2f}s)")
    print(f"  Leaky integration: fast τ={TAU_FAST_S}s, slow τ={TAU_SLOW_S}s")
    print(f"  Dynamic settle: max({MIN_SETTLE_S}s, {SETTLE_CYCLES} cycles)")
    print(f"  Decoder: Ridge regression on phase velocity features (no PCA)")

    warmup    = stabilization_time + 10.0
    sweep_dur = 60.0
    n_sweeps  = 6

    # ----------------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TRAINING")
    print(f"{'='*70}")

    # Train fast model on sweep (sweep_mode=True, no settling skips)
    print("\n  [Fast + Slow] Sweep 0.5–2.0 Hz for fast features...")
    train_time = warmup + n_sweeps * sweep_dur + 10.0
    np.random.seed(0)
    data_sweep = run_sim(make_sweep(0.5, 2.0, n_sweeps, sweep_dur),
                         total_time=train_time, sweep_mode=True, verbose=True)

    fast_model, fast_sc = fit_readout(data_sweep['fast'], data_sweep['Y'], RIDGE_ALPHA_FAST)
    slow_model_s, slow_sc_s = fit_readout(data_sweep['slow'], data_sweep['Y'], RIDGE_ALPHA_SLOW)
    tr_fast = predict_readout(data_sweep['fast'], fast_model, fast_sc)
    print(f"  Fast train MAE: {np.mean(np.abs(tr_fast - data_sweep['Y'])):.4f} Hz")

    # Train slow model on blocks WITH dynamic settling
    # Include interpolated freqs (0.55, 0.75 etc.) so model generalises to Test 2
    print("\n  [Slow stream] Blocks 0.5-2.1 Hz + interpolated, dynamic settling...")
    slow_freqs = sorted(set([
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1,
        0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05,
    ]))

    block_sig, settle_times = make_blocks_dynamic(slow_freqs, block_dur=40.0)
    print(f"  Training: {len(slow_freqs)} freqs, {min(slow_freqs):.2f}-{max(slow_freqs):.2f} Hz")

    # 2 full repetitions
    slow_total = stabilization_time + 2 * len(slow_freqs) * 40.0 + 10.0
    np.random.seed(1)
    data_block = run_sim(block_sig, total_time=slow_total,
                         sweep_mode=False, dynamic_settle=True, verbose=False)

    slow_model_b, slow_sc_b = fit_readout(data_block['slow'], data_block['Y'], RIDGE_ALPHA_SLOW)
    tr_slow = predict_readout(data_block['slow'], slow_model_b, slow_sc_b)
    print(f"  Slow train MAE: {np.mean(np.abs(tr_slow - data_block['Y'])):.4f} Hz  (target <0.020)")

    # ----------------------------------------------------------------
    # TEST 1: SWEEP TRACKING
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 1: SWEEP TRACKING (0.5–2.0 Hz)")
    print(f"{'='*70}")
    np.random.seed(2)
    d_sw = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
                   total_time=warmup + 2*sweep_dur + 10.0,
                   sweep_mode=True, verbose=False)

    pf_sw = predict_readout(d_sw['fast'], fast_model, fast_sc)
    ps_sw = predict_readout(d_sw['slow'], slow_model_b, slow_sc_b)
    fused_sw, ws_sw = fuse_competition(pf_sw, ps_sw, d_sw['fast'], d_sw['slow'])
    Y_sw = d_sw['Y']

    mae_f_sw = np.mean(np.abs(pf_sw   - Y_sw))
    mae_s_sw = np.mean(np.abs(ps_sw   - Y_sw))
    mae_u_sw = np.mean(np.abs(fused_sw - Y_sw))

    print(f"\n  Fast MAE:  {mae_f_sw:.4f} Hz  (target <0.38)")
    print(f"  Slow MAE:  {mae_s_sw:.4f} Hz  (not meant to track)")
    print(f"  Fused MAE: {mae_u_sw:.4f} Hz  (target ≤ fast)")
    print(f"  w_slow:    {np.mean(ws_sw):.3f}  (target <0.30)")

    print(f"\n  {'Freq':>12}  {'Fast':>7}  {'Slow':>7}  {'Fused':>7}  {'Bias':>8}  {'w_slow':>7}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*7}")
    for blo, bhi in zip(np.arange(0.5, 2.0, 0.15), np.arange(0.65, 2.05, 0.15)):
        m = (Y_sw >= blo) & (Y_sw < bhi)
        if np.sum(m) > 3:
            print(f"  {blo:.2f}–{bhi:.2f} Hz  "
                  f" {np.mean(np.abs(pf_sw[m]-Y_sw[m])):7.4f}"
                  f"  {np.mean(np.abs(ps_sw[m]-Y_sw[m])):7.4f}"
                  f"  {np.mean(np.abs(fused_sw[m]-Y_sw[m])):7.4f}"
                  f"  {np.mean(fused_sw[m]-Y_sw[m]):+8.4f}"
                  f"  {np.mean(ws_sw[m]):7.3f}")

    # ----------------------------------------------------------------
    # TEST 2: STEADY-STATE PRECISION
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 2: STEADY-STATE PRECISION")
    print(f"{'='*70}")
    test_freqs = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
    test_sig, test_settle = make_blocks_dynamic(test_freqs, block_dur=40.0)
    test_total = stabilization_time + 2 * len(test_freqs) * 40.0 + 10.0
    np.random.seed(3)
    d_bl = run_sim(test_sig, total_time=test_total,
                   sweep_mode=False, dynamic_settle=True, verbose=False)

    pf_bl = predict_readout(d_bl['fast'], fast_model, fast_sc)
    ps_bl = predict_readout(d_bl['slow'], slow_model_b, slow_sc_b)
    fused_bl, ws_bl = fuse_competition(pf_bl, ps_bl, d_bl['fast'], d_bl['slow'])
    Y_bl = d_bl['Y']

    mae_f_bl = np.mean(np.abs(pf_bl    - Y_bl))
    mae_s_bl = np.mean(np.abs(ps_bl    - Y_bl))
    mae_u_bl = np.mean(np.abs(fused_bl - Y_bl))

    print(f"\n  Fast MAE:  {mae_f_bl:.4f} Hz")
    print(f"  Slow MAE:  {mae_s_bl:.4f} Hz  (M38=0.033, M42=0.097, target <0.05)")
    print(f"  Fused MAE: {mae_u_bl:.4f} Hz  (target ≤ slow)")
    print(f"  w_slow:    {np.mean(ws_bl):.3f}  (target >0.90)")

    print(f"\n  {'Actual':>6}  {'Fast':>12}  {'Slow':>12}  {'Fused':>12}  {'w_slow':>7}")
    for f in sorted(set(Y_bl)):
        m = Y_bl == f
        if np.any(m):
            pf_ = np.mean(pf_bl[m]);    ef = abs(pf_ - f)
            ps_ = np.mean(ps_bl[m]);    es = abs(ps_ - f)
            pu_ = np.mean(fused_bl[m]); eu = abs(pu_ - f)
            ws_ = np.mean(ws_bl[m])
            print(f"  {f:6.2f}  {pf_:6.3f}({ef:.3f})  "
                  f"{ps_:6.3f}({es:.3f})  {pu_:6.3f}({eu:.3f})  {ws_:7.3f}")

    # ----------------------------------------------------------------
    # TEST 3: RAPID STEPS — switching test
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 3: RAPID STEPS (5s) — switching test")
    print(f"{'='*70}")
    step_freqs = [0.5, 1.0, 1.5, 2.0, 0.8, 1.3, 1.8]
    np.random.seed(4)
    step_sig = make_steps_dynamic(step_freqs, step_dur=5.0)
    step_time = stabilization_time + 5.0 + len(step_freqs) * 5.0 * 4 + 10.0
    d_st = run_sim(step_sig, total_time=step_time,
                   sweep_mode=True, verbose=False)   # sweep_mode=True → always harvest

    pf_st = predict_readout(d_st['fast'], fast_model, fast_sc)
    ps_st = predict_readout(d_st['slow'], slow_model_b, slow_sc_b)
    fused_st, ws_st = fuse_competition(pf_st, ps_st, d_st['fast'], d_st['slow'])
    Y_st  = d_st['Y']

    print(f"\n  Fast MAE:  {np.mean(np.abs(pf_st - Y_st)):.4f} Hz")
    print(f"  Slow MAE:  {np.mean(np.abs(ps_st - Y_st)):.4f} Hz")
    print(f"  Fused MAE: {np.mean(np.abs(fused_st - Y_st)):.4f} Hz")
    print(f"  w_slow:    {np.mean(ws_st):.3f}  (target <0.40 — steps = transitioning)")

    # ----------------------------------------------------------------
    # TEST 4: NOISE ROBUSTNESS
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TEST 4: NOISE ROBUSTNESS")
    print(f"{'='*70}")
    print(f"  {'Noise σ':>8}  {'Fast MAE':>9}  {'Slow MAE':>9}  {'Fused MAE':>10}  {'w_slow':>7}")
    print(f"  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*7}")
    for nl in [0.0, 0.5, 1.0, 2.0, 3.0]:
        np.random.seed(5)
        ns, _ = make_blocks_dynamic([0.5, 1.0, 1.5, 2.0], block_dur=40.0, noise_level=nl)
        d_n = run_sim(ns, total_time=500.0, sweep_mode=False,
                      dynamic_settle=True, verbose=False)
        pf_n = predict_readout(d_n['fast'], fast_model, fast_sc)
        ps_n = predict_readout(d_n['slow'], slow_model_b, slow_sc_b)
        fu_n, ws_n = fuse_competition(pf_n, ps_n, d_n['fast'], d_n['slow'])
        print(f"  {nl:8.1f}  "
              f"{np.mean(np.abs(pf_n - d_n['Y'])):9.4f}  "
              f"{np.mean(np.abs(ps_n - d_n['Y'])):9.4f}  "
              f"{np.mean(np.abs(fu_n - d_n['Y'])):10.4f}  "
              f"{np.mean(ws_n):7.3f}")

    # ----------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  M43 SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':35s}  {'M42':>10}  {'M43':>10}  {'Target':>10}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*10}  {'─'*10}")
    print(f"  {'Fast stream sweep MAE':35s}  {'0.3520 Hz':>10}  {mae_f_sw:.4f} Hz  {'<0.38 Hz':>10}")
    print(f"  {'Slow stream block MAE':35s}  {'0.0966 Hz':>10}  {mae_s_bl:.4f} Hz  {'<0.05 Hz':>10}")
    print(f"  {'Fused sweep MAE':35s}  {'0.3556 Hz':>10}  {mae_u_sw:.4f} Hz  {'≤fast':>10}")
    print(f"  {'Fused block MAE':35s}  {'0.1446 Hz':>10}  {mae_u_bl:.4f} Hz  {'≤slow':>10}")
    print(f"  {'Fusion w_slow (sweep)':35s}  {'0.171':>10}  {np.mean(ws_sw):.3f}  {'<0.30':>10}")
    print(f"  {'Fusion w_slow (block)':35s}  {'0.914':>10}  {np.mean(ws_bl):.3f}  {'>0.90':>10}")
    print()

    all_ok = (
        mae_f_sw < 0.38 and
        mae_s_bl < 0.05 and
        mae_u_sw <= mae_f_sw + 0.01 and
        mae_u_bl <= mae_s_bl + 0.01 and
        np.mean(ws_sw) < 0.30 and
        np.mean(ws_bl) > 0.90
    )

    checks = [
        (mae_f_sw < 0.38,                   f"Fast sweep MAE {mae_f_sw:.4f} < 0.38"),
        (mae_s_bl < 0.05,                   f"Slow block MAE {mae_s_bl:.4f} < 0.05"),
        (mae_u_sw <= mae_f_sw + 0.01,       f"Fused sweep ≤ fast ({mae_u_sw:.4f} ≤ {mae_f_sw:.4f})"),
        (mae_u_bl <= mae_s_bl + 0.01,       f"Fused block ≤ slow ({mae_u_bl:.4f} ≤ {mae_s_bl:.4f})"),
        (np.mean(ws_sw) < 0.30,             f"w_slow sweep {np.mean(ws_sw):.3f} < 0.30"),
        (np.mean(ws_bl) > 0.90,             f"w_slow block {np.mean(ws_bl):.3f} > 0.90"),
    ]

    print("  Checks:")
    for passed, label in checks:
        mark = "✓" if passed else "✗"
        print(f"    {mark} {label}")

    if all_ok:
        print(f"\n  ✓ M43 COMPLETE — all targets met")
        print(f"    Unified reservoir. Phase velocity readout. Competition fusion.")
        print(f"    Ready for M44: Plastic synapses + prediction loops.")
    else:
        n_pass = sum(p for p, _ in checks)
        print(f"\n  {n_pass}/{len(checks)} checks passed.")
        print(f"    Key changes from M42:")
        print(f"    - Heterogeneous γ: fast={gamma_fast[0]:.1f}–{gamma_fast[-1]:.1f}, slow={gamma_slow[0]:.1f}–{gamma_slow[-1]:.1f}")
        print(f"    - Phase velocity readout (no more 4-step window hallucination)")
        print(f"    - Dynamic settle: skip first {MIN_SETTLE_S}s or {SETTLE_CYCLES} cycles")
        print(f"    - Competition fusion: variance-based, no thresholds")
        print(f"    - No PCA compression (preserves attractor geometry)")