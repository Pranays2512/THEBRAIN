"""
M39 DIAGNOSTIC + FREQUENCY SWEEP
==================================
PART 1: COUPLING DIAGNOSTIC
  Tests S_global = 0 (fully decoupled) vs 0.12 (M38 current).
  If performance is identical → network is a bank of independent
  oscillators, not a network. The coupling is doing nothing.

PART 2: CONTINUOUS FREQUENCY SWEEP
  f(t) ramps linearly from 0.5 → 2.0 Hz over a long window.
  Plots predicted vs true frequency to reveal:
    - Is the manifold smooth or piecewise?
    - Where does accuracy degrade?
    - Are there dead zones (frequencies the reservoir can't encode)?

Run: python m39_diagnostic_sweep.py
Requires: m39_kuramoto.py in same directory
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Reuse all infrastructure from m39_kuramoto
from m39_kuramoto import (
    run_sim, build_network_m38, make_signal, make_regression_signal,
    classify_temporal, run_regression, energy_entropy,
    N, dt, stabilization_time, block_duration, transition_skip,
    window_seconds, window_steps, feature_sample_interval,
    ridge_alpha, eps, omega_hz
)

# =============================================================
# PART 1: COUPLING DIAGNOSTIC — Is W doing anything?
# =============================================================
print("=" * 70)
print("  COUPLING DIAGNOSTIC")
print("  S_global=0 (independent oscillators) vs S_global=0.12 (coupled)")
print("=" * 70)

print(f"\n  {'Test':32s}  {'S=0.00':>8}  {'S=0.12':>8}")
print(f"  {'─'*32}  {'─'*8}  {'─'*8}")

train_f  = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]
interp_f = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

results = {}
for S in [0.0, 0.12]:
    np.random.seed(7)
    W, W_in, Delta = build_network_m38()

    # Baseline classification
    sig = make_signal([0.5, 2.0])
    plv, ent, spec, Y, T = run_sim(sig, W, W_in, Delta, S_global=S)
    r_base = classify_temporal(np.hstack([plv, ent, spec]), Y, T)

    # Resolution floor
    np.random.seed(7)
    W, W_in, Delta = build_network_m38()
    sig = make_signal([0.5, 0.505])
    plv, ent, spec, Y, T = run_sim(sig, W, W_in, Delta, S_global=S)
    r_res = classify_temporal(np.hstack([plv, ent, spec]), Y, T)

    # Regression
    np.random.seed(7)
    W, W_in, Delta = build_network_m38()
    sig_tr = make_regression_signal(train_f)
    plv, ent, spec, Y, _ = run_sim(sig_tr, W, W_in, Delta, S_global=S, blk_dur=30.0)
    mae_tr, sc, pc, md = run_regression(plv, ent, spec, Y)

    np.random.seed(7)
    W, W_in, Delta = build_network_m38()
    sig_in = make_regression_signal(interp_f)
    plv_i, ent_i, spec_i, Y_i, _ = run_sim(sig_in, W, W_in, Delta, S_global=S, blk_dur=30.0)
    X_i = np.hstack([plv_i, ent_i, spec_i])
    mae_i = np.mean(np.abs(md.predict(pc.transform(sc.transform(X_i))) - Y_i))

    results[S] = {
        'base': r_base['test_acc'],
        'res': r_res['test_acc'],
        'mae': mae_i
    }

s0  = results[0.0]
s12 = results[0.12]
print(f"  {'Baseline 0.5 vs 2.0 Hz':32s}  {s0['base']*100:7.1f}%  {s12['base']*100:7.1f}%")
print(f"  {'Resolution 0.500 vs 0.505':32s}  {s0['res']*100:7.1f}%  {s12['res']*100:7.1f}%")
print(f"  {'Regression Interp MAE':32s}  {s0['mae']:7.4f}Hz  {s12['mae']:7.4f}Hz")
print(f"  {'Amplification':32s}  {(1/window_seconds)/s0['mae']:7.1f}x   {(1/window_seconds)/s12['mae']:7.1f}x")

print()
diff_base = abs(s0['base'] - s12['base'])
diff_res  = abs(s0['res']  - s12['res'])
diff_mae  = abs(s0['mae']  - s12['mae'])

if diff_mae < 0.003 and diff_res < 0.05:
    print("  ✗ CONFIRMED: Coupling is doing nothing.")
    print("    The network is a bank of 500 independent Hopf oscillators.")
    print("    Super-resolution comes from single-neuron gain dynamics, not coupling.")
    print("    Next step: redesign coupling to actually contribute.")
elif s12['mae'] < s0['mae'] - 0.003:
    print("  ✓ Coupling helps: MAE improves with S=0.12")
    print("    The network effect is real, just weak.")
else:
    print("  ~ Ambiguous: coupling has marginal effect.")


# =============================================================
# PART 2: CONTINUOUS FREQUENCY SWEEP
# =============================================================
print(f"\n{'='*70}")
print("  CONTINUOUS FREQUENCY SWEEP")
print("  f(t) ramps 0.5 → 2.0 Hz — tests manifold smoothness")
print(f"{'='*70}")

# --- Build the sweep signal ---
# Structure:
#   0 → stabilization_time: constant 1.0 Hz (let system settle)
#   stabilization_time → stabilization_time+50: constant 1.0 Hz (freeze xi, harvest baseline)
#   then: slow ramp 0.5 → 2.0 Hz over 300 seconds
#   then: fast ramp back 2.0 → 0.5 Hz over 150 seconds (test direction independence)

ramp_start = stabilization_time + 60.0   # give 60s after freeze before ramp
ramp_slow_dur = 300.0                      # slow ramp: 300s
ramp_fast_dur = 150.0                      # fast ramp back: 150s
f_lo, f_hi = 0.5, 2.0
total_sweep_time = ramp_start + ramp_slow_dur + ramp_fast_dur + 30.0

def sweep_signal(t):
    if t < ramp_start:
        f = 1.0  # warmup at midpoint frequency
    elif t < ramp_start + ramp_slow_dur:
        # Slow ramp up
        frac = (t - ramp_start) / ramp_slow_dur
        f = f_lo + (f_hi - f_lo) * frac
    elif t < ramp_start + ramp_slow_dur + ramp_fast_dur:
        # Fast ramp back down
        frac = (t - ramp_start - ramp_slow_dur) / ramp_fast_dur
        f = f_hi - (f_hi - f_lo) * frac
    else:
        f = f_lo
    I = np.sin(2 * np.pi * f * t)
    return I, f, f   # Y_val = true frequency for regression


print(f"\n  Sweep structure:")
print(f"    0 → {ramp_start:.0f}s:   warmup at 1.0 Hz")
print(f"    {ramp_start:.0f} → {ramp_start+ramp_slow_dur:.0f}s: SLOW ramp 0.5→2.0 Hz ({ramp_slow_dur:.0f}s)")
print(f"    {ramp_start+ramp_slow_dur:.0f} → {ramp_start+ramp_slow_dur+ramp_fast_dur:.0f}s: FAST ramp 2.0→0.5 Hz ({ramp_fast_dur:.0f}s)")
print(f"    Total: {total_sweep_time:.0f}s\n")

# --- Run sweep simulation ---
# Use a custom run that harvests ALL timesteps during the ramp
# (not block-gated) and records true vs predicted frequency

def run_sweep_sim(signal_func, total_time, S_global=0.12):
    """
    Modified sim for sweep: harvests every feature_sample_interval steps
    after stabilization, no block gating.
    """
    steps = int(total_time / dt)
    np.random.seed(9)
    W, W_in, Delta = build_network_m38()

    Psi = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1
    xi_vec = np.ones(N) * 0.5
    A_vec = np.zeros(N)
    E_avg_vec = np.ones(N) * 0.1
    alpha_global = 0.1
    Psi_ghost = Psi + (np.random.randn(N) + 1j*np.random.randn(N)) * 1e-5
    prev_dist = np.linalg.norm(Psi_ghost - Psi)
    Lyap_history = []
    xi_frozen = False; xi_frozen_val = None

    from m39_kuramoto import (lam, gamma_vec, tau_adapt_vec, kappa_adapt,
                               adapt_max, xi_min, xi_max, alpha_base, alpha_max,
                               target_lyap, eta_alpha, lyap_window,
                               learning_end_time, learn_interval, eta_hebb,
                               decay_hebb, noise_amp, input_gain, omega_vec,
                               eps, get_derivative, target_energy)

    psi_buffer = np.zeros((window_steps, N), dtype=complex)
    phi_input_buffer = np.zeros((window_steps, 1))
    buf_idx = 0; buf_filled = False

    feats_plv = []; feats_ent = []; feats_spec = []
    true_freqs = []; harvest_T = []

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

        instant_energy = np.abs(Psi)**2
        E_avg_vec = 0.99*E_avg_vec + 0.01*instant_energy

        if ct >= stabilization_time and not xi_frozen:
            xi_frozen = True; xi_frozen_val = xi_vec.copy()
            print(f"    Xi FROZEN at t={ct:.1f}s")

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

        # Harvest: after stabilization, every feature_sample_interval, no block gating
        if ct > stabilization_time and buf_filled and (t % feature_sample_interval == 0):
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
            for f_lo_b, f_hi_b in bands:
                mask = (freqs_fft >= f_lo_b) & (freqs_fft <= f_hi_b)
                spec_feats.append(np.mean(power[mask], axis=0) if np.any(mask) else np.zeros(N))
            spec = np.concatenate(spec_feats)

            feats_plv.append(plv); feats_ent.append(ent)
            feats_spec.append(spec); true_freqs.append(freq)
            harvest_T.append(ct)

    return (np.array(feats_plv), np.array(feats_ent), np.array(feats_spec),
            np.array(true_freqs), np.array(harvest_T))


print("  Step 1: Training regression model on discrete frequencies...")
np.random.seed(9)
W_tr, W_in_tr, Delta_tr = build_network_m38()
sig_tr = make_regression_signal(train_f)
plv_tr, ent_tr, spec_tr, Y_tr, _ = run_sim(
    sig_tr, W_tr, W_in_tr, Delta_tr, S_global=0.12, blk_dur=30.0)
X_tr = np.hstack([plv_tr, ent_tr, spec_tr])
scaler = StandardScaler(); X_sc = scaler.fit_transform(X_tr)
pca = PCA(n_components=50); X_p = pca.fit_transform(X_sc)
model = Ridge(alpha=ridge_alpha); model.fit(X_p, Y_tr)
print(f"  Train MAE: {np.mean(np.abs(model.predict(X_p) - Y_tr)):.4f} Hz")

print("\n  Step 2: Running continuous sweep simulation...")
plv_sw, ent_sw, spec_sw, true_f, T_sw = run_sweep_sim(sweep_signal, total_sweep_time)

print("\n  Step 3: Predicting frequency from sweep features...")
X_sw = np.hstack([plv_sw, ent_sw, spec_sw])
pred_f = model.predict(pca.transform(scaler.transform(X_sw)))

# --- Analyze results ---
# Split into slow ramp and fast ramp sections
slow_mask = (T_sw >= ramp_start) & (T_sw < ramp_start + ramp_slow_dur)
fast_mask = (T_sw >= ramp_start + ramp_slow_dur) & \
            (T_sw < ramp_start + ramp_slow_dur + ramp_fast_dur)

mae_slow = np.mean(np.abs(pred_f[slow_mask] - true_f[slow_mask])) if np.any(slow_mask) else 0
mae_fast = np.mean(np.abs(pred_f[fast_mask] - true_f[fast_mask])) if np.any(fast_mask) else 0

print(f"\n  Slow ramp (300s, 0.5→2.0 Hz): MAE = {mae_slow:.4f} Hz")
print(f"  Fast ramp (150s, 2.0→0.5 Hz): MAE = {mae_fast:.4f} Hz")

if mae_fast > mae_slow * 1.5:
    print("  ⚠ Fast ramp is worse — system has response lag (inertia)")
else:
    print("  ✓ Fast and slow ramps similar — low tracking inertia")

# --- Binned accuracy table ---
print(f"\n  Binned MAE across frequency range (slow ramp):")
print(f"  {'Freq range':>12}  {'True samples':>12}  {'MAE':>8}  {'Bias':>8}")
print(f"  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")

bins = np.arange(0.5, 2.05, 0.1)
tf_slow  = true_f[slow_mask]
pf_slow  = pred_f[slow_mask]

for i in range(len(bins)-1):
    blo, bhi = bins[i], bins[i+1]
    m = (tf_slow >= blo) & (tf_slow < bhi)
    if np.sum(m) > 3:
        bin_mae  = np.mean(np.abs(pf_slow[m] - tf_slow[m]))
        bin_bias = np.mean(pf_slow[m] - tf_slow[m])  # signed: + = overestimate
        print(f"  {blo:.1f}–{bhi:.1f} Hz      {np.sum(m):>12}  {bin_mae:8.4f}  {bin_bias:+8.4f}")

# --- Smoothness check ---
# If manifold is smooth, residuals should be uncorrelated with frequency
residuals_slow = pf_slow - tf_slow
corr = np.corrcoef(tf_slow, residuals_slow)[0,1]
print(f"\n  Residual-frequency correlation: {corr:+.3f}")
if abs(corr) < 0.15:
    print("  ✓ Smooth manifold — no systematic bias across frequency range")
elif corr > 0.15:
    print("  ⚠ Positive correlation — system underestimates high frequencies")
else:
    print("  ⚠ Negative correlation — system underestimates low frequencies")

# --- Text-based sweep plot ---
print(f"\n  Predicted vs True frequency (slow ramp, every 5th sample):")
print(f"  {'True':>6}  {'Pred':>6}  {'Err':>6}  {'Bar'}")
print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*30}")

step = max(1, len(tf_slow) // 30)
for i in range(0, len(tf_slow), step):
    err = pf_slow[i] - tf_slow[i]
    bar_len = int(abs(err) / 0.05)  # 1 char = 0.05 Hz
    bar = ('▶' if err > 0 else '◀') * min(bar_len, 20)
    print(f"  {tf_slow[i]:6.3f}  {pf_slow[i]:6.3f}  {err:+6.3f}  {bar}")

print(f"\n{'='*70}")
print("  FINAL SUMMARY")
print(f"{'='*70}")
print(f"  Coupling diagnostic:")
print(f"    S=0.00 Interp MAE: {s0['mae']:.4f} Hz  ({(1/window_seconds)/s0['mae']:.1f}x Fourier)")
print(f"    S=0.12 Interp MAE: {s12['mae']:.4f} Hz  ({(1/window_seconds)/s12['mae']:.1f}x Fourier)")
print(f"  Sweep MAE (slow): {mae_slow:.4f} Hz")
print(f"  Sweep MAE (fast): {mae_fast:.4f} Hz")
print(f"  Manifold smoothness correlation: {corr:+.3f}")