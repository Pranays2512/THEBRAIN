"""
M38: TONOTOPIC RESONANT ENCODER
================================
Changes from M37-FIXED:
  1. TONOTOPIC INPUT PROJECTION
     W_in is structured, not uniform. 5 neuron groups receive
     different gains and phases — cochlea-style spatial filter.
  2. ENTROPY REPLACES VARIANCE
     var(|Psi|^2) collapses under any noise (second-order statistic).
     Replaced with normalized energy entropy — much more robust.
  3. S_global: 0.08 → 0.12
     Slightly stronger nonlinear mixing. Still below sync threshold.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time as clock

# =============================================================
# M38 PHYSICS
# =============================================================
N = 500
lam = 0.8
eps = 1e-6
dt = 0.05
target_energy = 2.5
input_gain = 1.5

omega_hz = np.logspace(np.log10(0.3), np.log10(3.0), N)
omega_vec = 2.0 * np.pi * omega_hz

# M38: Slightly stronger coupling for sharper manifold curvature
S_global = 0.12

gamma_vec = np.linspace(0.1, 2.0, N)
tau_adapt_vec = np.linspace(0.2, 5.0, N)
kappa_adapt = 0.5; adapt_max = 2.0
eta_xi = 0.002; xi_min, xi_max = 0.1, 3.0
alpha_base, alpha_max = 0.1, 0.3; target_lyap = 0.1; eta_alpha = 0.0005
lyap_window = 100
learning_end_time = 100.0; learn_interval = 20; eta_hebb = 0.002; decay_hebb = 0.0001
noise_amp = 0.05
stabilization_time = 120.0; energy_gate = 0.5; ridge_alpha = 1000.0; density = 0.02
block_duration = 50.0; transition_skip = 10.0
window_seconds = 5.0; window_steps = int(window_seconds / dt); feature_sample_interval = 10


def build_network():
    W_real = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W_imag = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W = (W_real + 1j * W_imag)
    try:
        eigenvals = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
        if np.abs(eigenvals[0]) > 0: W = W * (0.9 / np.abs(eigenvals[0]))
    except: pass

    # M38 FIX 1: TONOTOPIC INPUT PROJECTION
    # 5 groups of 100 neurons, each with different gain + phase
    # Mimics cochlear preprocessing — coarse frequency filter before reservoir
    np.random.seed(42)
    W_in = np.zeros(N, dtype=complex)
    group_size = N // 5
    gains  = [2.0, 1.2, 0.5, 1.2, 0.8]
    phases = [0.0, 0.0, 0.0, np.pi, None]  # None = random phase
    for g in range(5):
        sl = slice(g * group_size, (g + 1) * group_size)
        ph = phases[g] if phases[g] is not None else np.random.uniform(0, 2*np.pi, group_size)
        base = (np.random.randn(group_size) + 1j * np.random.randn(group_size)) * 0.5
        W_in[sl] = base * gains[g] * np.exp(1j * ph)

    A_temp = sp.random(N, N, density=density, format='csr')
    A_temp = (A_temp + A_temp.T) * 0.5
    degrees = np.array(A_temp.sum(axis=1)).flatten()
    Delta = sp.diags(degrees) - A_temp
    return W, W_in, Delta


def energy_entropy(energy_series):
    """
    Normalized energy entropy per neuron over time window.
    Much more robust than variance under noise.
    energy_series: shape (T, N)
    returns: shape (N,)
    """
    # Normalize each neuron's energy to a probability distribution
    E = energy_series - energy_series.min(axis=0, keepdims=True) + eps
    E_norm = E / (E.sum(axis=0, keepdims=True) + eps)
    # Shannon entropy
    H = -np.sum(E_norm * np.log(E_norm + eps), axis=0)
    # Normalize by log(T) so values are in [0, 1]
    H_norm = H / np.log(energy_series.shape[0] + eps)
    return H_norm


def get_derivative(Psi_curr, xi_vec, adapt_curr, alpha_curr, noise_in, I_in, W_curr, W_in, Delta):
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


def run_sim_m38(signal_func, total_time=400.0, verbose=True, t_skip=None, blk_dur=None):
    if t_skip is None: t_skip = transition_skip
    if blk_dur is None: blk_dur = block_duration
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
    xi_frozen = False; xi_frozen_val = None

    psi_buffer = np.zeros((window_steps, N), dtype=complex)
    phi_input_buffer = np.zeros((window_steps, 1))
    buf_idx = 0; buf_filled = False

    feats_plv = []; feats_entropy = []; feats_spec = []
    targets_Y = []; harvest_T = []

    for t in range(steps):
        ct = t * dt
        noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
        I_val, Y_val, freq = signal_func(ct)

        if freq > 0: phi_in = (2 * np.pi * freq * ct) % (2 * np.pi)
        else: phi_in = 0.0

        Wc = W.tocsr()
        k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k2 = get_derivative(Psi+0.5*dt*k1, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k3 = get_derivative(Psi+0.5*dt*k2, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        k4 = get_derivative(Psi+dt*k3, xi_vec, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
        Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

        k1g = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
        Psi_ghost = Psi_ghost + dt * k1g

        instant_energy = np.abs(Psi)**2
        E_avg_vec = 0.99*E_avg_vec + 0.01*instant_energy
        mean_energy = np.mean(E_avg_vec)

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
            W[rows, cols] += update - decay_hebb * W[rows, cols]
            try:
                ev = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
                if np.abs(ev[0]) > 0: W = W * (0.9 / np.abs(ev[0]))
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

                # Feature 1: PLV (phase locking value)
                phi_neuron = np.angle(ordered_psi)
                delta_phi = np.angle(np.exp(1j * (phi_neuron - ordered_phi_in)))
                plv = np.abs(np.mean(np.exp(1j * delta_phi), axis=0))

                # Feature 2: ENERGY ENTROPY (replaces variance)
                # Robust to noise — measures temporal regularity, not amplitude
                energy_series = np.abs(ordered_psi)**2
                ent = energy_entropy(energy_series)

                # Feature 3: Spectral bands
                energy_centered = energy_series - energy_series.mean(axis=0, keepdims=True)
                fft_result = np.fft.rfft(energy_centered, axis=0)
                power = np.abs(fft_result)**2
                freqs_fft = np.fft.rfftfreq(window_steps, d=dt)
                bands = [(0.3, 0.7), (0.8, 1.5), (1.5, 2.5), (2.5, 5.0)]
                spec_feats = []
                for f_lo, f_hi in bands:
                    mask = (freqs_fft >= f_lo) & (freqs_fft <= f_hi)
                    if np.any(mask): spec_feats.append(np.mean(power[mask], axis=0))
                    else: spec_feats.append(np.zeros(N))
                spec = np.concatenate(spec_feats)

                feats_plv.append(plv)
                feats_entropy.append(ent)
                feats_spec.append(spec)
                targets_Y.append(Y_val)
                harvest_T.append(ct)

    return (np.array(feats_plv), np.array(feats_entropy),
            np.array(feats_spec), np.array(targets_Y), np.array(harvest_T))