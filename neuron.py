"""
M36: RESONANT RESERVOIR
========================
M35 proved the reservoir encodes frequency via energy variance + spectral
features, but regression failed due to manifold compression at edges.

Root cause: statistical estimation (counting cycles) hits time-bandwidth limit.

M36 fix: Replace with Hopf oscillator reservoir.
  - Each neuron has intrinsic frequency ω_i (log-spaced 0.3→3.0 Hz)
  - Constant-Q bandwidth: γ_i = ω_i / Q (Q=15)
  - Dynamics: dΨ_i = (iω_i + g_i - γ_i|Ψ_i|²)Ψ_i + weak coupling + input
  - Readout: |Ψ_i|² snapshot (spatial frequency map)

When input=f Hz, neuron with ω_i≈f resonates (large |Ψ_i|).
No windowing needed — single snapshot IS a frequency spectrum.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time as clock

# =============================================================
# M36: RESONANT RESERVOIR PHYSICS
# =============================================================
N = 500
lam = 0.8
eps = 1e-6
dt = 0.05
target_energy = 2.5
input_gain = 3.0  # M36: strong input to drive resonance

# M36: Natural frequencies — log-spaced from 0.3 to 3.0 Hz (in radians/s: ×2π)
omega_hz = np.logspace(np.log10(0.3), np.log10(3.0), N)
omega_vec = 2.0 * np.pi * omega_hz  # convert to angular frequency

# M36: Constant-Q damping — bandwidth scales with frequency
Q_factor = 15.0
gamma_vec = omega_hz / Q_factor  # bandwidth ~ ω/Q

# M36: Global homeostatic excitation (NOT per-neuron — that destroys resonance pattern)
# Per-neuron xi equalizes energy across neurons, killing the spatial frequency map
# Global xi controls overall gain without masking which neurons resonate
eta_xi = 0.002
xi_min = 0.1
xi_max = 3.0

# Adaptation
tau_adapt_vec = np.linspace(0.2, 5.0, N)
kappa_adapt = 0.5
adapt_max = 2.0

# Chaos control (may be less relevant for resonant system)
alpha_base = 0.01   # reduced — less diffusion needed
alpha_max = 0.1
target_lyap = 0.05
eta_alpha = 0.0005
lyap_window = 100

# M36: Very weak coupling — preserve individual resonances
coupling_strength = 0.1

# Learning
learning_end_time = 100.0
learn_interval = 20
eta_hebb = 0.001    # reduced — don't want coupling to overwhelm resonance
decay_hebb = 0.0001
noise_amp = 0.02    # reduced noise

# Protocol
stabilization_time = 120.0
energy_gate = 1.0   # wider gate for resonant system
ridge_alpha = 1000.0
density = 0.02
block_duration = 50.0
transition_skip = 5.0  # resonance settles faster than attractor

# M36: No windowing needed — snapshot readout
feature_sample_interval = 10  # sample every 0.5s


def build_network():
    W_real = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W_imag = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
    W = (W_real + 1j * W_imag)
    try:
        eigenvals = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
        max_eigen = np.abs(eigenvals[0])
        if max_eigen > 0:
            W = W * (0.9 / max_eigen)
    except:
        pass
    np.random.seed(42)
    W_in = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.5
    A_temp = sp.random(N, N, density=density, format='csr')
    A_temp = (A_temp + A_temp.T) * 0.5
    degrees = np.array(A_temp.sum(axis=1)).flatten()
    Delta = sp.diags(degrees) - A_temp
    return W, W_in, Delta


def get_derivative(Psi_curr, xi_curr, adapt_curr, alpha_curr, noise_in, I_in, W_curr, W_in, Delta):
    """M36: Hopf oscillator dynamics with weak coupling."""

    # Intrinsic Hopf oscillation — each neuron rotates at its own ω_i
    hopf_rotation = 1j * omega_vec * Psi_curr

    # Gain/loss
    num = np.real(Psi_curr.conj() * (W_curr @ Psi_curr))
    den = (np.abs(Psi_curr)**2) + eps
    R = num / den
    g_vec = xi_curr * np.tanh(1.0 - R) - lam

    # Saturation with constant-Q damping
    effective_gamma = gamma_vec + adapt_curr

    # Weak coupling (ε * W @ Ψ) — preserves individual resonances
    coupling = coupling_strength * (W_curr @ Psi_curr)

    # Spatial diffusion (very weak for resonant system)
    diffusion = alpha_curr * (Delta @ Psi_curr)

    dPsi = (hopf_rotation                                      # intrinsic frequency
            + g_vec * Psi_curr                                  # gain
            - effective_gamma * (np.abs(Psi_curr)**2) * Psi_curr  # saturation
            + coupling                                          # weak network coupling
            + diffusion                                         # spatial smoothing
            + noise_amp * noise_in                              # noise
            + W_in * I_in * input_gain)                         # input

    return dPsi


def get_signal(t):
    """50s alternating blocks: A B A B ..."""
    block = int(t / block_duration) % 2
    if block == 0:
        return np.sin(2 * np.pi * 0.5 * t), -1
    else:
        return np.sin(2 * np.pi * 2.0 * t), 1


# =============================================================
# RUN SIMULATION — Snapshot readout (no windowing)
# =============================================================
print("=" * 65)
print("  M36: RESONANT RESERVOIR")
print("=" * 65)
print(f"  N={N}, dt={dt}, total=400s")
print(f"  Natural frequencies: {omega_hz[0]:.2f} — {omega_hz[-1]:.2f} Hz (log-spaced)")
print(f"  Q factor: {Q_factor} → bandwidth: {gamma_vec[0]:.4f} — {gamma_vec[-1]:.4f}")
print(f"  Coupling strength: {coupling_strength}")
print(f"  Readout: |Ψ|² snapshot (no windowing)")
print(f"  Block: {block_duration}s, skip: {transition_skip}s")
print()

W, W_in, Delta = build_network()

# State
Psi = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1
xi_global = 0.5  # M36: SINGLE global gain, not per-neuron
A_vec = np.zeros(N)
E_avg_vec = np.ones(N) * 0.1
alpha_global = alpha_base
Psi_ghost = Psi + (np.random.randn(N) + 1j*np.random.randn(N)) * 1e-5
prev_dist = np.linalg.norm(Psi_ghost - Psi)
Lyap_history = []
xi_frozen = False
xi_frozen_val = None

total_time = 400.0
steps = int(total_time / dt)

# M36: Snapshot features — just |Ψ|² at each harvest point
features_X = []
targets_Y = []
harvest_times = []
skipped = 0

print("  Running simulation...")
t0 = clock.time()

for t in range(steps):
    ct = t * dt
    noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
    I_val, Y_val = get_signal(ct)
    xi_vec_broadcast = np.full(N, xi_global)  # broadcast scalar to vector
    Wc = W.tocsr()

    # RK4
    k1 = get_derivative(Psi, xi_vec_broadcast, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
    k2 = get_derivative(Psi+0.5*dt*k1, xi_vec_broadcast, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
    k3 = get_derivative(Psi+0.5*dt*k2, xi_vec_broadcast, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
    k4 = get_derivative(Psi+dt*k3, xi_vec_broadcast, A_vec, alpha_global, noise_vec, I_val, Wc, W_in, Delta)
    Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

    # Ghost
    k1g = get_derivative(Psi_ghost, xi_vec_broadcast, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
    k2g = get_derivative(Psi_ghost+0.5*dt*k1g, xi_vec_broadcast, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
    k3g = get_derivative(Psi_ghost+0.5*dt*k2g, xi_vec_broadcast, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
    k4g = get_derivative(Psi_ghost+dt*k3g, xi_vec_broadcast, A_vec, alpha_global, noise_vec, 0, Wc, W_in, Delta)
    Psi_ghost = Psi_ghost + (dt/6.0)*(k1g+2*k2g+2*k3g+k4g)

    # Homeostasis
    instant_energy = np.abs(Psi)**2
    E_avg_vec = 0.99*E_avg_vec + 0.01*instant_energy
    mean_energy = np.mean(E_avg_vec)

    if ct >= stabilization_time and not xi_frozen:
        xi_frozen = True
        xi_frozen_val = xi_global
        print(f"    Xi FROZEN at t={ct:.1f}s, xi_global={xi_global:.3f}")

    # M36: Global gain control — single scalar, preserves spatial pattern
    if not xi_frozen:
        mean_E = np.mean(E_avg_vec)
        error = target_energy - mean_E
        xi_global = np.clip(xi_global + eta_xi * error, xi_min, xi_max)
    else:
        xi_global = xi_frozen_val

    excess_energy = np.maximum(0, E_avg_vec - target_energy)
    A_vec = np.clip(A_vec + dt*((kappa_adapt*excess_energy - A_vec)/tau_adapt_vec), 0, adapt_max)

    # Chaos control
    current_dist = np.linalg.norm(Psi_ghost - Psi)
    if current_dist < 1e-7 or current_dist > 1.0:
        Psi_ghost = Psi + (np.random.randn(N)+1j*np.random.randn(N))*1e-4
        prev_dist = 1e-4
    else:
        Lyap_history.append(np.log(current_dist+1e-12) - np.log(prev_dist+1e-12))
        prev_dist = current_dist
    if len(Lyap_history) > lyap_window:
        Lyap_history.pop(0)
    lyap_smooth = np.mean(Lyap_history) if Lyap_history else 0.0
    alpha_global = np.clip(alpha_global + eta_alpha*(target_lyap - lyap_smooth), alpha_base, alpha_max)

    # Learning (weak)
    if ct < learning_end_time and (t % learn_interval == 0):
        rows, cols = W.nonzero()
        corr = Psi[rows] * np.conj(Psi[cols])
        update = eta_hebb * corr * np.abs(Psi[rows]) * np.abs(Psi[cols])
        W[rows, cols] += update - decay_hebb * W[rows, cols]
        try:
            eigenvals = sp.linalg.eigs(W.tocsr(), k=1, return_eigenvectors=False)
            if np.abs(eigenvals[0]) > 0:
                W = W * (0.9 / np.abs(eigenvals[0]))
        except:
            pass

    # M36: SNAPSHOT HARVEST — just |Ψ|², no windowing
    if ct > stabilization_time and (t % feature_sample_interval == 0):
        time_in_block = ct % block_duration
        if time_in_block >= transition_skip:
            features_X.append(np.abs(Psi)**2)  # energy per neuron = spatial freq map
            targets_Y.append(Y_val)
            harvest_times.append(ct)
        else:
            skipped += 1

    # Progress
    if t % 4000 == 0:
        print(f"    t={ct:6.1f}s  E={mean_energy:.3f}  samples={len(targets_Y)}")

t1 = clock.time()
print(f"  Simulation done in {t1-t0:.1f}s")

X = np.array(features_X)
Y = np.array(targets_Y)
T = np.array(harvest_times)

print(f"\n  Samples: {len(Y)} (skipped {skipped} transition)")
print(f"  Feature dims: {X.shape[1]} (|Ψ|² per neuron)")


# =============================================================
# TEMPORAL BLOCK SPLIT — Train on blocks 0-3, test on 4+
# =============================================================
print(f"\n--- Temporal Block Split Classification ---")

block_idx = (T / block_duration).astype(int)
first_block = int(stabilization_time / block_duration)
rel_block = block_idx - first_block

n_train_blocks = 4
train_mask = rel_block < n_train_blocks
test_mask = rel_block >= n_train_blocks

X_train, Y_train = X[train_mask], Y[train_mask]
X_test, Y_test = X[test_mask], Y[test_mask]

# Balance
classes = np.unique(Y_train)
min_c = min(np.sum(Y_train == c) for c in classes)
bal_idx = []
rng = np.random.default_rng(42)
for c in classes:
    ci = np.where(Y_train == c)[0]
    if len(ci) > min_c:
        ci = rng.choice(ci, size=min_c, replace=False)
    bal_idx.extend(ci)
X_tr = X_train[np.sort(bal_idx)]
Y_tr = Y_train[np.sort(bal_idx)]

# Scale + PCA + Ridge
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_test)

n_pca = min(50, len(X_tr), X_tr_sc.shape[1])
pca = PCA(n_components=n_pca)
X_tr_p = pca.fit_transform(X_tr_sc)
X_te_p = pca.transform(X_te_sc)

model = Ridge(alpha=ridge_alpha)
model.fit(X_tr_p, Y_tr)

pred_tr = model.predict(X_tr_p)
pred_te = model.predict(X_te_p)
acc_tr = np.mean((pred_tr > 0) == (Y_tr > 0))
acc_te = np.mean((pred_te > 0) == (Y_test > 0))

acc_A = np.mean((pred_te[Y_test == -1] > 0) == (Y_test[Y_test == -1] > 0)) if np.any(Y_test == -1) else 0
acc_B = np.mean((pred_te[Y_test == 1] > 0) == (Y_test[Y_test == 1] > 0)) if np.any(Y_test == 1) else 0

print(f"\n{'='*55}")
print(f"  M36: RESONANT RESERVOIR RESULTS")
print(f"{'='*55}")
print(f"  Features:    |Ψ|² snapshot ({X.shape[1]} dims)")
print(f"  Split:       Temporal blocks (train 0-{n_train_blocks-1}, test {n_train_blocks}+)")
print(f"  Samples:     {len(X_tr)} train, {len(X_test)} test")
print(f"  Train acc:   {acc_tr*100:.1f}%")
print(f"  Test acc:    {acc_te*100:.1f}%")
print(f"  Class A:     {acc_A*100:.1f}% (0.5 Hz)")
print(f"  Class B:     {acc_B*100:.1f}% (2.0 Hz)")
print(f"  Gap:         {(acc_tr-acc_te)*100:.1f}pp")
print(f"{'='*55}")

# Resonance diagnostic: which neurons respond to each frequency?
print(f"\n--- Resonance Diagnostic ---")
slow_mask = Y == -1  # 0.5 Hz
fast_mask = Y == 1   # 2.0 Hz

mean_energy_slow = np.mean(X[slow_mask], axis=0)
mean_energy_fast = np.mean(X[fast_mask], axis=0)

# Find peak neurons for each class
peak_slow = omega_hz[np.argmax(mean_energy_slow)]
peak_fast = omega_hz[np.argmax(mean_energy_fast)]

# Expected peaks
print(f"  Input 0.5 Hz → peak neuron at: {peak_slow:.2f} Hz (expected: 0.50)")
print(f"  Input 2.0 Hz → peak neuron at: {peak_fast:.2f} Hz (expected: 2.00)")

# Selectivity: ratio of peak to off-peak
selectivity_slow = np.max(mean_energy_slow) / (np.mean(mean_energy_slow) + eps)
selectivity_fast = np.max(mean_energy_fast) / (np.mean(mean_energy_fast) + eps)
print(f"  Selectivity (slow): {selectivity_slow:.2f}x peak-to-mean")
print(f"  Selectivity (fast): {selectivity_fast:.2f}x peak-to-mean")

if selectivity_slow > 3 and selectivity_fast > 3:
    print(f"  ✓ Strong resonance — spatial frequency map working")
elif selectivity_slow > 1.5 or selectivity_fast > 1.5:
    print(f"  ~ Moderate resonance — some frequency selectivity")
else:
    print(f"  ✗ No resonance — neurons not frequency-selective")

print()
