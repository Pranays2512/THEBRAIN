import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# =============================================================
# MILESTONE 33: TRANSITION-SKIPPED RANDOM SPLIT
#
# M32 broke one-sided class collapse — both classes above chance
# (A=52.75%, B=64.58%). But 41pp gap remains because:
#   First 60% of each block = transition transients (reservoir adjusting)
#   Last 40% of each block = steady-state
#   Classifier learns on transients, tests on steady-states
#
# M33 Strategy:
#   1. Skip first 15s of each 50s block (transition settling period)
#   2. Use remaining 35s of steady-state data
#   3. Randomly shuffle and split 60/40 into train/test
#   4. Both train and test now contain ONLY settled reservoir states
#   5. Same frozen-xi + Psi-only + PCA(50) + Ridge(α=1000) pipeline
# =============================================================

N = 500
lam = 0.8
gamma = 0.5           
eps = 1e-6

dt = 0.05
total_time = 400.0 
steps = int(total_time / dt)

# ARCHITECTURE
target_energy = 2.5   
input_gain = 1.5        # M26-level (gave best balanced result)

# EXCITATION (Xi)
eta_xi_up = 0.005       
eta_xi_down = 0.002     
xi_min = 0.1            
xi_max = 3.0            

# INHIBITION (A) — kept active even after xi freezes
tau_adapt = 1.0
kappa_adapt = 0.5       
adapt_max = 2.0

# DIFFUSION
alpha_base = 0.1      
alpha_max = 0.3
target_lyap = 0.1
eta_alpha = 0.0005
lyap_window = 100

S_global = 1.0

# LEARNING
learning_end_time = 100.0
learn_interval = 20      
eta_hebb = 0.002         
decay_hebb = 0.0001      
noise_amp = 0.05

# TRAINING PROTOCOL
stabilization_time = 120.0
train_cutoff_time = 260.0
block_duration = 50.0
transition_skip = 15.0      # M33: Skip first 15s of each block (transient settling)
energy_gate = 0.5           # Tight gate (restored)

# READOUT — M31: HEAVY REGULARIZATION
pca_dims = 50               # M31: Was 200 → 50. Force low-rank generalization.
ridge_alpha = 1000.0         # M31: Was 100 → 1000. Aggressively prevent overfitting.

# =============================================================
# BUILD STRUCTURE
# =============================================================
print("Initializing Milestone 33: Transition-Skipped Random Split...")
density = 0.02
W_real = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
W_imag = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
W = (W_real + 1j * W_imag)

def normalize_spectral_radius(W_matrix, target_sr=0.9):
    try:
        W_csr = W_matrix.tocsr()
        eigenvals = sp.linalg.eigs(W_csr, k=1, return_eigenvectors=False)
        max_eigen = np.abs(eigenvals[0])
        if max_eigen > 0:
            return W_matrix * (target_sr / max_eigen)
    except:
        pass
    return W_matrix

W = normalize_spectral_radius(W)

# Input Setup — Complex broadcast to all neurons
np.random.seed(42)
W_in = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.5

# =============================================================
# SIGNAL GENERATOR — Pure sine (no harmonic overlap)
# =============================================================
def get_signal(t):
    """50s alternating blocks: A B A B A B A B"""
    block = int(t / block_duration) % 2
    if block == 0:
        label = -1; freq = 0.5
    else:
        label = 1; freq = 2.0
    val = np.sin(2 * np.pi * freq * t)
    return val, label

# =============================================================
# STATE INITIALIZATION
# =============================================================
Psi = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1
xi_vec = np.ones(N) * 0.5      
A_vec = np.zeros(N)            
E_avg_vec = np.ones(N) * 0.1
alpha_global = alpha_base      

# Ghost State
Psi_ghost = Psi + (np.random.randn(N) + 1j*np.random.randn(N)) * 1e-5
prev_dist = np.linalg.norm(Psi_ghost - Psi)
Lyap_history = [] 

# M31: Xi freeze flag
xi_frozen = False
xi_frozen_val = None

# Storage — M31: stripped down, just Psi
states_X = []
targets_Y = []
harvest_times = []
skipped_harvest = 0

history = {
    't': [], 'energy': [], 'alpha': [], 'mean_xi': [], 'lyap_exp': [], 
    'w_norm': [], 'mean_adapt': []
}

# Build Delta
A_temp = sp.random(N, N, density=density, format='csr')
A_temp = (A_temp + A_temp.T) * 0.5
degrees = np.array(A_temp.sum(axis=1)).flatten()
D_mat = sp.diags(degrees)
Delta = D_mat - A_temp

# =============================================================
# DYNAMICS — Full equation restored (M30 proved removal hurts)
# =============================================================

def get_derivative(Psi_curr, xi_curr, adapt_curr, alpha_curr, noise_in, I_in, W_curr):
    W_eff = S_global * W_curr
    D = W_eff @ Psi_curr
    
    num = np.real(Psi_curr.conj() * D)
    den = (np.abs(Psi_curr)**2) + (np.abs(D)**2) + eps
    R = num / den
    
    g_vec = xi_curr * np.tanh(1.0 - R) - lam 
    effective_gamma = gamma + adapt_curr
    
    # FULL dynamics — 1j*W@Psi RESTORED
    dPsi = (1j*(W_eff @ Psi_curr) 
            + alpha_curr*(Delta @ Psi_curr) 
            + (g_vec * Psi_curr) 
            - (effective_gamma * (np.abs(Psi_curr)**2) * Psi_curr))
    
    dPsi += noise_amp * noise_in
    dPsi += W_in * I_in * input_gain
    return dPsi

print("Running Milestone 33: Transition-Skipped Random Split...")
print(f"  Learning ends: {learning_end_time}s | Harvest starts: {stabilization_time}s")
print(f"  Xi freezes at: {stabilization_time}s | A stays active")
print(f"  Block: {block_duration}s | Skip first {transition_skip}s (transients) | Use last {block_duration-transition_skip}s")
print(f"  Features: Psi ({2*N} dims) → Scale → PCA({pca_dims}) → Ridge(α={ridge_alpha})")

for t in range(steps):
    curr_time = t * dt
    
    # --- 0. Inputs ---
    noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
    I_val, Y_val = get_signal(curr_time)
    W_snap = W.tocsr() 
    
    # --- 1. PHYSICS STEP (RK4) ---
    k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global, noise_vec, I_val, W_snap)
    k2 = get_derivative(Psi + 0.5*dt*k1, xi_vec, A_vec, alpha_global, noise_vec, I_val, W_snap)
    k3 = get_derivative(Psi + 0.5*dt*k2, xi_vec, A_vec, alpha_global, noise_vec, I_val, W_snap)
    k4 = get_derivative(Psi + dt*k3, xi_vec, A_vec, alpha_global, noise_vec, I_val, W_snap)
    Psi = Psi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Ghost
    k1_g = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global, noise_vec, 0, W_snap) 
    k2_g = get_derivative(Psi_ghost + 0.5*dt*k1_g, xi_vec, A_vec, alpha_global, noise_vec, 0, W_snap)
    k3_g = get_derivative(Psi_ghost + 0.5*dt*k2_g, xi_vec, A_vec, alpha_global, noise_vec, 0, W_snap)
    k4_g = get_derivative(Psi_ghost + dt*k3_g, xi_vec, A_vec, alpha_global, noise_vec, 0, W_snap)
    Psi_ghost = Psi_ghost + (dt/6.0)*(k1_g + 2*k2_g + 2*k3_g + k4_g)
    
    # --- 2. HOMEOSTASIS ---
    instant_energy = np.abs(Psi)**2
    E_avg_vec = (1 - 0.01) * E_avg_vec + 0.01 * instant_energy
    mean_energy = np.mean(E_avg_vec)
    
    # M31: FREEZE XI after stabilization, keep A active
    if curr_time >= stabilization_time and not xi_frozen:
        xi_frozen = True
        xi_frozen_val = xi_vec.copy()
        print(f"  ** Xi FROZEN at t={curr_time:.1f}s, mean xi={np.mean(xi_vec):.3f} **")
    
    if not xi_frozen:
        # Xi still adapting during warm-up
        error_energy = target_energy - E_avg_vec
        if curr_time < 10.0:
            dXi = eta_xi_up * np.maximum(0, error_energy)
        else:
            rate = np.where(error_energy < 0, eta_xi_down, eta_xi_up)
            dXi = rate * error_energy
        xi_vec += dXi
        xi_vec = np.clip(xi_vec, xi_min, xi_max)
    else:
        # Xi locked — use frozen value
        xi_vec = xi_frozen_val.copy()
    
    # A (inhibition) always active — provides fine energy control with frozen xi
    excess_energy = np.maximum(0, E_avg_vec - target_energy)
    dA = (kappa_adapt * excess_energy - A_vec) / tau_adapt
    A_vec += dt * dA
    A_vec = np.clip(A_vec, 0.0, adapt_max)

    # --- 3. CHAOS CONTROL ---
    current_dist = np.linalg.norm(Psi_ghost - Psi)
    instant_lyap = 0.0
    
    if current_dist < 1e-7 or current_dist > 1.0:
        Psi_ghost = Psi + (np.random.randn(N) + 1j*np.random.randn(N)) * 1e-4
        prev_dist = 1e-4
    else:
        instant_lyap = np.log(current_dist + 1e-12) - np.log(prev_dist + 1e-12)
        Lyap_history.append(instant_lyap)
        prev_dist = current_dist

    if len(Lyap_history) > lyap_window: Lyap_history.pop(0)
    lyap_smooth = np.mean(Lyap_history) if len(Lyap_history) > 0 else 0.0
    
    error_lyap = target_lyap - lyap_smooth
    alpha_global += eta_alpha * error_lyap
    alpha_global = np.clip(alpha_global, alpha_base, alpha_max)

    # --- 4. LEARNING ---
    if curr_time < learning_end_time and (t % learn_interval == 0):
        rows, cols = W.nonzero()
        amp_i = np.abs(Psi[rows])
        amp_j = np.abs(Psi[cols])
        corr = Psi[rows] * np.conj(Psi[cols])
        update = eta_hebb * corr * amp_i * amp_j
        W[rows, cols] += update - decay_hebb * W[rows, cols]
        W = normalize_spectral_radius(W)

    # --- 5. DATA HARVEST — M31: ONLY Psi, nothing else ---
    if curr_time > stabilization_time:
        if abs(mean_energy - target_energy) < energy_gate:
            # Just Psi. No temporal stacking. No homeo vars. No phase velocity.
            states_X.append(np.concatenate([Psi.real, Psi.imag]))
            targets_Y.append(Y_val)
            harvest_times.append(curr_time)
        else:
            skipped_harvest += 1

    # --- 6. RECORD ---
    if t % 20 == 0:
        history['t'].append(curr_time)
        history['energy'].append(mean_energy)
        history['mean_xi'].append(np.mean(xi_vec))
        history['alpha'].append(alpha_global)
        history['lyap_exp'].append(lyap_smooth)
        history['w_norm'].append(sp.linalg.norm(W))
        history['mean_adapt'].append(np.mean(A_vec))

    # --- 7. PROGRESS ---
    if t % 2000 == 0:
        _, curr_label = get_signal(curr_time)
        class_str = "A(slow)" if curr_label == -1 else "B(fast)"
        learn_str = "LEARNING" if curr_time < learning_end_time else "frozen"
        xi_str = "FROZEN" if xi_frozen else "active"
        print(f"  t={curr_time:6.1f}s [{class_str}] [W:{learn_str}] [xi:{xi_str}]  "
              f"E={mean_energy:.3f}  xi={np.mean(xi_vec):.3f}  A={np.mean(A_vec):.3f}  "
              f"alpha={alpha_global:.4f}  |W|={sp.linalg.norm(W):.3f}")

# =============================================================
# TRAIN READOUT — M33: Transition-Skipped Random Split
# =============================================================
print(f"\n--- Training Readout Layer ---")
print(f"  Samples collected: {len(states_X)} (skipped {skipped_harvest} energy-gated)")

if len(states_X) == 0:
    print("  ERROR: No samples collected! Energy gate too tight.")
    exit(1)

X = np.array(states_X)
Y = np.array(targets_Y)
T = np.array(harvest_times)

print(f"  Feature dimensions: {X.shape[1]} (Psi only — no stacking, no homeo)")

# M33: TRANSITION-SKIPPED RANDOM SPLIT
# 1. Compute position within each block
# 2. Skip first transition_skip seconds (transient settling)
# 3. Randomly split remaining steady-state samples 60/40
time_in_block = T % block_duration  # position within current block (0 to block_duration)
settled_mask = time_in_block >= transition_skip  # only keep steady-state portion

X_settled = X[settled_mask]
Y_settled = Y[settled_mask]
T_settled = T[settled_mask]

n_skipped_transient = np.sum(~settled_mask)
print(f"  Transition skip: {transition_skip}s → dropped {n_skipped_transient} transient samples")
print(f"  Settled samples: {len(Y_settled)} (from last {block_duration-transition_skip}s of each block)")

if len(Y_settled) == 0:
    print("  ERROR: No settled samples! Check transition_skip vs block_duration.")
    exit(1)

# Random 60/40 split of settled data
rng_split = np.random.default_rng(42)
shuffled_idx = rng_split.permutation(len(Y_settled))
split_point = int(0.6 * len(Y_settled))
train_idx = np.sort(shuffled_idx[:split_point])
test_idx = np.sort(shuffled_idx[split_point:])

X_train, Y_train = X_settled[train_idx], Y_settled[train_idx]
X_test, Y_test = X_settled[test_idx], Y_settled[test_idx]

# Keep full arrays for later use (only settled)
X = X_settled
Y = Y_settled
T = T_settled

n_A_train = np.sum(Y_train == -1)
n_B_train = np.sum(Y_train == 1)
n_A_test = np.sum(Y_test == -1)
n_B_test = np.sum(Y_test == 1)

print(f"  Random split (seed=42): 60% train, 40% test")
print(f"  Train samples: {len(Y_train)} ({n_A_train} A, {n_B_train} B)")
print(f"  Test samples:  {len(Y_test)} ({n_A_test} A, {n_B_test} B)")

if len(Y_test) == 0:
    print("  ERROR: No test samples!")
    exit(1)

# Class balancing (train only)
if n_A_train > 0 and n_B_train > 0:
    min_class = min(n_A_train, n_B_train)
    idx_A = np.where(Y_train == -1)[0]
    idx_B = np.where(Y_train == 1)[0]
    rng = np.random.default_rng(42)
    if n_A_train > min_class:
        idx_A = rng.choice(idx_A, size=min_class, replace=False)
    if n_B_train > min_class:
        idx_B = rng.choice(idx_B, size=min_class, replace=False)
    balanced_idx = np.sort(np.concatenate([idx_A, idx_B]))
    X_train_bal = X_train[balanced_idx]
    Y_train_bal = Y_train[balanced_idx]
    print(f"  Balanced train: {len(Y_train_bal)} ({np.sum(Y_train_bal==-1)} A, {np.sum(Y_train_bal==1)} B)")
else:
    X_train_bal = X_train
    Y_train_bal = Y_train
    print("  WARNING: One class missing!")

# M33: StandardScaler + PCA + Ridge
from sklearn.decomposition import PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X)

actual_pca = min(pca_dims, len(X_train_bal), X_train_scaled.shape[1])
print(f"\n  StandardScaler → PCA: {X.shape[1]} → {actual_pca} dims")
pca = PCA(n_components=actual_pca)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
X_all_pca = pca.transform(X_all_scaled)
explained_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"  PCA explained variance: {explained_var:.1f}%")
print(f"  Ridge α = {ridge_alpha}")

model = Ridge(alpha=ridge_alpha)
model.fit(X_train_pca, Y_train_bal)

# Accuracies
pred_train = model.predict(X_train_pca)
acc_train = np.mean((pred_train > 0) == (Y_train_bal > 0))

pred_test = model.predict(X_test_pca)
acc_test = np.mean((pred_test > 0) == (Y_test > 0))

classA_mask = Y_test == -1
classB_mask = Y_test == 1
acc_A = np.mean((pred_test[classA_mask] > 0) == (Y_test[classA_mask] > 0)) if np.any(classA_mask) else 0.0
acc_B = np.mean((pred_test[classB_mask] > 0) == (Y_test[classB_mask] > 0)) if np.any(classB_mask) else 0.0

pred_all = model.predict(X_all_pca)
acc_full = np.mean((pred_all > 0) == (Y > 0))

print(f"\n{'='*55}")
print(f"  MILESTONE 33: TRANSITION-SKIPPED RANDOM SPLIT")
print(f"{'='*55}")
print(f"  Transition skip:        {transition_skip}s per block")
print(f"  Settled window:         {block_duration-transition_skip}s per block")
print(f"  Transient dropped:      {n_skipped_transient} samples")
print(f"  Features:               Psi only ({X.shape[1]} dims)")
print(f"  PCA features:           {actual_pca} ({explained_var:.1f}% var)")
print(f"  Full dataset accuracy:  {acc_full*100:.2f}%")
print(f"  Train accuracy:         {acc_train*100:.2f}%")
print(f"  Test accuracy:          {acc_test*100:.2f}%")
print(f"  Test Class A (slow):    {acc_A*100:.2f}%")
print(f"  Test Class B (fast):    {acc_B*100:.2f}%")
print(f"  Train-Test gap:         {(acc_train-acc_test)*100:.1f}pp")
print(f"{'='*55}")

# Xi drift check
train_xis = [history['mean_xi'][i] for i, t_val in enumerate(history['t']) 
             if stabilization_time < t_val < train_cutoff_time]
test_xis = [history['mean_xi'][i] for i, t_val in enumerate(history['t']) 
            if t_val >= train_cutoff_time]
train_energies = [history['energy'][i] for i, t_val in enumerate(history['t']) 
                  if stabilization_time < t_val < train_cutoff_time]
test_energies = [history['energy'][i] for i, t_val in enumerate(history['t']) 
                 if t_val >= train_cutoff_time]
if train_xis and test_xis:
    print(f"\n  Xi during train:     {np.mean(train_xis):.3f} ± {np.std(train_xis):.4f}")
    print(f"  Xi during test:      {np.mean(test_xis):.3f} ± {np.std(test_xis):.4f}")
    print(f"  Δxi train→test:     {abs(np.mean(train_xis) - np.mean(test_xis)):.4f}")
if train_energies and test_energies:
    print(f"  Energy during train: {np.mean(train_energies):.3f} ± {np.std(train_energies):.3f}")
    print(f"  Energy during test:  {np.mean(test_energies):.3f} ± {np.std(test_energies):.3f}")
    print(f"  ΔE train→test:      {abs(np.mean(train_energies) - np.mean(test_energies)):.4f}")

print()

# =============================================================
# VISUALIZATION
# =============================================================
plt.style.use('dark_background')
fig, axs = plt.subplots(6, 1, figsize=(14, 18), sharex=False)
fig.suptitle(f'Milestone 33: Transition-Skipped Random Split (skip={transition_skip}s, PCA={actual_pca}, Ridge α={ridge_alpha})\n'
             f'Train: {acc_train*100:.1f}% | Test: {acc_test*100:.1f}% '
             f'(A:{acc_A*100:.0f}% B:{acc_B*100:.0f}%) | Gap: {(acc_train-acc_test)*100:.0f}pp', 
             fontsize=14, fontweight='bold')

times = history['t']

axs[0].plot(times, history['energy'], color='cyan', linewidth=0.8)
axs[0].axhline(y=target_energy, color='red', linestyle='--', alpha=0.7, label=f'Target={target_energy}')
axs[0].axvline(x=stabilization_time, color='yellow', linestyle=':', alpha=0.5, label='Xi freeze + harvest start')
axs[0].set_title('Energy')
axs[0].set_ylabel('|Ψ|²')
axs[0].legend(loc='upper right', fontsize=8)

axs[1].plot(times, history['mean_xi'], color='yellow', linewidth=0.8, label='ξ')
axs[1].plot(times, history['mean_adapt'], color='purple', linewidth=0.8, label='A')
axs[1].axvline(x=stabilization_time, color='white', linestyle=':', alpha=0.5, label='Xi freeze')
axs[1].axvline(x=train_cutoff_time, color='red', linestyle='--', alpha=0.5, label='Train/Test')
axs[1].set_title('Homeostatic Variables (Xi should be FLAT after freeze line)')
axs[1].legend(loc='upper right', fontsize=8)

axs[2].plot(times, history['lyap_exp'], color='lime', linewidth=0.8)
axs[2].axhline(y=target_lyap, color='red', linestyle='--', alpha=0.7)
axs[2].set_title('Lyapunov')
axs[2].set_ylabel('λ')

axs[3].plot(times, history['alpha'], color='orange', linewidth=0.8)
axs[3].set_title('Alpha')
axs[3].set_ylabel('α')

axs[4].plot(times, history['w_norm'], color='red', linewidth=0.8)
axs[4].axvline(x=learning_end_time, color='yellow', linestyle=':', alpha=0.5, label='Learn freeze')
axs[4].set_title('||W||')
axs[4].legend(loc='upper right')

# Predictions — M33: use train_idx/test_idx into settled data
if len(Y_test) > 0:
    axs[5].scatter(T[train_idx], Y[train_idx], c='gray', s=1, alpha=0.3, label='Train')
    axs[5].scatter(T[test_idx], Y_test, c='white', s=2, alpha=0.5, label='Test (true)')
    correct = (pred_test > 0) == (Y_test > 0)
    colors = np.where(correct, 'lime', 'red')
    axs[5].scatter(T[test_idx], np.sign(pred_test), c=colors, s=2, alpha=0.5, label='Test (pred)')
axs[5].set_title('Predictions (green=correct, random split)')
axs[5].set_yticks([-1, 1])
axs[5].set_yticklabels(['A', 'B'])
axs[5].legend(loc='upper right', markerscale=5)

plt.tight_layout()
plt.savefig('/Users/pranay./Documents/THEBRAIN/m33_results.png', dpi=150, bbox_inches='tight')
print("Plot saved to m33_results.png")
plt.show()