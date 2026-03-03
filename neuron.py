import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# -----------------------------
# 1. PARAMETERS
# -----------------------------
N = 500
alpha = 0.1
xi = 1.2
lam = 0.8
gamma = 0.5
eps = 1e-6

dt = 0.05
total_time = 300.0 # Longer experiment
steps = int(total_time / dt)

# LEARNING
eta = 0.005        # Stronger learning for this test
decay_rate = 0.0005

# -----------------------------
# 2. BUILD STRUCTURE
# -----------------------------
print("Building network...")
density = 0.02
W_real = sp.random(N, N, density=density, format='csr', data_rvs=np.random.randn)
W_imag = sp.random(N, N, density=density, format='csr', data_rvs=np.random.randn)
W = (W_real + 1j * W_imag) / np.sqrt(N * density)

# Laplacian
A = sp.random(N, N, density=density, format='csr')
A = (A + A.T) * 0.5
degrees = np.array(A.sum(axis=1)).flatten()
D = sp.diags(degrees)
Delta = D - A

# -----------------------------
# 3. DEFINE THE "MEMORY"
# -----------------------------
# We create a specific pattern P.
# Let's make it a simple spatial wave: Pattern A.
# Nodes 0-100 active, others quiet.
# In a real brain, this would be a specific sparse code.
Pattern_A = np.zeros(N, dtype=np.complex128)
Pattern_A[0:50] = 1.0 * np.exp(1j * 0.5) # Specific amplitude and phase
Pattern_A[100:150] = 1.0 * np.exp(1j * -0.5)
# Normalize
Pattern_A = Pattern_A / np.linalg.norm(Pattern_A)

# Function to measure similarity (Recall)
def calculate_recall(Psi, Pattern):
    # Overlap: dot product normalized
    # We want to measure if the PHASE and AMPLITUDE align
    overlap = np.abs(np.vdot(Pattern, Psi)) / (np.linalg.norm(Pattern) * np.linalg.norm(Psi))
    return overlap

# -----------------------------
# 4. STATE STORAGE
# -----------------------------
Psi = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1

# History
history = {
    't': [],
    'energy': [],
    'coherence': [],
    'recall_A': [],
    'spectral_radius': [],
    'phase': [] # 0=Chaos, 1=Training, 2=Recall
}

# -----------------------------
# 5. DYNAMICS
# -----------------------------

def get_derivative(Psi_curr):
    D = W @ Psi_curr
    num = np.real(Psi_curr.conj() * D)
    den = (np.abs(Psi_curr)**2) + (np.abs(D)**2) + eps
    R = num / den
    g = xi * np.tanh(1.0 - R) - lam
    dPsi = 1j*(W @ Psi_curr) + alpha*(Delta @ Psi_curr) + (g * Psi_curr) - (gamma * (np.abs(Psi_curr)**2) * Psi_curr)
    return dPsi

def apply_plasticity(Psi_curr):
    rows, cols = W.nonzero()
    amp_i = np.abs(Psi_curr[rows])
    amp_j = np.abs(Psi_curr[cols])
    corr = Psi_curr[rows] * np.conj(Psi_curr[cols])
    update = eta * corr * amp_i * amp_j
    W.data = W.data + update - decay_rate * W.data

# -----------------------------
# 6. EXPERIMENT TIMELINE
# -----------------------------
# Phase 1: Baseline (0-50s) - Just noise
# Phase 2: Training (50-150s) - Inject Pattern A + Learn
# Phase 3: Recall Test (150-300s) - Remove input, see if it holds

print("Running Experiment...")

for t in range(steps):
    curr_time = t*dt
    
    # --- DYNAMICS ---
    k1 = get_derivative(Psi)
    k2 = get_derivative(Psi + 0.5*dt*k1)
    k3 = get_derivative(Psi + 0.5*dt*k2)
    k4 = get_derivative(Psi + dt*k3)
    Psi = Psi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # --- INPUT INJECTION (Phase 2 Only) ---
    if 50.0 < curr_time < 150.0:
        # Inject Pattern A strongly
        # Force the state toward Pattern A
        injection_strength = 2.0
        Psi = Psi + dt * injection_strength * (Pattern_A - Psi) 
        phase_tag = 1 # Training
        
        # LEARN
        apply_plasticity(Psi)
        
    else:
        phase_tag = 0 if curr_time < 50 else 2
        # Slow background learning in other phases
        if t % 10 == 0: apply_plasticity(Psi)

    # --- DIAGNOSTICS ---
    if t % 20 == 0:
        history['t'].append(curr_time)
        history['energy'].append(np.mean(np.abs(Psi)**2))
        
        phases = np.angle(Psi)
        coherence = np.abs(np.mean(np.exp(1j * phases)))
        history['coherence'].append(coherence)
        
        recall = calculate_recall(Psi, Pattern_A)
        history['recall_A'].append(recall)
        
        history['phase'].append(phase_tag)
        
        # Calculate Spectral Radius (Expensive, so do it rarely)
        if t % 200 == 0:
            # Approximate largest eigenvalue magnitude of W
            # Using sparse eigsh for real part of largest eigenvalue of W*W.T? 
            # Simplification: Just check norm of W for now as proxy for weight growth
            # Or: Eigendecomposition of effective linear part if needed.
            # For now, we skip spectral radius to keep code fast, use ||W|| as proxy
            history['spectral_radius'].append(np.linalg.norm(W.data))

# -----------------------------
# 7. VISUALIZATION
# -----------------------------
print("Plotting results...")
plt.style.use('dark_background')
fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
fig.suptitle('Milestone 1: Associative Memory Formation Test', fontsize=18)

# Background shading for phases
phases = history['phase']
times = history['t']
for i, p in enumerate(phases):
    c = 'black'
    if p == 1: c = 'darkgreen' # Training
    if p == 2: c = 'darkblue'  # Recall
    axs[0].axvspan(times[i], times[i]+(times[1]-times[0]), facecolor=c, alpha=0.2)

# Plot 1: Energy
axs[0].plot(times, history['energy'], color='cyan')
axs[0].set_ylabel('System Energy')
axs[0].set_title('Metabolic Stability')

# Plot 2: Coherence
axs[1].plot(times, history['coherence'], color='yellow')
axs[1].set_ylabel('Kuramoto Coherence')
axs[1].set_title('Network Synchronization')

# Plot 3: Recall Metric (THE CRITICAL PLOT)
axs[2].plot(times, history['recall_A'], color='lime', linewidth=2)
axs[2].set_ylabel('Recall Overlap with Pattern A')
axs[2].set_title('INTELLIGENCE METRIC: Memory Retention')
axs[2].set_ylim(0, 1)
axs[2].axhline(y=0.5, color='red', linestyle='--', label='Recall Threshold')

# Plot 4: Weight Norm (Growth)
axs[3].plot(times, np.interp(times, times[::10], history['spectral_radius']), color='red')
axs[3].set_ylabel('Weight Magnitude ||W||')
axs[3].set_xlabel('Time (s)')

# Add text annotations
axs[2].text(75, 0.9, "TRAINING", color='white', fontsize=12)
axs[2].text(200, 0.9, "RECALL TEST", color='white', fontsize=12)

plt.show()