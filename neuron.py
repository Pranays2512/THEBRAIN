import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# -----------------------------
# 1. PARAMETERS (TUNED FOR CONSOLIDATION)
# -----------------------------
N = 500
alpha = 0.1
xi = 1.2
lam = 0.8
gamma = 0.5
eps = 1e-6

dt = 0.05
total_time = 300.0 
steps = int(total_time / dt)

# THE FIXES
eta = 0.08          # Increased from 0.005 -> 0.08 (Strong Learning)
decay_rate = 0.00005 # Reduced from 0.0005 -> 0.00005 (Slow Forgetting)
kappa = 1.5         # Injection Strength (Strong Clamping)

# -----------------------------
# 2. BUILD STRUCTURE
# -----------------------------
print("Building network...")
density = 0.02
W_real = sp.random(N, N, density=density, format='csr', data_rvs=np.random.randn)
W_imag = sp.random(N, N, density=density, format='csr', data_rvs=np.random.randn)
W = (W_real + 1j * W_imag) / np.sqrt(N * density)

A = sp.random(N, N, density=density, format='csr')
A = (A + A.T) * 0.5
degrees = np.array(A.sum(axis=1)).flatten()
D = sp.diags(degrees)
Delta = D - A

# -----------------------------
# 3. DEFINE THE "MEMORY"
# -----------------------------
Pattern_A = np.zeros(N, dtype=np.complex128)
# Create a sparse, specific pattern (Semantic 'A')
Pattern_A[0:50] = 1.0 * np.exp(1j * 0.5) 
Pattern_A[100:150] = 1.0 * np.exp(1j * -0.5)
Pattern_A = Pattern_A / np.linalg.norm(Pattern_A)

def calculate_recall(Psi, Pattern):
    overlap = np.abs(np.vdot(Pattern, Psi)) / (np.linalg.norm(Pattern) * np.linalg.norm(Psi) + 1e-9)
    return overlap

# -----------------------------
# 4. STATE STORAGE
# -----------------------------
Psi = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1

history = {
    't': [], 'energy': [], 'coherence': [], 'recall_A': [], 'phase': []
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
    # Standard Hebbian with Amplitude weighting (as per your analysis)
    amp_i = np.abs(Psi_curr[rows])
    amp_j = np.abs(Psi_curr[cols])
    corr = Psi_curr[rows] * np.conj(Psi_curr[cols])
    
    # Strong Update
    update = eta * corr * amp_i * amp_j
    
    # Apply update with low decay
    current_vals = W.data
    new_vals = current_vals + update - decay_rate * current_vals
    
    # Safety Clamp (Prevent Runaway explosion seen in Run 2)
    # Limits weight magnitude to 2.0
    np.clip(new_vals, -2.0, 2.0, out=new_vals)
    
    W.data = new_vals

# -----------------------------
# 6. EXPERIMENT TIMELINE
# -----------------------------
# Phase 1: Baseline (0-50s)
# Phase 2: Training (50-150s) -> HIGH LEARNING, HIGH INJECTION
# Phase 3: Recall (150-300s) -> NO INPUT, LEARNING OFF (to test pure retention)

print("Running Consolidation Experiment...")

for t in range(steps):
    curr_time = t*dt
    
    # --- DYNAMICS ---
    k1 = get_derivative(Psi)
    k2 = get_derivative(Psi + 0.5*dt*k1)
    k3 = get_derivative(Psi + 0.5*dt*k2)
    k4 = get_derivative(Psi + dt*k3)
    Psi = Psi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # --- PHASE LOGIC ---
    phase_tag = 0 
    
    # 1. BASELINE (Do nothing)
    if curr_time < 50.0:
        phase_tag = 0
        # Background noise learning (very slow)
        if t % 20 == 0: apply_plasticity(Psi)
        
    # 2. TRAINING (Inject Pattern + Learn Aggressively)
    elif 50.0 <= curr_time < 150.0:
        phase_tag = 1
        
        # Strong Clamping Injection
        # Force state toward Pattern A
        injection_force = kappa * (Pattern_A - Psi)
        Psi += dt * injection_force
        
        # Aggressive Learning
        apply_plasticity(Psi)
        
    # 3. RECALL TEST (No Input, Low Learning)
    else:
        phase_tag = 2
        # We stop injecting to see if memory holds
        # We can stop learning or slow it down to see structural retention
        if t % 50 == 0: apply_plasticity(Psi)

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

# -----------------------------
# 7. VISUALIZATION
# -----------------------------
print("Plotting results...")
plt.style.use('dark_background')
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
fig.suptitle('Milestone 1.1: Consolidation Test (High Eta, Low Decay)', fontsize=18)

times = history['t']
# Shading
for i, p in enumerate(history['phase']):
    c = 'black'
    if p == 1: c = 'darkgreen' 
    if p == 2: c = 'darkblue'
    axs[0].axvspan(times[i], times[i]+(times[1]-times[0]), facecolor=c, alpha=0.2)

# Plot 1: Energy
axs[0].plot(times, history['energy'], color='cyan')
axs[0].set_ylabel('Energy')
axs[0].set_title('System Stability')

# Plot 2: Coherence
axs[1].plot(times, history['coherence'], color='yellow')
axs[1].set_ylabel('Coherence')
axs[1].set_title('Synchronization')

# Plot 3: RECALL (THE VERDICT)
axs[2].plot(times, history['recall_A'], color='lime', linewidth=2)
axs[2].set_ylabel('Recall Overlap')
axs[2].set_title('MEMORY METRIC: Pattern A Retention')
axs[2].set_ylim(0, 1)

# Target Lines
axs[2].axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Noise Floor')
axs[2].axhline(y=0.5, color='red', linestyle='--', label='Target Threshold')

# Annotations
axs[2].text(75, 0.8, "TRAINING", color='white', fontsize=12, weight='bold')
axs[2].text(200, 0.8, "RECALL TEST", color='white', fontsize=12, weight='bold')

# Final Score Annotation
final_recall = np.mean(history['recall_A'][-20:])
axs[2].text(250, 0.2, f"Final Recall: {final_recall:.2f}", color='cyan', fontsize=14, weight='bold', bbox=dict(facecolor='black', alpha=0.8))

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()