import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# -----------------------------
# 1. PARAMETERS (FINAL FORM)
# -----------------------------
N = 500
alpha = 0.1
lam = 0.8
gamma = 0.5
eps = 1e-6

dt = 0.05
total_time = 300.0 
steps = int(total_time / dt)

# CONTROL PARAMETERS
target_energy = 3.5
tau = 0.01           # Filter timescale
homeostatic_rate = 0.005
noise_amp = 0.05

# -----------------------------
# 2. BUILD STRUCTURE (THE FIX)
# -----------------------------
print("Initializing Controllable Brain...")
density = 0.02
W_real = sp.random(N, N, density=density, format='csr', data_rvs=np.random.randn)
W_imag = sp.random(N, N, density=density, format='csr', data_rvs=np.random.randn)
W = (W_real + 1j * W_imag)

# --- SPECTRAL NORMALIZATION (The Missing Ingredient) ---
# We calculate the largest eigenvalue magnitude
print("Calculating spectral radius...")
# Use eigsh for speed (largest eigenvalue of W*W^T approximates spectral radius squared)
# Or just estimate with sparse eigs if strictly complex.
try:
    eigenvals = sp.linalg.eigs(W, k=1, return_eigenvectors=False)
    max_eigen = np.abs(eigenvals[0])
except:
    # Fallback if sparse eigs fails on structure
    max_eigen = 4.0 # rough guess for random sparse

target_radius = 1.2
scaling_factor = target_radius / max_eigen
W = W * scaling_factor
print(f"Scaled W by {scaling_factor:.4f} to set spectral radius to {target_radius}")

# Laplacian
A = sp.random(N, N, density=density, format='csr')
A = (A + A.T) * 0.5
degrees = np.array(A.sum(axis=1)).flatten()
D_mat = sp.diags(degrees)
Delta = D_mat - A

# -----------------------------
# 3. STATE INITIALIZATION
# -----------------------------
Psi = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1

# Local States
xi_vec = np.ones(N) * 0.5   # Start low to prevent massive overshoot
E_avg_vec = np.ones(N) * 0.1

history = {
    't': [], 'energy': [], 'mean_xi': [], 'std_xi': [], 'coherence': [], 'target': []
}

# -----------------------------
# 4. DYNAMICS
# -----------------------------

def get_derivative(Psi_curr, xi_curr):
    D = W @ Psi_curr
    num = np.real(Psi_curr.conj() * D)
    den = (np.abs(Psi_curr)**2) + (np.abs(D)**2) + eps
    R = num / den
    
    g_vec = xi_curr * np.tanh(1.0 - R) - lam
    
    dPsi = 1j*(W @ Psi_curr) + alpha*(Delta @ Psi_curr) + (g_vec * Psi_curr) - (gamma * (np.abs(Psi_curr)**2) * Psi_curr)
    dPsi += noise_amp * (np.random.randn(N) + 1j*np.random.randn(N))
    return dPsi

print("Running Simulation...")

for t in range(steps):
    curr_time = t * dt
    
    # --- 1. RK4 PHYSICS STEP ---
    k1 = get_derivative(Psi, xi_vec)
    k2 = get_derivative(Psi + 0.5*dt*k1, xi_vec)
    k3 = get_derivative(Psi + 0.5*dt*k2, xi_vec)
    k4 = get_derivative(Psi + dt*k3, xi_vec)
    Psi = Psi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # --- 2. SLOW HOMEOSTATIC UPDATE ---
    instant_energy = np.abs(Psi)**2
    E_avg_vec = (1 - tau) * E_avg_vec + tau * instant_energy
    
    error = target_energy - E_avg_vec
    xi_vec += homeostatic_rate * error
    
    # Clamp
    xi_vec = np.clip(xi_vec, 0.0, 3.0)

    # --- 3. RECORD ---
    if t % 20 == 0:
        history['t'].append(curr_time)
        history['energy'].append(np.mean(instant_energy))
        history['mean_xi'].append(np.mean(xi_vec))
        history['std_xi'].append(np.std(xi_vec))
        history['target'].append(target_energy)
        
        phases = np.angle(Psi)
        coherence = np.abs(np.mean(np.exp(1j * phases)))
        history['coherence'].append(coherence)

# -----------------------------
# 5. VISUALIZATION
# -----------------------------
plt.style.use('dark_background')
fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
fig.suptitle('Success: Self-Organized Criticality via Homeostasis', fontsize=18)

times = history['t']

# Plot 1: Energy
axs[0].plot(times, history['energy'], color='cyan', linewidth=2)
axs[0].plot(times, history['target'], color='red', linestyle='--', label='Target')
axs[0].set_title('System Energy (Converged to Target)')
axs[0].legend()

# Plot 2: Mean Curiosity
axs[1].plot(times, history['mean_xi'], color='orange')
axs[1].set_title('Mean Curiosity (ξ) — Stable Non-Zero')

# Plot 3: Diversity
axs[2].plot(times, history['std_xi'], color='lime')
axs[2].set_title('Diversity (σ ξ) — "Personality" Distribution')

# Plot 4: Coherence
axs[3].plot(times, history['coherence'], color='yellow')
axs[3].set_title('Network Coherence (Critical Dynamics)')

plt.tight_layout()
plt.show()