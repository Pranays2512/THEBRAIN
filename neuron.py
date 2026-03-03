import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# -----------------------------
# 1. PARAMETERS (ATTRACTOR TEST)
# -----------------------------
N = 500
alpha = 0.1
xi = 1.2
lam = 0.8
gamma = 0.5
eps = 1e-6

dt = 0.05
total_time = 150.0 
steps = int(total_time / dt)

# No Learning
eta = 0.0           
decay_rate = 0.0
beta = 1.5          # Strength of the implanted memory

# -----------------------------
# 2. BUILD STRUCTURE
# -----------------------------
print("Building network...")
density = 0.02
W_real = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn) # LIL for construction
W_imag = sp.random(N, N, density=density, format='lil', data_rvs=np.random.randn)
W = (W_real + 1j * W_imag) / np.sqrt(N * density)

# -----------------------------
# 3. DEFINE THE "MEMORY"
# -----------------------------
Pattern_A = np.zeros(N, dtype=np.complex128)
# Create a specific sparse pattern
idx_active = np.concatenate([np.arange(0, 50), np.arange(100, 150)])
Pattern_A[idx_active] = 1.0 
# Give it a random phase structure
Pattern_A = Pattern_A * np.exp(1j * np.random.uniform(0, 2*np.pi, N))
Pattern_A = Pattern_A / np.linalg.norm(Pattern_A)

# -----------------------------
# 4. BURN-IN MEMORY (Hopfield Rule)
# -----------------------------
print("Burning memory into W...")

# We want to add: beta * (Pattern_A_i * conj(Pattern_A_j))
# We need to add this to EXISTING connections.
# But if connections don't exist, we must create them for the pattern to hold.

rows, cols = W.nonzero()
# Update existing weights
# W_ij += beta * P_i * conj(P_j)
update = beta * (Pattern_A[rows] * np.conj(Pattern_A[cols]))
W[rows, cols] += update

# Ensure the pattern's self-support connections exist (Critical for attractor)
# For the active indices, ensure they connect to each other
active_indices = idx_active
for i in active_indices:
    # Connect to a subset of other active nodes
    targets = np.random.choice(active_indices, 10) 
    W[i, targets] += beta * (Pattern_A[i] * np.conj(Pattern_A[targets]))

# Convert to CSR for fast math
W = W.tocsr()

# -----------------------------
# 5. DYNAMICS
# -----------------------------
def get_derivative(Psi_curr):
    D = W @ Psi_curr
    num = np.real(Psi_curr.conj() * D)
    den = (np.abs(Psi_curr)**2) + (np.abs(D)**2) + eps
    R = num / den
    g = xi * np.tanh(1.0 - R) - lam
    
    # Standard Dynamics
    dPsi = 1j*(W @ Psi_curr) + alpha*(Delta @ Psi_curr) + (g * Psi_curr) - (gamma * (np.abs(Psi_curr)**2) * Psi_curr)
    return dPsi

# Laplacian (needs to be recalculated if W changed significantly, but we keep original topology for now)
A = sp.random(N, N, density=density, format='csr')
A = (A + A.T) * 0.5
degrees = np.array(A.sum(axis=1)).flatten()
D = sp.diags(degrees)
Delta = D - A

# -----------------------------
# 6. RUN EXPERIMENT
# -----------------------------
print("Running Attractor Test...")

# START STATE: 80% Noise, 20% Pattern (Weak Hint)
Psi = 0.2 * Pattern_A + 0.8 * (np.random.randn(N) + 1j * np.random.randn(N))
Psi = Psi / np.linalg.norm(Psi) * 2.0 # Normalize to reasonable energy

history = {'t': [], 'recall': [], 'energy': [], 'phase_diff': []}

for t in range(steps):
    # RK4
    k1 = get_derivative(Psi)
    k2 = get_derivative(Psi + 0.5*dt*k1)
    k3 = get_derivative(Psi + 0.5*dt*k2)
    k4 = get_derivative(Psi + dt*k3)
    Psi = Psi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Record
    if t % 10 == 0:
        history['t'].append(t*dt)
        history['energy'].append(np.mean(np.abs(Psi)**2))
        
        # Recall Metric (Magnitude of projection)
        # We take absolute value to handle the oscillation (Limit Cycle)
        overlap = np.abs(np.vdot(Pattern_A, Psi)) / (np.linalg.norm(Pattern_A) * np.linalg.norm(Psi) + 1e-9)
        history['recall'].append(overlap)
        
        # Phase diff (are we locking?)
        # Not strictly necessary but interesting

# -----------------------------
# 7. VISUALIZATION
# -----------------------------
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Milestone 2: Explicit Attractor Test (Hopfield Burn-In)', fontsize=16)

axs[0].plot(history['t'], history['energy'], color='cyan')
axs[0].set_title('System Energy')
axs[0].set_ylabel('Energy')

axs[1].plot(history['t'], history['recall'], color='lime', linewidth=2)
axs[1].set_title('Memory Recall (Attraction to Pattern A)')
axs[1].set_ylabel('Overlap |<Pattern|Psi>|')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylim(0, 1)

# Success Threshold
axs[1].axhline(y=0.6, color='red', linestyle='--', label='Attraction Threshold')
axs[1].legend()

# Annotations
avg_recall = np.mean(history['recall'][-20:])
status = "SUCCESS: Attractor Formed" if avg_recall > 0.5 else "FAILURE: Pattern Unstable"
axs[1].text(0.5, 0.9, status, transform=axs[1].transAxes, fontsize=14, weight='bold', color='white')

plt.tight_layout()
plt.show()