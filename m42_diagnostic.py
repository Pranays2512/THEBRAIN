"""
M42 DIAGNOSTIC
==============
Stop patching. Find the exact break point.

Strategy: Start from M40 (proven 0.31 Hz) and change ONE thing at a time.
Each test either passes or fails. First failure = root cause.

Test 0: M40 exact replica         → should get 0.31 Hz (baseline)
Test 1: M40 + wider oscillators   → does range change break anything?
Test 2: M40 + wider sweep         → does sweep range change break anything?
Test 3: M40 + both range changes  → combined effect
Test 4: M42 fast stream exact     → reproduce M42 failure
Test 5: Ridge alpha sweep         → find optimal regularization

If Test 0 fails → environment issue
If Test 1 fails → oscillator range is the problem
If Test 2 fails → sweep range is the problem
If Test 3 fails but 1,2 pass → interaction effect
If Test 4 matches Test 3 → we know exactly what to fix
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================
# SHARED PARAMETERS (M40 proven values)
# =============================================================
N   = 500
lam = 0.8
eps = 1e-6
dt  = 0.05
target_energy = 2.5
input_gain    = 1.5

S_local     = 0.15
sigma_local = 10.0

gamma_vec     = np.linspace(0.5, 3.0, N)
tau_adapt_vec = np.linspace(0.05, 0.5, N)
kappa_adapt   = 0.5;  adapt_max = 2.0
xi_min, xi_max = 0.1, 3.0
alpha_base, alpha_max = 0.1, 0.3
target_lyap = 0.1;  eta_alpha = 0.0005
lyap_window = 50

learning_end_time = 60.0
learn_interval    = 20
eta_hebb          = 0.002
decay_hebb        = 0.0001
noise_amp         = 0.05

stabilization_time      = 60.0
feature_sample_interval = 2
FAST_SECONDS            = 0.2
fast_steps              = int(FAST_SECONDS / dt)  # 4


# =============================================================
# BUILD NETWORK — parameterized by freq range
# =============================================================
def build_network(freq_min, freq_max):
    # Oscillator frequencies
    omega_hz_  = np.logspace(np.log10(freq_min), np.log10(freq_max), N)
    omega_vec_ = 2.0 * np.pi * omega_hz_

    idx = np.arange(N)
    ii, jj = np.meshgrid(idx, idx, indexing='ij')
    W_dense = np.exp(-(ii-jj).astype(float)**2 / (2.0*sigma_local**2))
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
        sl = slice(g*group_size, (g+1)*group_size)
        ph = (phases[g] if phases[g] is not None
              else np.random.uniform(0, 2*np.pi, group_size))
        base = (np.random.randn(group_size) + 1j*np.random.randn(group_size)) * 0.5
        W_in[sl] = base * gains[g] * np.exp(1j*ph)

    A_sym   = (W_local + W_local.T) * 0.5
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    Delta   = sp.diags(degrees) - A_sym

    return W_local, W_in, Delta, omega_hz_, omega_vec_


# =============================================================
# DERIVATIVE
# =============================================================
def get_derivative(Psi, xi_vec, adapt, alpha, noise, I_in,
                   W, W_in, Delta, omega_vec_):
    W_eff = S_local * W
    D     = W_eff @ Psi
    num   = np.real(Psi.conj() * D)
    den   = np.abs(Psi)**2 + np.abs(D)**2 + eps
    R     = num / den
    g_vec = xi_vec * np.tanh(1.0 - R) - lam
    eff_gamma = gamma_vec + adapt
    dPsi = (1j*omega_vec_*Psi
            + W_eff@Psi
            + alpha*(Delta@Psi)
            + g_vec*Psi
            - eff_gamma*(np.abs(Psi)**2)*Psi)
    dPsi += noise_amp*noise + W_in*I_in*input_gain
    return dPsi


# =============================================================
# FEATURES
# =============================================================
def energy_entropy(energy_series):
    E      = energy_series - energy_series.min(axis=0, keepdims=True) + eps
    E_norm = E / (E.sum(axis=0, keepdims=True) + eps)
    H      = -np.sum(E_norm * np.log(E_norm + eps), axis=0)
    return H / np.log(max(energy_series.shape[0], 2) + eps)

def extract_features_fast(psi_buf, phi_buf):
    phi_neuron = np.angle(psi_buf)
    delta_phi  = np.angle(np.exp(1j*(phi_neuron - phi_buf)))
    plv = np.abs(np.mean(np.exp(1j*delta_phi), axis=0))
    energy = np.abs(psi_buf)**2
    ent    = energy_entropy(energy)
    ec  = energy - energy.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(ec, axis=0)
    pwr = np.abs(fft)**2
    fq  = np.fft.rfftfreq(fast_steps, d=dt)
    bands = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)]
    spec = np.concatenate([
        np.mean(pwr[(fq >= lo) & (fq <= hi)], axis=0)
        if np.any((fq >= lo) & (fq <= hi)) else np.zeros(N)
        for lo, hi in bands
    ])
    return plv, ent, spec


# =============================================================
# SIMULATION — fast stream only, parameterized
# =============================================================
def run_fast_stream(osc_min, osc_max, sweep_min, sweep_max,
                    n_sweeps=6, sweep_dur=60.0, seed=0,
                    ridge_alpha=1000.0, n_pca=50, verbose=False):
    """
    Run full fast-stream pipeline with given parameters.
    Returns train_mae, test_mae
    """
    warmup     = stabilization_time + 10.0
    total_time = warmup + n_sweeps * sweep_dur + 10.0

    W_local, W_in, Delta, omega_hz_, omega_vec_ = build_network(osc_min, osc_max)

    def make_sweep(f_start, f_end):
        def sig(t):
            if t < warmup:
                f = (f_start + f_end) / 2.0
            else:
                elapsed   = t - warmup
                sweep_idx = int(elapsed / sweep_dur)
                frac      = (elapsed % sweep_dur) / sweep_dur
                f = (f_start + (f_end-f_start)*frac if sweep_idx % 2 == 0
                     else f_end - (f_end-f_start)*frac)
            return np.sin(2*np.pi*f*t), f, f
        return sig

    def run_sim(signal_func, rseed):
        np.random.seed(rseed)
        steps = int(total_time / dt)

        Psi          = (np.random.randn(N) + 1j*np.random.randn(N)) * 0.1
        xi_vec       = np.ones(N) * 0.5
        A_vec        = np.zeros(N)
        E_avg_vec    = np.ones(N) * 0.1
        alpha_global = alpha_base
        Psi_ghost    = Psi + (np.random.randn(N)+1j*np.random.randn(N))*1e-5
        prev_dist    = np.linalg.norm(Psi_ghost - Psi)
        Lyap_hist    = []
        xi_frozen    = False; xi_frozen_val = None

        psi_buf = np.zeros((fast_steps, N), dtype=complex)
        phi_buf = np.zeros((fast_steps, 1))
        filled  = False

        plv_all=[]; ent_all=[]; spec_all=[]; Y_all=[]
        Wc = W_local.tocsr()
        Wl = W_local.copy()

        for t in range(steps):
            ct        = t * dt
            noise_vec = (np.random.randn(N) + 1j*np.random.randn(N))
            I_val, Y_val, freq = signal_func(ct)
            phi_in = (2*np.pi*freq*ct) % (2*np.pi) if freq > 0 else 0.0

            k1 = get_derivative(Psi, xi_vec, A_vec, alpha_global, noise_vec,
                                 I_val, Wc, W_in, Delta, omega_vec_)
            k2 = get_derivative(Psi+0.5*dt*k1, xi_vec, A_vec, alpha_global,
                                 noise_vec, I_val, Wc, W_in, Delta, omega_vec_)
            k3 = get_derivative(Psi+0.5*dt*k2, xi_vec, A_vec, alpha_global,
                                 noise_vec, I_val, Wc, W_in, Delta, omega_vec_)
            k4 = get_derivative(Psi+dt*k3, xi_vec, A_vec, alpha_global,
                                 noise_vec, I_val, Wc, W_in, Delta, omega_vec_)
            Psi = Psi + (dt/6.0)*(k1+2*k2+2*k3+k4)

            k1g = get_derivative(Psi_ghost, xi_vec, A_vec, alpha_global,
                                  noise_vec, 0, Wc, W_in, Delta, omega_vec_)
            Psi_ghost = Psi_ghost + dt*k1g

            instant_energy = np.abs(Psi)**2
            E_avg_vec = 0.99*E_avg_vec + 0.01*instant_energy

            if ct >= stabilization_time and not xi_frozen:
                xi_frozen = True; xi_frozen_val = xi_vec.copy()

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
                rows, cols = Wl.nonzero()
                corr   = Psi[rows] * np.conj(Psi[cols])
                update = np.real(eta_hebb * corr * np.abs(Psi[rows]) * np.abs(Psi[cols]))
                current_w = np.asarray(Wl[rows, cols]).flatten()
                new_w     = np.abs(current_w + update - decay_hebb*current_w)
                Wl        = Wl.tolil()
                Wl[rows, cols] = new_w
                Wl = Wl.tocsr()
                try:
                    ev = sp.linalg.eigs(Wl, k=1, return_eigenvectors=False)
                    if np.abs(ev[0]) > 0: Wl = Wl * (0.9/np.abs(ev[0]))
                except: pass
                Wc = Wl.tocsr()

            fast_idx = t % fast_steps
            psi_buf[fast_idx] = Psi.copy()
            phi_buf[fast_idx] = phi_in
            if t >= fast_steps: filled = True

            if ct > stabilization_time and filled and (t % feature_sample_interval == 0):
                f_psi = np.roll(psi_buf, -fast_idx-1, axis=0)
                f_phi = np.roll(phi_buf, -fast_idx-1, axis=0)
                fp, fe, fs = extract_features_fast(f_psi, f_phi)
                plv_all.append(fp); ent_all.append(fe); spec_all.append(fs)
                Y_all.append(Y_val)

        return (np.array(plv_all), np.array(ent_all),
                np.array(spec_all), np.array(Y_all))

    # Train
    sig_train = make_sweep(sweep_min, sweep_max)
    plv, ent, spec, Y = run_sim(sig_train, seed)
    X = np.hstack([plv, ent, spec])
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    n  = min(n_pca, X_sc.shape[0]-1, X_sc.shape[1])
    pc = PCA(n_components=n); X_p = pc.fit_transform(X_sc)
    md = Ridge(alpha=ridge_alpha); md.fit(X_p, Y)
    train_pred = md.predict(X_p)
    train_mae  = np.mean(np.abs(train_pred - Y))

    # Test (fresh sim, same sweep range)
    sig_test = make_sweep(sweep_min, sweep_max)
    plv_te, ent_te, spec_te, Y_te = run_sim(sig_test, seed+10)
    X_te  = np.hstack([plv_te, ent_te, spec_te])
    X_tep = pc.transform(sc.transform(X_te))
    test_pred = md.predict(X_tep)
    test_mae  = np.mean(np.abs(test_pred - Y_te))

    return train_mae, test_mae


# =============================================================
# MAIN DIAGNOSTIC
# =============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  M42 DIAGNOSTIC — Find the exact break point")
    print("  Changing ONE variable at a time from M40 baseline")
    print("=" * 70)

    results = {}

    # ----------------------------------------------------------
    # TEST 0: M40 EXACT REPLICA
    # Must get ~0.31 Hz. If not, environment issue.
    # ----------------------------------------------------------
    print(f"\n  Test 0: M40 exact replica")
    print(f"  osc=0.5–2.0, sweep=0.5–2.0, ridge=1000, pca=50")
    tr, te = run_fast_stream(0.5, 2.0, 0.5, 2.0,
                              ridge_alpha=1000, n_pca=50, seed=0)
    results['T0_M40_exact'] = te
    status = "✓" if te < 0.33 else "✗ BROKEN"
    print(f"  Train MAE: {tr:.4f} Hz  |  Test MAE: {te:.4f} Hz  {status}")
    print(f"  M40 reference: 0.3108 Hz")

    # ----------------------------------------------------------
    # TEST 1: WIDER OSCILLATORS ONLY
    # Change only osc range, keep sweep same
    # ----------------------------------------------------------
    print(f"\n  Test 1: Wider oscillators only")
    print(f"  osc=0.4–2.1, sweep=0.5–2.0, ridge=1000, pca=50")
    tr, te = run_fast_stream(0.4, 2.1, 0.5, 2.0,
                              ridge_alpha=1000, n_pca=50, seed=0)
    results['T1_wider_osc'] = te
    delta = te - results['T0_M40_exact']
    status = "✓ OK" if abs(delta) < 0.03 else f"✗ BROKE (+{delta:.4f} Hz)"
    print(f"  Train MAE: {tr:.4f} Hz  |  Test MAE: {te:.4f} Hz  {status}")

    # ----------------------------------------------------------
    # TEST 2: WIDER SWEEP ONLY
    # Change only sweep range, keep osc same as M40
    # ----------------------------------------------------------
    print(f"\n  Test 2: Wider sweep range only")
    print(f"  osc=0.5–2.0, sweep=0.4–2.1, ridge=1000, pca=50")
    tr, te = run_fast_stream(0.5, 2.0, 0.4, 2.1,
                              ridge_alpha=1000, n_pca=50, seed=0)
    results['T2_wider_sweep'] = te
    delta = te - results['T0_M40_exact']
    status = "✓ OK" if abs(delta) < 0.03 else f"✗ BROKE (+{delta:.4f} Hz)"
    print(f"  Train MAE: {tr:.4f} Hz  |  Test MAE: {te:.4f} Hz  {status}")

    # ----------------------------------------------------------
    # TEST 3: BOTH WIDER (M42 current state)
    # ----------------------------------------------------------
    print(f"\n  Test 3: Both wider (M42 current state)")
    print(f"  osc=0.4–2.1, sweep=0.4–2.1, ridge=1000, pca=50")
    tr, te = run_fast_stream(0.4, 2.1, 0.4, 2.1,
                              ridge_alpha=1000, n_pca=50, seed=0)
    results['T3_both_wider'] = te
    delta = te - results['T0_M40_exact']
    status = "✓ OK" if abs(delta) < 0.03 else f"✗ BROKE (+{delta:.4f} Hz)"
    print(f"  Train MAE: {tr:.4f} Hz  |  Test MAE: {te:.4f} Hz  {status}")

    # ----------------------------------------------------------
    # TEST 4: RIDGE ALPHA SWEEP
    # Find optimal regularization for fast stream
    # ----------------------------------------------------------
    print(f"\n  Test 4: Ridge alpha sweep (osc=0.4–2.1, sweep=0.4–2.1)")
    print(f"  {'Alpha':>8}  {'Train MAE':>10}  {'Test MAE':>10}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}")
    best_alpha = 1000; best_te = 999
    for alpha in [10, 50, 100, 200, 500, 1000, 2000]:
        tr, te = run_fast_stream(0.4, 2.1, 0.4, 2.1,
                                  ridge_alpha=alpha, n_pca=50, seed=0)
        marker = " ← M40 used" if alpha == 1000 else ""
        better = " ← BETTER" if te < results['T0_M40_exact'] else ""
        print(f"  {alpha:>8}  {tr:10.4f}  {te:10.4f}{marker}{better}")
        if te < best_te:
            best_te = te; best_alpha = alpha
    results['best_alpha'] = best_alpha
    results['best_alpha_mae'] = best_te

    # ----------------------------------------------------------
    # TEST 5: PCA COMPONENTS SWEEP
    # Find optimal compression
    # ----------------------------------------------------------
    print(f"\n  Test 5: PCA components sweep (best alpha={best_alpha})")
    print(f"  {'n_pca':>6}  {'Train MAE':>10}  {'Test MAE':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
    best_pca = 50; best_pca_te = 999
    for n_pca in [30, 50, 80, 100, 120]:
        tr, te = run_fast_stream(0.4, 2.1, 0.4, 2.1,
                                  ridge_alpha=best_alpha, n_pca=n_pca, seed=0)
        better = " ← BETTER" if te < results['T0_M40_exact'] else ""
        print(f"  {n_pca:>6}  {tr:10.4f}  {te:10.4f}{better}")
        if te < best_pca_te:
            best_pca_te = te; best_pca = n_pca
    results['best_pca'] = best_pca
    results['best_pca_mae'] = best_pca_te

    # ----------------------------------------------------------
    # TEST 6: BEST CONFIG CONFIRMED
    # ----------------------------------------------------------
    print(f"\n  Test 6: Best config confirmed (3 seeds for stability)")
    print(f"  osc=0.4–2.1, sweep=0.4–2.1, alpha={best_alpha}, pca={best_pca}")
    maes = []
    for seed in [0, 1, 2]:
        tr, te = run_fast_stream(0.4, 2.1, 0.4, 2.1,
                                  ridge_alpha=best_alpha,
                                  n_pca=best_pca, seed=seed)
        maes.append(te)
        print(f"  seed={seed}: train={tr:.4f}  test={te:.4f}")
    print(f"  Mean test MAE: {np.mean(maes):.4f} Hz  ±{np.std(maes):.4f}")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Test':35s}  {'MAE':>8}  {'vs M40':>8}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}")
    ref = results['T0_M40_exact']
    for k, v in results.items():
        if k.startswith('T') and isinstance(v, float):
            delta = v - ref
            sign  = '+' if delta >= 0 else ''
            print(f"  {k:35s}  {v:8.4f}  {sign}{delta:.4f}")
    print(f"\n  Best alpha:    {best_alpha}")
    print(f"  Best PCA:      {best_pca}")
    print(f"  Best test MAE: {best_pca_te:.4f} Hz")
    print(f"  M40 reference: {ref:.4f} Hz")
    if best_pca_te < ref:
        print(f"\n  ✓ FOUND BETTER CONFIG: beats M40 by {ref-best_pca_te:.4f} Hz")
    elif best_pca_te < ref + 0.02:
        print(f"\n  ✓ MATCHES M40: within 0.02 Hz — use this config for M42")
    else:
        print(f"\n  ✗ GAP REMAINS: {best_pca_te - ref:.4f} Hz worse than M40")
        print(f"    → The issue is in the dynamics, not the readout")
        print(f"    → Check: Hebbian learning interaction with wider osc range")