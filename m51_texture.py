"""
M51: TEXTURE SENSE — SECOND SENSORY MODALITY
============================================

WHAT THIS IS
------------
M51 is the brain's second sense. Where M50 hears frequency, M51 feels
texture — a static ambient property of each location that is INDEPENDENT
of sound frequency.

In the simulated world (World5), texture is a fixed scalar value per node
assigned so that every aliased pair (nodes sharing the same frequency) has
maximally separated textures:

  A=0.12 vs I=0.88  (same sound 0.5Hz — texture separates them by 0.76)
  B=0.27 vs J=0.73  (same sound 0.7Hz — separation 0.46)
  ...etc.

In a real robot/device deployment, texture would come from:
  - A camera (image features → texture descriptor)
  - A touch sensor (surface resistance → scalar)
  - A light sensor (ambient brightness → scalar)
  - Any sensor whose output is INDEPENDENT of frequency

M51 doesn't change regardless of which physical sensor provides the input.
That's the point: the modality architecture is sensor-agnostic.

WHAT M51 DOES
-------------
M51 takes the raw texture value (a scalar in [0, 1]) and produces a
rich input vector for M54b (the texture cortex). It:

  1. Repeats the texture value N_TEX_REPEAT times — same trick as M54's
     freq_norm×8, ensures the texture signal dominates the input so the
     SOM separates textures cleanly.

  2. Adds spatial context: the texture value modulated at 3 spatial
     frequencies. This gives the cortex richer texture geometry to learn
     from, analogous to how M50's PLV vector gives M54 spectral context.

  3. Adds a noise signal: simulates real sensor noise. Amount controlled
     by TEX_NOISE_STD. This is what makes the system robust — it learns
     to identify textures despite noise, not from noiseless signals.

  4. Clips everything to [0, 1] — safe for the SOM.

OUTPUT
------
Returns a dict with:
  texture_vec  : (INPUT_DIM_TEX,) float32 — input vector for M54b
  texture_val  : float — raw texture value (for logging)
  texture_noisy: float — noisy version (what M54b actually sees)

PARAMETERS
----------
INPUT_DIM_TEX   = 16  — 8 repeats + 6 spatial + 2 scalars
                       Smaller than M54's 30 because texture has less
                       natural structure to exploit. 16 is enough for
                       the SOM to form 8 distinct clusters (one per
                       aliased pair) with room for noise tolerance.

N_TEX_REPEAT    = 8   — number of times texture_norm is repeated.
                       Gives texture 50% of the input signal weight.

TEX_NOISE_STD   = 0.04 — Gaussian noise std (σ) added to texture.
                       4% noise. Calibrated so aliased pairs (min
                       separation 0.38) are still clearly separated
                       after noise: z = 0.38 / (0.04√2) ≈ 6.7σ.
                       Essentially zero confusion between pairs.
                       Biologically: sensor noise in mechanoreceptors /
                       photoreceptors is typically 2-5% of signal range.

BIOLOGICAL GROUNDING
--------------------
M51 models somatosensory cortex layer 4 (S1-L4) afferent input from
mechanoreceptors (Merkel discs for texture, Meissner corpuscles for
fine spatial detail). The repeated texture value maps to the dense
topographic representation in barrel cortex — the same stimulus
encoded by multiple redundant columns for noise robustness.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

N_TEX_REPEAT   = 8      # repeats of raw texture value (dominant signal)
N_SPATIAL      = 6      # spatial modulation components
N_SCALARS_TEX  = 2      # [texture_norm, 1-texture_norm] — explicit complement
INPUT_DIM_TEX  = N_TEX_REPEAT + N_SPATIAL + N_SCALARS_TEX   # = 16

TEX_NOISE_STD  = 0.04   # Gaussian noise σ on raw texture value
SPATIAL_FREQS  = np.array([1.0, 2.0, 4.0])   # 3 spatial modulation frequencies


# ═══════════════════════════════════════════════════════════════
# TEXTURE SENSE MODULE
# ═══════════════════════════════════════════════════════════════

class TextureSense:
    """
    M51 — second sensory modality (texture/spatial).

    Converts a scalar texture value into a rich vector for M54b.
    Drop-in parallel to M50: M50 → decoded_freq → M54,
                             M51 → texture_val  → M54b.

    Parameters
    ----------
    seed : int — random seed for reproducible noise
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self.t    = 0

        # Running stats for diagnostics
        self._last_texture_val   = 0.0
        self._last_texture_noisy = 0.0

    # ── Main step ─────────────────────────────────────────────

    def step(self, texture_val: float) -> dict:
        """
        Convert a raw texture value into the M54b input vector.

        Parameters
        ----------
        texture_val : float — raw texture in [0, 1] from the world.
                              World5 provides this via info['texture'].
                              Future real sensors plug in here.

        Returns
        -------
        dict with keys:
          texture_vec   : (INPUT_DIM_TEX,) float32 — input for M54b
          texture_val   : float — raw (noiseless) texture value
          texture_noisy : float — noisy version actually encoded
        """
        tex = float(np.clip(texture_val, 0.0, 1.0))

        # ── 1. Add sensor noise ────────────────────────────────
        noise = float(self._rng.normal(0.0, TEX_NOISE_STD))
        tex_noisy = float(np.clip(tex + noise, 0.0, 1.0))

        # ── 2. Build input vector ──────────────────────────────
        # Part A: repeated texture value (dominant signal, 50% of dims)
        tex_repeated = np.full(N_TEX_REPEAT, tex_noisy, dtype=np.float64)

        # Part B: spatial modulation components
        # sin(2π × f × tex) for f in SPATIAL_FREQS
        # cos(2π × f × tex) for f in SPATIAL_FREQS
        # These give the cortex "spatial harmonics" — texture at
        # multiple scales, making the SOM topology richer.
        spatial = np.zeros(N_SPATIAL, dtype=np.float64)
        for i, f in enumerate(SPATIAL_FREQS):
            phase = 2.0 * np.pi * f * tex_noisy
            spatial[2*i]     = (np.sin(phase) + 1.0) * 0.5   # normalise to [0,1]
            spatial[2*i + 1] = (np.cos(phase) + 1.0) * 0.5

        # Part C: explicit scalar pair [tex, 1-tex]
        # Gives the SOM a direct complementary signal so it doesn't
        # need to discover the inversion from spatial harmonics alone.
        scalars = np.array([tex_noisy, 1.0 - tex_noisy], dtype=np.float64)

        # ── 3. Concatenate and clip ────────────────────────────
        vec = np.concatenate([tex_repeated, spatial, scalars]).astype(np.float32)
        vec = np.clip(vec, 0.0, 1.0)

        # Store for diagnostics
        self._last_texture_val   = tex
        self._last_texture_noisy = tex_noisy
        self.t += 1

        return {
            'texture_vec':    vec,
            'texture_val':    tex,
            'texture_noisy':  tex_noisy,
        }

    # ── Diagnostics ───────────────────────────────────────────

    def summary(self):
        print(f"  TextureSense (M51) — step {self.t}")
        print(f"  Last texture:  {self._last_texture_val:.3f} "
              f"(noisy: {self._last_texture_noisy:.3f})")
        print(f"  Input dim:     {INPUT_DIM_TEX}  "
              f"(repeats={N_TEX_REPEAT}, spatial={N_SPATIAL}, scalars={N_SCALARS_TEX})")
        print(f"  Noise σ:       {TEX_NOISE_STD}  "
              f"(min pair separation: 0.38 → z={0.38/(TEX_NOISE_STD*1.414):.1f}σ)")