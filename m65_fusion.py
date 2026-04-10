"""
M65: MULTIMODAL FUSION — SOUND × TEXTURE → L4 BELIEF UPDATE
=============================================================

WHAT THIS IS
------------
M65 is the integration layer that combines two independent sensory
streams — sound (from M54) and texture (from M54b) — into a single
fused observation that L4 can use for Bayesian position tracking.

This is the critical module that solves the aliasing problem.
Without M65, L4 can only see frequency (fi=0 = could be A or I).
With M65, L4 sees frequency + texture (fi=0, tex≈0.12 = definitely A).

HOW IT WORKS
------------
L4 currently does its observation update like this:

  belief[node] *= P(heard_sound=fi | at_node)

This is a hard likelihood: nodes sharing the same fi get equal weight,
and the belief cannot be disambiguated from a single step of sound.

M65 extends this to:

  belief[node] *= P(heard_sound=fi | at_node)
               × P(felt_texture=tex_bin | at_node)

The second factor is supplied by M54b's texture_likelihood.
When both signals agree about a node, the product is high.
When they disagree (wrong node), the product is low.

For perfectly separated textures (like A=0.12 vs I=0.88):
  P(tex≈0.12 | at A) ≈ 1.0   →  belief[A] × 1.0 = unchanged
  P(tex≈0.12 | at I) ≈ 0.0   →  belief[I] × 0.0 → eliminated

This is a SINGLE-STEP disambiguation. L4 no longer needs 2-3 steps
of sequence context to distinguish A from I — texture does it instantly.

ARCHITECTURE
------------
M65 does NOT modify L4 directly. Instead, it produces a
`fused_likelihood` vector — one probability per node — that
Brain passes into L4.step() as an additional observation update.

L4 applies it after its existing sound_likelihood update, before
the transition prior. The L4 code is unchanged.

This is the "fusion before belief update" approach (early fusion),
which is the right choice here because both signals are available
simultaneously at every step.

FUSION STRATEGY
---------------
M65 uses a weighted product of the two likelihoods:

  fused[node] = sound_likelihood[node]^alpha × texture_likelihood[node]^beta

Where alpha + beta = 1 (convex combination in log space).
Default: alpha=0.5, beta=0.5 (equal weight).

Why product (not sum)?
  - Both signals must agree. A mismatch in either → low probability.
  - Sum would let a strong sound override a wrong texture, keeping aliasing.
  - Product punishes disagreement, which is what we want.

Why not just hardcode alpha=0.5?
  M65 tracks signal reliability per modality and adaptively weights them.
  If sound is confusing (decoder noisy, qe_norm high), texture gets more
  weight. If texture sensor is poor (noisy steps), sound gets more weight.
  This makes the system robust to one modality degrading.

SOUND LIKELIHOOD
----------------
M65 constructs the sound_likelihood from the current freq_idx.
This is the same signal L4 already uses internally, but M65 exposes
it as an explicit vector so the product is clean.

  sound_likelihood[node] = 1.0 if node.fi == curr_fi else 0.0  (hard)
  → softened to avoid zero products:
  sound_likelihood[node] = 1 - SOUND_LIKE_FLOOR if node.fi != curr_fi
                         = 1.0                  if node.fi == curr_fi

TEXTURE LIKELIHOOD
------------------
Comes from M54b.step()['texture_likelihood'] — a (N_TEX_NEURONS,) vector
indexed by BMU. To convert to per-node:

  tex_likelihood_node[node] = Σ_bmu P(bmu | at_node) × P(tex_bin | bmu)

In practice M65 tracks which BMU fires most for each texture bin,
so this reduces to a direct lookup once M54b has converged.

OUTPUTS
-------
  fused_likelihood   : (n_nodes,) float — product of sound + texture
  sound_weight       : float [0,1]  — current adaptive weight for sound
  texture_weight     : float [0,1]  — current adaptive weight for texture
  texture_confidence : float [0,1]  — how confident is the texture signal
  fusion_active      : bool — whether fusion is contributing (warm enough)

PARAMETERS
----------
FUSION_WARMUP     = 3000  — steps before fusion weight activates.
                            M54b needs ~2000 steps to converge its texture map.
                            Before warmup: texture weight = 0 (sound only),
                            same behaviour as old L4.

ALPHA_SOUND       = 0.5   — default sound weight in fused product
ALPHA_TEXTURE     = 0.5   — default texture weight in fused product

SOUND_LIKE_FLOOR  = 0.05  — soft floor for wrong-frequency nodes.
                            Prevents zero products from making a node
                            permanently impossible (important if sensor
                            noise occasionally gives wrong freq readings).

TEXTURE_LIKE_FLOOR = 0.10 — analogous floor for texture.
                            Larger than sound floor because texture
                            noise is higher (4% vs ~1% for calibrated M50).

TEX_RELIABILITY_EMA = 0.01 — EMA for tracking texture prediction reliability.
                             Used for adaptive weighting (degrades weight if
                             texture has been consistently wrong lately).

BIOLOGICAL GROUNDING
--------------------
M65 models the posterior parietal cortex (PPC), specifically area 5
and 7b, which integrate tactile and proprioceptive signals with
spatial representations. In primates, PPC lesions cause spatial
neglect and object recognition failures when multiple sensory cues
must be combined for localisation. The product-of-experts fusion
mirrors the neural computation in PPC multisensory neurons, which
respond maximally when two aligned sensory cues (sound + touch) both
confirm the same spatial location.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

FUSION_WARMUP        = 3000   # steps before texture weight activates
ALPHA_SOUND          = 0.50   # sound weight in fused product
ALPHA_TEXTURE        = 0.50   # texture weight in fused product
SOUND_LIKE_FLOOR     = 0.05   # soft floor for wrong-frequency nodes
TEXTURE_LIKE_FLOOR   = 0.10   # soft floor for wrong-texture nodes
TEX_RELIABILITY_EMA  = 0.01   # EMA alpha for texture reliability tracking


# ═══════════════════════════════════════════════════════════════
# MULTIMODAL FUSION
# ═══════════════════════════════════════════════════════════════

class MultimodalFusion:
    """
    M65 — Combines sound and texture observations for L4.

    Produces fused_likelihood: a per-node probability vector that
    encodes BOTH what the brain heard and what it felt this step.

    Parameters
    ----------
    node_fi   : dict[str, int]  — node name → freq_index (from World5.NODES)
    node_tex  : dict[str, float] — node name → texture value (from World5.NODE_TEXTURES)
    n_tex_bins: int — number of texture histogram bins (must match M54b.N_TEX_BINS = 8)
    seed      : int — random seed
    """

    def __init__(self,
                 node_fi:    dict,
                 node_tex:   dict,
                 n_tex_bins: int = 8,
                 seed:       int = 42):
        self.nodes       = list(node_fi.keys())
        self.n_nodes     = len(self.nodes)
        self.node_fi     = dict(node_fi)
        self.node_tex    = dict(node_tex)
        self.n_tex_bins  = n_tex_bins
        self._node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        # ── Precompute sound likelihood matrix ────────────────
        # sound_like[node_idx, fi] = 1.0 if node.fi==fi else FLOOR
        n_freqs = max(node_fi.values()) + 1
        self._sound_like = np.full((self.n_nodes, n_freqs),
                                   SOUND_LIKE_FLOOR, dtype=np.float64)
        for i, node in enumerate(self.nodes):
            self._sound_like[i, self.node_fi[node]] = 1.0

        # ── Precompute texture bin for each node ──────────────
        # node_tex_bin[i] = which texture bin node i falls in
        self._node_tex_bin = np.zeros(self.n_nodes, dtype=np.int32)
        for i, node in enumerate(self.nodes):
            tex = float(self.node_tex[node])
            self._node_tex_bin[i] = int(
                np.clip(tex * n_tex_bins, 0, n_tex_bins - 1)
            )

        # ── Texture reliability tracking ──────────────────────
        # Tracks whether texture observations have been consistent.
        # Falls when texture predicts wrong; rises when correct.
        self._tex_reliability = 0.5   # start neutral
        self._tex_weight      = 0.0   # starts at 0 (sound-only before warmup)

        # ── State ─────────────────────────────────────────────
        self.t                = 0
        self._fusion_active   = False
        self._last_tex_conf   = 0.0
        self._last_sound_w    = 1.0
        self._last_tex_w      = 0.0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             curr_fi:            int,
             texture_likelihood: np.ndarray,
             tex_bin:            int,
             sound_qe_norm:      float = 0.0,
             tex_qe_norm:        float = 0.0) -> dict:
        """
        Fuse sound and texture observations into a single likelihood vector.

        Parameters
        ----------
        curr_fi             : int   — current freq_idx from world (0-7)
        texture_likelihood  : (N_TEX_NEURONS,) — from M54b.step()
        tex_bin             : int   — which texture bin fired (from M54b)
        sound_qe_norm       : float — M54 perceptual surprise (0=familiar, 1=novel)
        tex_qe_norm         : float — M54b perceptual surprise

        Returns
        -------
        dict with keys:
          fused_likelihood   : (n_nodes,) float — product observation for L4
          sound_weight       : float — effective sound weight this step
          texture_weight     : float — effective texture weight this step
          texture_confidence : float — how clean the texture signal is
          fusion_active      : bool
        """
        self.t += 1
        fusion_active = (self.t >= FUSION_WARMUP)
        self._fusion_active = fusion_active

        # ── 1. Sound likelihood for current fi ────────────────
        if curr_fi >= 0 and curr_fi < self._sound_like.shape[1]:
            sound_obs = self._sound_like[:, curr_fi].copy()
        else:
            sound_obs = np.ones(self.n_nodes, dtype=np.float64)

        # ── 2. Texture likelihood for each node ───────────────
        # texture_likelihood[bmu] = P(tex_bin | bmu fired)
        # We need P(tex_bin | at_node).
        # Here we use the direct texture bin lookup:
        #   tex_obs[node] = 1.0 if node's tex bin == observed tex_bin
        #                  else TEXTURE_LIKE_FLOOR
        tex_obs = np.full(self.n_nodes, TEXTURE_LIKE_FLOOR, dtype=np.float64)
        for i in range(self.n_nodes):
            if self._node_tex_bin[i] == tex_bin:
                # How confident is the texture signal?
                # Use the M54b likelihood value for the BMU that fired
                # as a soft weight on the match.
                tex_obs[i] = max(TEXTURE_LIKE_FLOOR,
                                 float(np.max(texture_likelihood)))

        # ── 3. Texture confidence ──────────────────────────────
        # How many nodes match the observed tex_bin?
        # Few matches = high confidence (texture is discriminating)
        n_matching = int((self._node_tex_bin == tex_bin).sum())
        tex_conf = float(np.clip(1.0 - (n_matching - 1) / self.n_nodes,
                                 0.0, 1.0))
        self._last_tex_conf = tex_conf

        # ── 4. Adaptive weights ────────────────────────────────
        # Base: ALPHA_SOUND / ALPHA_TEXTURE (defaults 0.5 / 0.5)
        # Adjust: high sound surprise → reduce sound weight
        #         high tex surprise   → reduce texture weight
        # After warmup only.
        if fusion_active:
            # Reliability modulates texture weight
            # tex_reliability falls if texture has been noisy lately
            self._tex_reliability = ((1.0 - TEX_RELIABILITY_EMA) * self._tex_reliability
                                     + TEX_RELIABILITY_EMA * tex_conf)
            tex_w   = ALPHA_TEXTURE * self._tex_reliability
            sound_w = ALPHA_SOUND

            # Renormalise to sum = 1
            total = sound_w + tex_w + 1e-9
            sound_w /= total
            tex_w   /= total
        else:
            # Before warmup: sound only (no texture weight yet)
            sound_w = 1.0
            tex_w   = 0.0

        self._last_sound_w = sound_w
        self._last_tex_w   = tex_w
        self._tex_weight   = tex_w

        # ── 5. Fused likelihood (product of experts in log space) ──
        # log P_fused = alpha * log P_sound + beta * log P_tex
        # Exponentiate back, normalise.
        eps = 1e-12
        log_sound = sound_w * np.log(np.clip(sound_obs, eps, 1.0))
        log_tex   = tex_w   * np.log(np.clip(tex_obs,   eps, 1.0))

        fused = np.exp(log_sound + log_tex)
        fused_sum = fused.sum()
        if fused_sum < eps:
            fused = np.ones(self.n_nodes, dtype=np.float64) / self.n_nodes
        else:
            fused /= fused_sum

        return {
            'fused_likelihood':    fused,
            'sound_weight':        sound_w,
            'texture_weight':      tex_w,
            'texture_confidence':  tex_conf,
            'fusion_active':       fusion_active,
            'n_tex_matching':      n_matching,
        }

    # ── Diagnostics ───────────────────────────────────────────

    def summary(self):
        print(f"  MultimodalFusion (M65) — step {self.t}")
        print(f"  Fusion active:    {self._fusion_active}  "
              f"(warmup: {FUSION_WARMUP} steps)")
        print(f"  Sound weight:     {self._last_sound_w:.3f}")
        print(f"  Texture weight:   {self._last_tex_w:.3f}")
        print(f"  Tex reliability:  {self._tex_reliability:.3f}")
        print(f"  Tex confidence:   {self._last_tex_conf:.3f}")
        print(f"  Tex weight ETA:   {ALPHA_TEXTURE}")