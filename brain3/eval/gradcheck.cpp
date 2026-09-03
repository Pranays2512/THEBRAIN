// eval/gradcheck.cpp — FINITE-DIFFERENCE GRADIENT VALIDATION
//
// brain3's fuzzy hemisphere now trains on live traffic (~1M parameters). The
// integration probe proved the LM's loss falls on repeated input. That proves
// something is learning; it does NOT prove the gradients are correct. A loss
// can fall under a gradient that is wrong in scale, or that is missing a term,
// or that reads a weight after updating it. Those errors are invisible to a
// loss curve and they contaminate every experiment downstream.
//
// This harness compares each learner's ANALYTIC update against a central
// difference of the true objective.
//
// METHOD. These learners apply their updates in place (theta -= lr*g) rather
// than exposing a gradient buffer, so the analytic gradient is recovered from
// the weight delta:
//
//     g_analytic = (theta_before - theta_after) / lr          [descent]
//     g_analytic = (theta_after - theta_before) / lr          [ascent]
//
// with any decoupled L2 shrink term subtracted off, since that is a
// regularizer and not part of the gradient. The numeric estimate is the
// standard symmetric difference
//
//     g_numeric  = ( L(theta + eps) - L(theta - eps) ) / (2 eps)
//
// Agreement is scored by relative error, with an absolute floor so that
// coordinates whose true gradient is ~0 do not produce spurious failures.
//
// This file is a measurement instrument. It prints what it checked, what it
// could not check, and exits non-zero only on a genuine mismatch.
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>

#include <fstream>

#include "fuzzy/core/sparse_lstm.hpp"
#include "fuzzy/core/basal_ganglia.hpp"
#include "fuzzy/engines/synthesis/unified_proposer.hpp"

// These learners still live in the legacy brain2 namespace even though the
// files are under brain3/.
using brain2::SparseLSTMLayer;
using brain2::BasalGanglia;

// ─────────────────────────────────────────────────────────────────────────────
// Scoring
// ─────────────────────────────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;
static std::vector<std::string> g_failures;

// Relative error, essentially pure: the denominator floor exists only to stop
// division by a genuinely-zero gradient.
//
// An earlier version of this file used max(1, |a|, |n|), which silently turned
// the whole check into an ABSOLUTE tolerance of 0.02 and reported "pass" for
// every gradient smaller than that — including a BasalGanglia actor column
// whose analytic value was flatly zero. A gradient check that cannot fail is
// worse than no gradient check.
static double rel_err(double a, double n) {
    const double denom = std::max(std::max(std::fabs(a), std::fabs(n)), 1e-12);
    return std::fabs(a - n) / denom;
}

// Coordinates whose true gradient is smaller than this are NOT scored, and the
// count of skipped coordinates is reported.
//
// Justification, so this is not a convenient way to hide failures: the forward
// pass accumulates in float, so L carries ~1e-7 relative error. A symmetric
// difference divides that by 2*eps, putting a noise floor of roughly
// 1e-7 / 2e-3 = 5e-5 on any numeric gradient estimate. Separately, the analytic
// value is recovered from a float weight delta, which quantizes at ~1e-7 near
// unit magnitude. Below ~1e-4 the two estimates simply cannot be compared, and
// scoring them measures float rounding rather than calculus.
static constexpr double kNoiseFloor = 1e-4;

struct CheckStats {
    int n = 0, agree = 0, skipped = 0;
    double worst_rel = 0.0;
    int    worst_idx = -1;
    double worst_a = 0.0, worst_n = 0.0;

    // Score one coordinate, or skip it if it sits under the noise floor.
    void add(int idx, double a, double nu) {
        if (std::max(std::fabs(a), std::fabs(nu)) < kNoiseFloor) { ++skipped; return; }
        const double r = rel_err(a, nu);
        ++n;
        if (r <= 2e-2) ++agree;
        if (r > worst_rel) { worst_rel = r; worst_idx = idx; worst_a = a; worst_n = nu; }
    }
};

static void report(const std::string& what, const CheckStats& s, double tol,
                   const std::string& note = "") {
    const bool ok = (s.n > 0) && (s.agree == s.n);
    if (ok) ++g_pass; else ++g_fail;
    std::cout << (ok ? "  PASS  " : "  FAIL  ")
              << std::left << std::setw(52) << what
              << s.agree << "/" << s.n << " within " << tol;
    if (s.skipped) std::cout << "  (" << s.skipped << " below noise floor)";
    if (!ok) {
        std::cout << "   worst[" << s.worst_idx << "] analytic=" << std::setprecision(6)
                  << s.worst_a << " numeric=" << s.worst_n
                  << " rel=" << s.worst_rel;
    }
    std::cout << "\n";
    if (!note.empty()) std::cout << "        " << note << "\n";
    if (!ok) {
        g_failures.push_back(what + "  (" + std::to_string(s.agree) + "/" +
                             std::to_string(s.n) + ", worst rel=" +
                             std::to_string(s.worst_rel) + ")");
    }
}

static void assert_true(const std::string& what, bool cond, const std::string& got) {
    if (cond) ++g_pass; else ++g_fail;
    std::cout << (cond ? "  PASS  " : "  FAIL  ")
              << std::left << std::setw(52) << what;
    if (!cond) std::cout << "   " << got;
    std::cout << "\n";
    if (!cond) g_failures.push_back(what + "  (" + got + ")");
}

#define SECTION(t) std::cout << "\n\033[1m" << t << "\033[0m\n"

// ═════════════════════════════════════════════════════════════════════════════
// 1. SparseLSTMLayer — the BPTT gate derivatives
// ═════════════════════════════════════════════════════════════════════════════
// The single largest and most error-prone block of hand-derived calculus in the
// repository: forget/input/cell/output gate derivatives, the cell-state carry,
// and the recurrent path. Checked over a 2-step sequence, because with h_prev
// zeroed at reset the recurrent matrix Wh contributes nothing on step 1 and a
// 1-step check would silently skip it entirely.
static void check_sparse_lstm() {
    SECTION("1. SparseLSTMLayer — 2-step BPTT (Wx, Wh, b)");

    const int I = 6, H = 6;
    std::mt19937 rng(7);

    // Objective: L = 0.5 * || h(step 2) - target ||^2, so dL/dh2 = h2 - target
    // and dL/dh1 = 0. Small inputs keep dpre well inside the +-5 clip so the
    // clip does not silently flatten the gradient we are trying to measure.
    std::normal_distribution<float> nd(0.f, 0.35f);
    std::vector<float> x0(I), x1(I), tgt(H);
    for (auto& v : x0)  v = nd(rng);
    for (auto& v : x1)  v = nd(rng);
    for (auto& v : tgt) v = nd(rng);

    auto make = [&]() {
        std::mt19937 r(1234);
        return SparseLSTMLayer(I, H, H, r);   // k_active == H: all neurons active
    };

    auto loss_of = [&](SparseLSTMLayer& L) {
        L.reset_state();
        L.forward(x0, /*record_history=*/false);
        auto h2 = L.forward(x1, /*record_history=*/false);
        double s = 0.0;
        for (int j = 0; j < H; ++j) { const double d = h2[j] - tgt[j]; s += d * d; }
        return 0.5 * s;
    };

    // Analytic: run one BPTT step and recover g from the weight delta.
    //
    // lr = 1 deliberately. The recovery theta_before - theta_after = lr*g is
    // algebraically exact rather than a finite-difference approximation, so lr
    // does not trade off bias — it only sets how many float mantissa bits the
    // delta occupies. At lr = 1e-3 a gradient of 1e-3 produces a delta of 1e-6,
    // which near unit weight magnitude is barely 8 representable steps, and the
    // quantization dominated the comparison.
    const float lr = 1.0f;
    SparseLSTMLayer A = make();
    const std::vector<float> Wx0(A.Wx.begin(), A.Wx.end());
    const std::vector<float> Wh0(A.Wh.begin(), A.Wh.end());
    const std::vector<float> b0 (A.b.begin(),  A.b.end());
    {
        A.reset_state();
        A.forward(x0, true);
        auto h2 = A.forward(x1, true);
        std::vector<float> d2(H);
        for (int j = 0; j < H; ++j) d2[j] = h2[j] - tgt[j];
        std::vector<std::vector<float>> dseq = { std::vector<float>(H, 0.f), d2 };
        A.backward_through_time(dseq, lr, 2);
    }

    struct Probe { const char* name; const std::vector<float>* before;
                   std::vector<float>* live; size_t n; };
    SparseLSTMLayer N = make();
    Probe probes[] = {
        {"Wx (input->gates)",     &Wx0, nullptr, A.Wx.size()},
        {"Wh (recurrent->gates)", &Wh0, nullptr, A.Wh.size()},
        {"b  (gate biases)",      &b0,  nullptr, A.b.size()},
    };
    std::vector<float>* live_after[] = { &A.Wx, &A.Wh, &A.b };
    std::vector<float>* live_num[]   = { &N.Wx, &N.Wh, &N.b };

    const double eps = 2e-3, tol = 2e-2;
    for (int p = 0; p < 3; ++p) {
        CheckStats st;
        const size_t n = probes[p].n;
        // Sample coordinates rather than sweeping: each numeric probe is two
        // full forward passes, and coverage matters more than exhaustiveness.
        std::mt19937 pick(99 + p);
        const int trials = (int)std::min<size_t>(24, n);
        for (int t = 0; t < trials; ++t) {
            const size_t c = pick() % n;
            const double g_ana = ((double)(*probes[p].before)[c] - (double)(*live_after[p])[c]) / lr;

            // restore the numeric copy, then perturb one coordinate
            *live_num[p] = *probes[p].before;
            const float saved = (*live_num[p])[c];
            (*live_num[p])[c] = (float)(saved + eps); const double Lp = loss_of(N);
            (*live_num[p])[c] = (float)(saved - eps); const double Lm = loss_of(N);
            (*live_num[p])[c] = saved;
            const double g_num = (Lp - Lm) / (2.0 * eps);
            st.add((int)c, g_ana, g_num);
        }
        report(std::string("BPTT gradient: ") + probes[p].name, st, tol);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. BasalGanglia — actor-critic
// ═════════════════════════════════════════════════════════════════════════════
// apply_grad() treats td_error as a constant (it is a function argument and is
// never differentiated through), so the scalar objective it ascends is
//
//     F(theta) = td * [ log pi(a|s) + V(s) ]
//
// The actor term contributes td*(1{i==a} - p_i)*h1[j] to dF/dW2[i][j] — the
// (1 - p_a) factor on the chosen row AND a -p_i push-down on every other row.
// The critic term contributes td*h1[j] to dF/dW_v[j].
namespace bg_check {

static std::vector<float> hidden_of(const BasalGanglia& g, const std::vector<float>& inp) {
    std::vector<float> h(g.hidden, 0.f);
    const int in = 5 * g.n_dims;
    for (int i = 0; i < g.hidden; ++i) {
        float s = g.b1[i];
        const float* row = g.W1.data() + (size_t)i * in;
        for (int k = 0; k < in; ++k) s += row[k] * inp[k];
        h[i] = std::tanh(s);
    }
    return h;
}

static std::vector<double> probs_of(const BasalGanglia& g, const std::vector<float>& h) {
    std::vector<double> lg(g.n_ops, 0.0);
    for (int i = 0; i < g.n_ops; ++i) {
        double s = g.b2[i];
        const float* row = g.W2.data() + (size_t)i * g.hidden;
        for (int j = 0; j < g.hidden; ++j) s += row[j] * h[j];
        lg[i] = s;
    }
    const double mx = *std::max_element(lg.begin(), lg.end());
    double z = 0.0;
    for (auto& v : lg) { v = std::exp(v - mx); z += v; }
    for (auto& v : lg) v /= z;
    return lg;
}

static double value_of(const BasalGanglia& g, const std::vector<float>& h) {
    double v = g.b_v[0];
    for (int j = 0; j < g.hidden; ++j) v += g.W_v[j] * h[j];
    return v;
}

// F(theta) = td * ( log pi(a|s) + V(s) ), recomputing h1 from W1 so that the
// hidden-layer coordinates are actually exercised.
static double objective(const BasalGanglia& g, const std::vector<float>& inp,
                        int a, double td) {
    const auto h = hidden_of(g, inp);
    const auto p = probs_of(g, h);
    return td * (std::log(std::max(p[(size_t)a], 1e-300)) + value_of(g, h));
}

} // namespace bg_check

static void check_basal_ganglia() {
    SECTION("2. BasalGanglia — TD actor-critic (the only RL in brain3)");

    const int nd = 4;                  // 5*nd = 20 inputs; small and fast
    const int a  = 3;                  // chosen op
    const double td = 0.7;             // inside the +-2 clip
    const float  lr = 1.0f;            // exact recovery; see the LSTM note above

    std::mt19937 rng(11);
    std::normal_distribution<float> nd_(0.f, 0.5f);
    std::vector<float> inp(5 * nd);
    for (auto& v : inp) v = nd_(rng);

    // The default init (W1 ~ N(0,0.02)) drives h1 to ~1e-2 and leaves the 31
    // logits near-uniform, so p_a ~ 1/31 and every gradient here is ~1e-3 —
    // small enough that a broken term hides inside any reasonable tolerance.
    // Scale the hidden layer into its normal operating range and concentrate
    // the policy so the (1 - p_a) factor is a large, unmissable effect.
    auto make = [&]() {
        BasalGanglia g(nd, lr, 42);
        std::mt19937 wr(2024);
        std::normal_distribution<float> w(0.f, 0.6f);
        for (auto& v : g.W1)  v = w(wr);
        for (auto& v : g.W2)  v = w(wr);
        for (auto& v : g.W_v) v = w(wr);
        g.b2[a] = 4.0f;                 // p_a ~ 0.8, so (1 - p_a) ~ 0.2
        return g;
    };

    BasalGanglia A = make();
    const std::vector<float> W2_0(A.W2.begin(), A.W2.end());
    const std::vector<float> Wv_0(A.W_v.begin(), A.W_v.end());
    const std::vector<float> W1_0(A.W1.begin(), A.W1.end());
    const auto h1 = bg_check::hidden_of(A, inp);
    A.apply_grad(a, h1, inp, (float)td);

    BasalGanglia N = make();
    const double eps = 1e-3, tol = 2e-2;
    const double shrink = 0.005;       // decoupled L2 in apply_grad; not gradient

    // ── critic: W_v ──────────────────────────────────────────────────────────
    {
        CheckStats st;
        std::mt19937 pick(5);
        for (int t = 0; t < 24; ++t) {
            const size_t c = pick() % Wv_0.size();
            const double g_ana = ((double)A.W_v[c] - (double)Wv_0[c]) / lr + shrink * Wv_0[c];
            N.W_v = Wv_0; N.W2 = W2_0; N.W1 = W1_0;
            const float saved = N.W_v[c];
            N.W_v[c] = (float)(saved + eps); const double Fp = bg_check::objective(N, inp, a, td);
            N.W_v[c] = (float)(saved - eps); const double Fm = bg_check::objective(N, inp, a, td);
            N.W_v[c] = saved;
            st.add((int)c, g_ana, (Fp - Fm) / (2.0 * eps));
        }
        report("critic W_v ascent == dF/dW_v", st, tol);
    }

    // ── actor: W2, chosen row a ─────────────────────────────────────────────
    {
        CheckStats st;
        for (int j = 0; j < 24; ++j) {
            const size_t c = (size_t)a * A.hidden + (size_t)j;
            const double g_ana = ((double)A.W2[c] - (double)W2_0[c]) / lr + shrink * W2_0[c];
            N.W_v = Wv_0; N.W2 = W2_0; N.W1 = W1_0;
            const float saved = N.W2[c];
            N.W2[c] = (float)(saved + eps); const double Fp = bg_check::objective(N, inp, a, td);
            N.W2[c] = (float)(saved - eps); const double Fm = bg_check::objective(N, inp, a, td);
            N.W2[c] = saved;
            st.add((int)c, g_ana, (Fp - Fm) / (2.0 * eps));
        }
        report("actor W2[chosen row] == td*(1-p_a)*h1", st, tol,
               "a mismatch of exactly 1/(1-p_a) here means the softmax "
               "normalization term is missing");
    }

    // ── actor: W2, a NON-chosen row (must receive -td*p_i*h1) ───────────────
    {
        const int other = (a + 1) % A.n_ops;
        CheckStats st;
        for (int j = 0; j < 16; ++j) {
            const size_t c = (size_t)other * A.hidden + (size_t)j;
            const double g_ana = ((double)A.W2[c] - (double)W2_0[c]) / lr + shrink * W2_0[c];
            N.W_v = Wv_0; N.W2 = W2_0; N.W1 = W1_0;
            const float saved = N.W2[c];
            N.W2[c] = (float)(saved + eps); const double Fp = bg_check::objective(N, inp, a, td);
            N.W2[c] = (float)(saved - eps); const double Fm = bg_check::objective(N, inp, a, td);
            N.W2[c] = saved;
            st.add((int)c, g_ana, (Fp - Fm) / (2.0 * eps));
        }
        report("actor W2[non-chosen row] == -td*p_i*h1", st, tol,
               "an all-zero analytic column here means non-chosen logits are "
               "never pushed down, so the policy cannot converge");
    }

    // ── shared hidden layer: W1 ─────────────────────────────────────────────
    // Both heads backprop into h1, so this is where a stale W2 read shows up.
    {
        CheckStats st;
        std::mt19937 pick(17);
        for (int t = 0; t < 24; ++t) {
            const size_t c = pick() % W1_0.size();
            const double g_ana = ((double)A.W1[c] - (double)W1_0[c]) / lr + shrink * W1_0[c];
            N.W_v = Wv_0; N.W2 = W2_0; N.W1 = W1_0;
            const float saved = N.W1[c];
            N.W1[c] = (float)(saved + eps); const double Fp = bg_check::objective(N, inp, a, td);
            N.W1[c] = (float)(saved - eps); const double Fm = bg_check::objective(N, inp, a, td);
            N.W1[c] = saved;
            st.add((int)c, g_ana, (Fp - Fm) / (2.0 * eps));
        }
        report("shared hidden W1 == dF/dW1", st, tol,
               "this is the coordinate set that a post-update W2 read corrupts");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. RecurrentIntuitionBlock — optimizer bookkeeping
// ═════════════════════════════════════════════════════════════════════════════
// The router's gradient is obscured by Adam, so rather than pretend to check it
// we check the two bookkeeping invariants that are unambiguous and that
// silently corrupt Adam's bias correction and every reported training count.
static void check_intuition_bookkeeping() {
    SECTION("3. RecurrentIntuitionBlock — Adam bookkeeping");

    using brain3::engines::synthesis::RecurrentIntuitionBlock;
    RecurrentIntuitionBlock b;

    std::vector<double> feats(b.input_dim, 0.25);
    feats[b.input_dim - 1] = 1.0;                 // bias feature

    auto [probs, states] = b.forward(feats, 5, 0.85);
    const long long t_before = b.adam_t_;
    const int        u_before = b.total_updates;
    b.backward(feats, states, probs, 0, 1.0);
    const long long t_after = b.adam_t_;
    const int        u_after = b.total_updates;

    // Adam's timestep must advance once per optimizer STEP. adam_upd() bumps it
    // per parameter GROUP, and backward() calls adam_upd six times, so the
    // bias-correction denominators race ahead by 6x per update.
    assert_true("adam_t_ advances exactly 1 per backward()",
                (t_after - t_before) == 1,
                "advanced by " + std::to_string(t_after - t_before) +
                " (bias correction 1-b^t is computed at the wrong t)");

    assert_true("total_updates advances exactly 1 per backward()",
                (u_after - u_before) == 1,
                "advanced by " + std::to_string(u_after - u_before) +
                " (reported training-step count is inflated)");
}

int main() {
    std::cout << "==========================================================\n"
              << " BRAIN3 GRADIENT VALIDATION (finite differences)\n"
              << " Analytic updates vs central differences of the objective.\n"
              << "==========================================================\n";

    check_sparse_lstm();
    check_basal_ganglia();
    check_intuition_bookkeeping();

    std::cout << "\n==========================================================\n"
              << " RESULT: " << g_pass << " passed, " << g_fail << " failed\n"
              << "==========================================================\n";
    for (const auto& f : g_failures) std::cout << "  - " << f << "\n";

    std::cout << "\n NOT COVERED (stated so the score is not mistaken for "
                 "full coverage):\n"
              << "   • LMHead W_proj/bias — Adam-updated, so the weight delta "
                 "is sign(g)\n     rather than g and is not directly invertible.\n"
              << "   • HierarchicalPredictor gate weights — documented as a "
                 "1-step RTRL\n     approximation, so a mismatch against exact "
                 "BPTT is by design.\n"
              << "   • Attention backward in train_lm_sequence_per_token — "
                 "reachable only\n     through Predictor's private members.\n";

    return g_fail == 0 ? 0 : 1;
}
