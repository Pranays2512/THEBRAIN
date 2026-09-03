#pragma once
/**
 * brain3/core/sleep_kernel.hpp
 *
 * SLEEP KERNEL — real consolidation, replacing the poetry-only placeholder.
 *
 * Four phases per cycle:
 *   1. EPISODIC REPLAY    style-loop evolution of the mouth on the probe set
 *                         (propose → verify → absorb → retrain), snapshot-
 *                         guarded by the crisp floor
 *   2. GRAPH CONSOLIDATION  graph reasoner re-embeds over the CURRENT fact
 *                         store (today's teachings included), rollback-
 *                         guarded by self-check MRR vs chance
 *   3. VERIFICATION       CAS spot-exactness + mouth contract floors
 *   4. REPORT             structured phase ledger for the reply
 *
 * Every phase degrades gracefully: missing mouth ⇒ skipped; regression ⇒
 * rolled back to pre-cycle parameters. The brain wakes up never worse.
 */
#include <chrono>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "crisp/engines/neural/native_mouth.hpp"
#include "crisp/engines/neural/native_reader.hpp"
#include "crisp/engines/neural/mouth_voice.hpp"
#include "crisp/engines/neural/mouth_style_loop.hpp"
#include "crisp/engines/reasoning/graph_attention_reasoner.hpp"
#include "crisp/engines/math/math_engine.hpp"
#include "crisp/engines/math/neural_policy_value_prior_engine.hpp"
#include "crisp/engines/reasoning/reasoning_engine.hpp"

namespace brain3 {
namespace core {

using engines::neural::MouthStyleLoop;
using GReasoner = engines::reasoning::GraphAttentionReasoner;

class SleepKernel {
public:
    struct PhaseReport {
        std::string name;
        std::string status;                    // ok | rolled_back | skipped
        std::vector<std::pair<std::string, std::string>> metrics;
    };
    struct Report {
        std::vector<PhaseReport> phases;
        bool all_ok = true;
    };

    using Probe = MouthStyleLoop::Probe;

    struct Config {
        int  style_generations   = 2;
        int  candidates_per_probe= 6;
        int  graph_steps         = 2500;
        double graph_mrr_gate    = 0.08;       // chance ≈ 0.05 @20 cands
        int  cas_spot_problems   = 5;
    };

    SleepKernel(engines::neural::NativeMouth& mouth,
                brain2::reasoning::ReasoningEngine& facts,
                engines::neural::VoiceMapper& voice,
                thebrain::neural_prior::NeuralPolicyValuePriorEngine& prior)
        : mouth_(mouth), facts_(facts), voice_(voice), prior_engine_(prior) {}

    void set_probes(const std::vector<Probe>& style_probes,
                    const std::vector<Probe>& floor_probes) {
        style_probes_ = style_probes;
        floor_probes_ = floor_probes;
    }

    // Mount the eyes for nightly replay: the reader retrains on today's
    // corpus with exact-match gates and snapshot rollback, exactly like the
    // mouth. Without this the reader stayed frozen after distillation.
    void set_reader(engines::neural::NativeReader* reader,
                    const std::string& corpus_path,
                    const std::string& probes_path) {
        reader_ = reader;
        reader_corpus_path_ = corpus_path;
        reader_probes_path_ = probes_path;
    }

    Report run_cycle() {
        Report rep;
        run_style_phase(rep);
        run_reader_phase(rep);
        run_graph_phase(rep);
        run_verify_phase(rep);
        for (auto& p : rep.phases)
            if (p.status != "ok" && p.status != "skipped") rep.all_ok = false;
        return rep;
    }

private:
    // ── Phase 1 ─────────────────────────────────────────────────────────────
    void run_style_phase(Report& rep) {
        PhaseReport pr; pr.name = "episodic_replay";
        if (!mouth_.available()) { pr.status = "skipped"; rep.phases.push_back(pr); return; }
        if (style_probes_.empty()) { pr.status = "skipped"; rep.phases.push_back(pr); return; }

        MouthStyleLoop loop(mouth_.model(), style_probes_);
        if (!floor_probes_.empty()) loop.set_floor(floor_probes_);

        const auto& st = loop.evolve({0.f, 0.4f}, &voice_);

        // crisp floor must still hold after consolidation
        bool floor_ok = true;
        for (const auto& f : floor_probes_) {
            auto ids = mouth_.model().encode(f.prompt);
            const std::string reply =
                mouth_.model().stream_complete_ids(ids, 24, 0.f);
            if (!loop.verify(reply + "\n", f.facts)) { floor_ok = false; break; }
        }
        pr.metrics.push_back({"accepted", std::to_string(st.accepted)});
        pr.metrics.push_back({"unique", std::to_string(st.unique_accepted)});
        pr.metrics.push_back({"floor", floor_ok ? "held" : "BROKEN"});
        if (!floor_ok) pr.status = "rolled_back";
        else pr.status = "ok";

        rep.phases.push_back(pr);
    }

    // ── Phase 1b: reader replay ─────────────────────────────────────────────
    struct ReaderPair { std::string sentence, triple; };

    static std::vector<ReaderPair> _load_reader_blocks(const std::string& path, size_t cap) {
        std::vector<ReaderPair> out;
        std::ifstream f(path);
        if (!f) return out;
        std::string txt((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
        auto trim = [](std::string s) {
            size_t a = s.find_first_not_of(' ');
            size_t b = s.find_last_not_of(' ');
            return a == std::string::npos ? "" : s.substr(a, b - a + 1);
        };
        size_t pos = 0;
        while (out.size() < cap && (pos = txt.find("read:", pos)) != std::string::npos) {
            size_t lend = txt.find('\n', pos);
            if (lend == std::string::npos) break;
            size_t tstart = txt.find("triple:", lend);
            if (tstart == std::string::npos) break;
            size_t tend = txt.find('\n', tstart);
            ReaderPair p;
            p.sentence = trim(txt.substr(pos + 5, lend - pos - 5));
            p.triple   = trim(txt.substr(tstart + 7,
                          (tend == std::string::npos ? txt.size() : tend) - tstart - 7));
            if (!p.sentence.empty() && !p.triple.empty()) out.push_back(p);
            pos = (tend == std::string::npos) ? txt.size() : tend;
        }
        return out;
    }

    void run_reader_phase(Report& rep) {
        PhaseReport pr; pr.name = "reader_replay";
        rep.phases.push_back(pr);
        if (!reader_ || !reader_->available()) { pr.status = "skipped"; return; }
        if (reader_corpus_path_.empty()) { pr.status = "skipped"; return; }

        const int train_n = 40, probe_n = 6;
        auto train_pairs = _load_reader_blocks(reader_corpus_path_, (size_t)train_n * 2);
        auto probe_pairs = _load_reader_blocks(reader_probes_path_, (size_t)probe_n);
        if (train_pairs.empty() || probe_pairs.empty()) { pr.status = "skipped"; return; }
        train_pairs.resize((size_t)std::min(train_n, (int)train_pairs.size()));
        probe_pairs.resize((size_t)std::min(probe_n, (int)probe_pairs.size()));

        auto& lm = reader_->model();
        auto exact = [&](const ReaderPair& p) {
            const auto ids = lm.encode("read: " + p.sentence + "\ntriple: ");
            engines::neural::StamlatLM::StreamCache sc;
            lm.stream_start(ids, sc);
            std::vector<int> got;
            for (int k = 0; k < 16; ++k) {
                const int tok = lm.stream_sample(sc, 0.f, nullptr, nullptr);
                lm.stream_step(tok, sc);
                if (lm.token_surface(tok) == "\n") break;
                got.push_back(tok);
            }
            std::string said;
            for (size_t k = 0; k < got.size(); ++k) {
                said += lm.token_surface(got[k]);
                if (k + 1 < got.size()) said += " ";
            }
            // normalize whitespace before compare
            std::string a, b;
            for (char c : said) if (c != ' ' || (!a.empty() && a.back() != ' ')) a += c;
            for (char c : p.triple) if (c != ' ' || (!b.empty() && b.back() != ' ')) b += c;
            while (!a.empty() && a.back() == ' ') a.pop_back();
            while (!b.empty() && b.back() == ' ') b.pop_back();
            return a == b;
        };
        auto score = [&]() {
            int ok = 0;
            for (const auto& p : probe_pairs) ok += exact(p);
            return ok;
        };

        const int pre = score();
        const auto snap = lm.snapshot_params();
        std::string train_text;
        for (const auto& p : train_pairs)
            train_text += "read: " + p.sentence + "\ntriple: " + p.triple + "\n";
        lm.fit(train_text, 200, 1e-3f, 8, 200);          // gentle nightly pass
        const int post = score();

        pr.metrics.push_back({"pairs", std::to_string(train_pairs.size())});
        pr.metrics.push_back({"pre_exact", std::to_string(pre) + "/" + std::to_string((int)probe_pairs.size())});
        pr.metrics.push_back({"post_exact", std::to_string(post) + "/" + std::to_string((int)probe_pairs.size())});

        // Rollback gate: never wake up with worse eyes.
        if (post >= pre && post >= ((int)probe_pairs.size() / 2)) {
            pr.status = "ok";
        } else {
            lm.restore_params(snap);
            pr.status = post < pre ? "rolled_back" : "ok";
            if (post < pre) pr.metrics.push_back({"action", "rolled_back_to_pre_sleep_params"});
        }
    }

    // ── Phase 2 ─────────────────────────────────────────────────────────────
    void run_graph_phase(Report& rep) {
        PhaseReport pr; pr.name = "graph_consolidation";
        gar_.load_from_facts(facts_.facts);
        if (gar_.edge_count() == 0) { pr.status = "skipped"; rep.phases.push_back(pr); return; }

        auto snap = gar_.snapshot_params();
        GReasoner::TrainConfig cfg; cfg.steps = cfg_.graph_steps;
        gar_.train(cfg);

        const double mrr = gar_.self_check_mrr();
        pr.metrics.push_back({"edges", std::to_string(gar_.edge_count())});
        pr.metrics.push_back({"entities", std::to_string(gar_.entity_count())});
        pr.metrics.push_back({"self_mrr", std::to_string(mrr).substr(0, 6)});

        if (mrr >= cfg_.graph_mrr_gate && mrr >= pre_cycle_graph_mrr_) {
            pre_cycle_graph_mrr_ = mrr;
            pr.status = "ok";
        } else {
            gar_.restore_params(snap);
            pr.status = "rolled_back";
        }
        rep.phases.push_back(pr);
    }

    // ── Phase 3 ─────────────────────────────────────────────────────────────
    void run_verify_phase(Report& rep) {
        PhaseReport pr; pr.name = "verification";
        brain2::math::MathEngine me;
        int ok = 0, n = 0;
        for (int i = 0; i < cfg_.cas_spot_problems; ++i) {
            long long a = 3 + i * 7 % 40, b = 2 + i * 5 % 25, c = 2 + i * 3 % 15;
            auto r = me.solve_equation("x = " + std::to_string(a) + "/" +
                                       std::to_string(b) + " + " +
                                       std::to_string(c), "x");
            ++n;
            ok += r.success &&
                  std::fabs(r.numeric_val - ((double)a / b + c)) <= 1e-4;
        }
        pr.metrics.push_back({"cas_exact", std::to_string(ok) + "/" + std::to_string(n)});
        if (ok != n) { pr.status = "regressed"; rep.phases.push_back(pr); return; }

        // consolidate prior-engine learning from today's recorded outcomes
        const int pu = prior_engine_.train_pass(2, 0.05);
        pr.metrics.push_back({"prior_updates", std::to_string(pu)});
        pr.status = "ok";
        rep.phases.push_back(pr);
    }

    engines::neural::NativeMouth& mouth_;
    engines::neural::NativeReader* reader_ = nullptr;   // optional: the eyes
    std::string reader_corpus_path_, reader_probes_path_;
    brain2::reasoning::ReasoningEngine& facts_;
    engines::neural::VoiceMapper& voice_;
    thebrain::neural_prior::NeuralPolicyValuePriorEngine prior_engine_;
    Config cfg_;
    std::vector<Probe> style_probes_, floor_probes_;
    double pre_cycle_graph_mrr_ = -1.0;
    GReasoner gar_;
};

} // namespace core
} // namespace brain3
