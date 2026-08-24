#pragma once
/**
 * brain3/crisp/engines/neural/mouth_style_loop.hpp
 *
 * STYLE LOOP — "Discovery" applied to voice.
 *
 * Fuzzy proposes, crisp disposes, the survivor becomes the self:
 *   1. PROPOSE  sample K candidate replies per probe at the mood's decoding
 *               policy (temperature + style bias from VoiceMapper).
 *   2. VERIFY   crisp checker: content classes satisfied (any-of equivalence
 *               groups), structurally terminated, mean reply NLL below gate.
 *   3. ABSORB   accepted replies become weighted SFT examples — weight 1 on
 *               reply emission, 0 on the prompt prefix (conditioning only).
 *   4. RETRAIN  Adam steps over the accumulated corpus sharpen the voice
 *               toward what survived verification.
 *
 * Measured per generation: acceptance rate climbs, mean NLL of accepted
 * replies drops post-retrain. unique_replies exposes the diversity cost of
 * sharpening honestly rather than hiding it.
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "crisp/engines/neural/stamlat_transformer.hpp"
#include "crisp/engines/neural/mouth_voice.hpp"

namespace brain3 {
namespace engines {
namespace neural {

struct StyleLoopConfig {
    int   candidates_per_probe = 8;
    float propose_temp         = 0.8f;
    float nll_gate             = 2.0f;     // mean reply NLL ceiling (nats/token)
    int   max_reply_tokens     = 24;
    float sft_lr               = 1.5e-3f;
    int   sft_epochs           = 3;
    int   freq_cap             = 6;        // max duplicates per distinct reply
    bool  seed_with_facts      = true;     // hint-exploration: some candidates
    int   seeded_per_probe     = 4;        // start with a forced fact token
    int   max_absorb_per_probe = 5;        // per-generation absorption throttle
};

struct GenerationStats {
    int    generation       = 0;
    int    proposed         = 0;
    int    accepted         = 0;
    int    unique_accepted  = 0;
    double acceptance_rate  = 0.;
    double mean_nll_pre     = 0.;   // accepted replies, before retrain
    double mean_nll_post    = 0.;   // same replies, after retrain
    float  temperature      = 0.f;  // mood policy used for proposals
    int    biased_tokens    = 0;
    bool   rolled_back      = false; // retrain regressed the floor → undone
};

class MouthStyleLoop {
public:
    using FactGroups = std::vector<std::vector<std::string>>;

    struct Probe {
        std::string prompt;      // raw text, e.g. "user: hello\nbrain: "
        FactGroups  facts;       // any-of content classes that MUST survive
        std::string intent;      // label for reporting
    };

    MouthStyleLoop(StamlatLM& lm, std::vector<Probe> probes,
                   StyleLoopConfig cfg = {})
        : lm_(lm), probes_(std::move(probes)), cfg_(cfg) {}

    const GenerationStats& evolve(const brain2::EmotionState& mood,
                                  const VoiceMapper* voice = nullptr) {
        VoicePolicy pol;
        if (voice) pol = voice->policy(lm_, mood);
        else       pol.temperature = cfg_.propose_temp;

        GenerationStats st;
        st.generation   = (int)history_.size() + 1;
        st.temperature  = pol.temperature;
        st.biased_tokens = (int)pol.bias.size();

        std::vector<std::pair<std::vector<int>, std::vector<int>>> fresh_pairs;
        std::vector<double> fresh_nll;

        for (const auto& probe : probes_) {
            const auto prompt_ids = lm_.encode(probe.prompt);
            // fact-token seeds: hint-exploration cycling through EVERY
            // contract surface (not just the first), so variants that
            // unconstrained sampling never reaches still get proposed.
            // The seed is part of the UTTERANCE: verified, absorbed, and
            // trained with emission weight — the loop learns to say it
            // unprompted.
            std::vector<int> seed_pool;
            if (cfg_.seed_with_facts)
                for (const auto& g : probe.facts)
                    for (const auto& s : g) {
                        const int sid = find_surface_id(s);
                        if (sid >= 0) seed_pool.push_back(sid);
                    }
            int probe_absorbed = 0;
            for (int k = 0; k < cfg_.candidates_per_probe; ++k) {
                ++st.proposed;

                // ── propose (collect ids so absorption needs no re-parse) ──
                // For seeded candidates the utterance = seed + sampled tail:
                // the crisp checker judges what the mouth SAID, and SFT
                // carries weight on emitting the seed too — that is how the
                // loop teaches the model to reach the variant unprompted.
                const bool seeded = !seed_pool.empty() &&
                                    (k % 2 == 1) &&
                                    k / 2 < cfg_.seeded_per_probe;
                std::vector<int> ctx_ids = prompt_ids;
                std::vector<int> utt_ids;
                if (seeded) {
                    const int sid = seed_pool[(size_t)(k / 2) % seed_pool.size()];
                    ctx_ids.push_back(sid);
                    utt_ids.push_back(sid);
                }
                StamlatLM::StreamCache sc;
                lm_.stream_start(ctx_ids, sc);
                for (int n = 0; n < cfg_.max_reply_tokens; ++n) {
                    const int tok =
                        lm_.stream_sample(sc, pol.temperature, nullptr, &pol.bias);
                    utt_ids.push_back(tok);
                    lm_.stream_step(tok, sc);
                    if (lm_.token_surface(tok) == "\n") break;
                }
                std::string reply;
                if (!utt_ids.empty() &&
                    lm_.token_surface(utt_ids.back()) == "\n")
                    reply = lm_.decode(utt_ids);

                // ── verify (crisp) ──────────────────────────────────────────
                if (!verify(reply, probe.facts)) continue;
                const double nll = reply_nll(prompt_ids, utt_ids);
                if (!(nll < cfg_.nll_gate)) continue;

                // ── absorb ──────────────────────────────────────────────────
                // Frequency-weighted: a reply seen again adds a duplicate
                // example (capped), raising its gradient share. Sharpening
                // thus follows what the voice ACTUALLY keeps saying —
                // throttled per probe so one dominant line cannot flood
                // the corpus and drag shared embeddings around.
                ++st.accepted;
                if (++probe_absorbed <= cfg_.max_absorb_per_probe) {
                    const auto fit = freq_.find(reply);
                    if (fit == freq_.end()) {
                        freq_.emplace(reply, 1);
                        absorbed_keys_.push_back(reply);
                        corpus_.push_back(to_example(prompt_ids, utt_ids));
                        fresh_pairs.emplace_back(prompt_ids, utt_ids);
                        fresh_nll.push_back(nll);
                        ++st.unique_accepted;
                    } else if (++fit->second <= cfg_.freq_cap) {
                        corpus_.push_back(to_example(prompt_ids, utt_ids));
                    }
                }
            }
        }

        st.acceptance_rate = st.proposed ? (double)st.accepted / st.proposed : 0.;
        if (!fresh_nll.empty()) {
            double s = 0.; for (double v : fresh_nll) s += v;
            st.mean_nll_pre = s / (double)fresh_nll.size();
        }

        // ── retrain on everything absorbed so far ────────────────────────
        if (!fresh_pairs.empty()) {
            const auto checkpoint = lm_.snapshot_params();
            std::mt19937 shuf(1234 + st.generation);
            for (int ep = 0; ep < cfg_.sft_epochs; ++ep) {
                std::shuffle(corpus_.begin(), corpus_.end(), shuf);
                // small batches keep Adam steps well-conditioned at this scale
                for (size_t i = 0; i < corpus_.size(); i += 16) {
                    std::vector<SftExample> batch(corpus_.begin() + i,
                        corpus_.begin() + i + std::min<size_t>(16, corpus_.size() - i));
                    lm_.sft_step(batch, cfg_.sft_lr);
                }
            }
            double s = 0.;
            for (const auto& [p_ids, r_ids] : fresh_pairs)
                s += reply_nll(p_ids, r_ids);          // post-retrain drift check
            st.mean_nll_post = s / (double)fresh_pairs.size();

            // ── crisp disposes of bad retrains: floor must still hold ────
            if (!floor_.empty() && !floor_holds_()) {
                lm_.restore_params(checkpoint);
                st.rolled_back = true;
                double s2 = 0.;
                for (const auto& [p_ids, r_ids] : fresh_pairs)
                    s2 += reply_nll(p_ids, r_ids);
                st.mean_nll_post = s2 / (double)fresh_pairs.size();
            }
        } else {
            st.mean_nll_post = st.mean_nll_pre;
        }

        history_.push_back(st);
        return history_.back();
    }

    // Crisp verification: content classes + termination. (NLL gate lives in
    // evolve() since it needs model access.)
    bool verify(const std::string& reply, const FactGroups& groups) const {
        if (reply.empty() || reply.back() != '\n') return false;   // ran off
        for (const auto& g : groups) {
            bool hit = false;
            for (const auto& r : g)
                if (reply.find(r) != std::string::npos) { hit = true; break; }
            if (!hit) return false;                                // fact lost
        }
        return true;
    }

    const std::vector<GenerationStats>& history() const { return history_; }
    const std::vector<SftExample>&      corpus()  const { return corpus_; }
    const std::vector<std::string>&     replies() const { return absorbed_keys_; }
    const StyleLoopConfig&              config()  const { return cfg_; }

    // Broad-group probes the retrained model must still satisfy greedily
    // after every generation; a violated floor rolls the retrain back.
    void set_floor(std::vector<Probe> floor_probes) {
        floor_ = std::move(floor_probes);
    }

private:
    int find_surface_id(const std::string& s) const {
        for (int id = lm_.char_vocab_size(); id < lm_.total_vocab_size(); ++id)
            if (lm_.token_surface(id) == s) return id;
        return -1;
    }

    // Weighted training pair: prompt positions are conditioning (w=0),
    // reply-emission positions carry w=1. Truncated to one ctx+1 window.
    SftExample to_example(const std::vector<int>& prompt_ids,
                          const std::vector<int>& reply_ids) const {
        std::vector<int> seq = prompt_ids;
        seq.insert(seq.end(), reply_ids.begin(), reply_ids.end());
        const size_t P = prompt_ids.size(), R = reply_ids.size();
        if (seq.size() > (size_t)lm_.config().ctx + 1) {
            const size_t drop = seq.size() - ((size_t)lm_.config().ctx + 1);
            seq.erase(seq.begin(), seq.begin() + (long)drop);
            const size_t kept_P = P > drop ? P - drop : 0;
            return build_example(seq, kept_P, R);
        }
        return build_example(seq, P, R);
    }

    static SftExample build_example(const std::vector<int>& seq,
                                    size_t prompt_len, size_t reply_len) {
        SftExample e;
        const size_t T = seq.size() - 1;
        e.x.assign(seq.begin(), seq.end() - 1);
        e.y.assign(seq.begin() + 1, seq.end());
        e.w.assign(T, 0.f);
        // position t predicts seq[t+1]; weights on targets inside the reply
        for (size_t t = 0; t < T; ++t)
            if (t + 1 >= prompt_len && t + 1 < prompt_len + reply_len)
                e.w[t] = 1.f;
        return e;
    }

    // Mean NLL of reply tokens given prompt under the CURRENT parameters.
    // The scored targets are always the final R positions of the sequence,
    // so left-truncation cannot shift the boundary.
    double reply_nll(const std::vector<int>& prompt_ids,
                     const std::vector<int>& reply_ids) const {
        std::vector<int> seq = prompt_ids;
        seq.insert(seq.end(), reply_ids.begin(), reply_ids.end());
        if (seq.size() > (size_t)lm_.config().ctx + 1) {
            const size_t drop = seq.size() - ((size_t)lm_.config().ctx + 1);
            seq.erase(seq.begin(), seq.begin() + (long)drop);
        }
        const size_t first_target = seq.size() - 1 - reply_ids.size();
        const Mat logits = lm_.full_logits(seq);
        const int V = lm_.total_vocab_size();
        double sum = 0.; int cnt = 0;
        for (size_t t = first_target; t + 1 < seq.size(); ++t) {
            double mx = -1e30;
            for (int v = 0; v < V; ++v) mx = std::max(mx, (double)logits.at((int)t, v));
            double Z = 0.;
            for (int v = 0; v < V; ++v) Z += std::exp((double)logits.at((int)t, v) - mx);
            sum += -(double)logits.at((int)t, seq[t + 1]) + mx + std::log(Z);
            ++cnt;
        }
        return cnt ? sum / cnt : 1e9;
    }

    StamlatLM&              lm_;
    std::vector<Probe>      probes_;
    StyleLoopConfig         cfg_;
    std::vector<SftExample> corpus_;         // freq-weighted via capped duplicates
    std::vector<std::string> absorbed_keys_; // distinct replies, insertion order
    std::unordered_map<std::string, int> freq_;
    std::vector<Probe>      floor_;
    std::vector<GenerationStats> history_;

    bool floor_holds_() const {
        for (const auto& f : floor_) {
            StamlatLM::StreamCache sc;
            lm_.stream_start(lm_.encode(f.prompt), sc);
            std::string reply;
            for (int n = 0; n < 24; ++n) {
                const int tok = lm_.stream_sample(sc, 0.f);
                reply += lm_.token_surface(tok);
                lm_.stream_step(tok, sc);
                if (!reply.empty() && reply.back() == '\n') break;
            }
            if (!verify(reply, f.facts)) return false;
        }
        return true;
    }
};

} // namespace neural
} // namespace engines
} // namespace brain3
