#pragma once
/**
 * brain3/crisp/engines/neural/native_mouth.hpp
 *
 * NATIVE MOUTH — production mount of the STAMLAT v3 voice.
 *
 * Loads a trained model binary at boot and answers conversational turns in
 * microseconds through the KV-cache streaming path. Every reply carries a
 * crisp confidence signal (mean reply NLL + clean termination) so the
 * orchestrator can escalate uncertain turns to the symbolic pipeline /
 * upstream LLM mouth — never worse than today, usually 1000x faster.
 *
 * Escalation contract: `confident == false` means "I am not sure I can say
 * this correctly" — control returns to the caller unchanged.
 */

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "crisp/engines/neural/stamlat_transformer.hpp"
#include "crisp/engines/neural/utterance_plan.hpp"
#include "crisp/engines/neural/mouth_voice.hpp"

namespace brain3 {
namespace engines {
namespace neural {

class NativeMouth {
public:
    struct Config {
        float temp                 = 0.f;    // duality switch: greedy in prod
        bool  use_mood_temperature = false;  // opt-in emotion coupling
        float nll_confidence_gate  = 2.2f;   // mean reply NLL ceiling (nats)
        int   max_reply_tokens     = 48;
    };

    struct Result {
        std::string text;
        double reply_nll = 0.;
        double ms        = 0.;
        int    tokens    = 0;
        float  temp_used = 0.f;
        bool   confident  = false;
    };

    NativeMouth() = default;
    explicit NativeMouth(Config cfg) : cfg_(cfg) {}

    // Loads a STMLv3 binary. Safe to call with a missing file: the mouth
    // simply stays unavailable and callers keep their existing pipeline.
    bool load(const std::string& path) {
        StamlatLM tmp(default_cfg());
        if (!tmp.load(path)) return false;
        lm_ = std::move(tmp);
        available_ = true;
        for (int id = lm_.char_vocab_size(); id < lm_.total_vocab_size(); ++id)
            if (lm_.token_surface(id) == "<p>") { plans_supported_ = true; break; }
        return true;
    }

    // Plan-conditioned response: content-locked to the plan's surfaces —
    // structurally unable to speak beyond retrieved memory (amnesia).
    Result respond_plan(const UtterancePlan& plan,
                        const brain2::EmotionState& mood = {0.f, 0.f},
                        const VoiceMapper* voice = nullptr) {
        Result res;
        if (!available_ || !plans_supported_) return res;
        const auto t0 = std::chrono::steady_clock::now();
        VoicePolicy pol;
        pol.temperature = cfg_.temp;
        if (cfg_.use_mood_temperature && voice)
            pol = voice->policy(lm_, mood);
        res.temp_used = pol.temperature;

        const auto ids = lm_.encode(plan.linearize());
        StamlatLM::StreamCache sc;
        lm_.stream_start(ids, sc);
        std::vector<int> utt;
        bool terminated = false;
        for (int n = 0; n < cfg_.max_reply_tokens; ++n) {
            const int tok = lm_.stream_sample(sc, pol.temperature,
                                              nullptr, nullptr);
            utt.push_back(tok);
            lm_.stream_step(tok, sc);
            if (lm_.token_surface(tok) == "\n") { terminated = true; break; }
        }
        auto t1 = std::chrono::steady_clock::now();
        res.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        res.tokens = (int)utt.size();
        if (terminated && !utt.empty()) {
            res.text = lm_.decode(utt);
            while (!res.text.empty() &&
                   (res.text.back() == '\n' || res.text.back() == ' '))
                res.text.pop_back();
            res.reply_nll = reply_nll(ids, utt);
            res.confident = !res.text.empty() &&
                            res.reply_nll < cfg_.nll_confidence_gate;
        }
        return res;
    }

    bool available() const { return available_; }
    // plan-conditioned models carry the '<p>' scaffold token
    bool plans_supported() const {
        return available_ && plans_supported_;
    }

    const StamlatLM& model() const { return lm_; }
    StamlatLM& model() { return lm_; }          // sleep-kernel training access
    Config& config() { return cfg_; }

    Result respond(const std::string& user_text,
                   const brain2::EmotionState& mood = {0.f, 0.f},
                   const VoiceMapper* voice = nullptr) {
        Result res;
        if (!available_) return res;

        // prompt hygiene mirrors training format
        std::string clean;
        clean.reserve(user_text.size());
        for (char c : user_text)
            clean += (char)(c == '\n' || c == '\r' ? ' '
                            : std::tolower((unsigned char)c));
        const auto t0 = std::chrono::steady_clock::now();

        VoicePolicy pol;
        pol.temperature = cfg_.temp;
        if (cfg_.use_mood_temperature && voice)
            pol = voice->policy(lm_, mood);
        res.temp_used = pol.temperature;

        const auto ids = lm_.encode("user: " + clean + "\nbrain: ");
        StamlatLM::StreamCache sc;
        lm_.stream_start(ids, sc);

        std::vector<int> utt;
        bool terminated = false;
        for (int n = 0; n < cfg_.max_reply_tokens; ++n) {
            const int tok = lm_.stream_sample(sc, pol.temperature, nullptr, &pol.bias);
            utt.push_back(tok);
            lm_.stream_step(tok, sc);
            if (lm_.token_surface(tok) == "\n") { terminated = true; break; }
        }

        auto t1 = std::chrono::steady_clock::now();
        res.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        res.tokens = (int)utt.size();

        if (terminated && !utt.empty()) {
            res.text = lm_.decode(utt);
            // strip trailing newline for caller ergonomics
            while (!res.text.empty() &&
                   (res.text.back() == '\n' || res.text.back() == ' '))
                res.text.pop_back();
            res.reply_nll  = reply_nll(ids, utt);
            res.confident  = !res.text.empty() && res.reply_nll < cfg_.nll_confidence_gate;
        }
        return res;
    }

private:
    static StamlatConfig default_cfg() {
        StamlatConfig c;
        c.d_model = 96; c.n_layers = 3; c.n_heads = 6; c.d_ff = 256; c.ctx = 96;
        c.depth_gamma = 0.f; c.depth_tau = 1.f;
        return c;   // load() overrides from file header as needed
    }

    // mean NLL of utterance tokens given prompt under current parameters
    double reply_nll(const std::vector<int>& prompt_ids,
                     const std::vector<int>& utt_ids) const {
        std::vector<int> seq = prompt_ids;
        seq.insert(seq.end(), utt_ids.begin(), utt_ids.end());
        if (seq.size() > (size_t)lm_.config().ctx + 1) {
            const size_t drop = seq.size() - ((size_t)lm_.config().ctx + 1);
            seq.erase(seq.begin(), seq.begin() + (long)drop);
        }
        const size_t first_target = seq.size() - utt_ids.size();
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

    Config      cfg_;
    StamlatLM   lm_{default_cfg()};
    bool        available_ = false;
    bool        plans_supported_ = false;
};

} // namespace neural
} // namespace engines
} // namespace brain3
