#pragma once
/**
 * brain3/crisp/engines/neural/native_reader.hpp
 *
 * THE BRAIN — NATIVE READER (the EYES)
 *
 * A StamlatLM trained as a parser: english sentence -> "(s r o)" triple.
 * Same organ family as NativeMouth (the MOUTH) — one transformer
 * architecture, two sensory-motor roles, selected purely by prompt.
 *
 * Discipline identical to the mouth:
 *   - deterministic decoding (temperature 0)
 *   - mean reply NLL must fall under the confidence gate, else the parse
 *     is REFUSED (confident=false) and callers fall back to symbolic paths.
 *   - the reader PROPOSES; crisp gates still dispose (quarantine etc.)
 *
 * Trained via train_native_mouth --corpus reader_corpus.txt
 * Default artifact: stamlat_reader.bin
 */

#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include "crisp/engines/neural/stamlat_transformer.hpp"

namespace brain3 {
namespace engines {
namespace neural {

class NativeReader {
public:
    struct Parsed {
        std::string subj, rel, obj;
        double reply_nll = 0.0;
        double ms = 0.0;
        bool confident = false;
    };

    bool load(const std::string& path) {
        StamlatLM tmp(default_cfg());
        if (!tmp.load(path)) return false;
        lm_ = std::move(tmp);
        available_ = true;
        return true;
    }

    bool available() const { return available_; }
    const StamlatLM& model() const { return lm_; }
    StamlatLM& model() { return lm_; }          // sleep-kernel training access

    // English sentence -> subject/relation/object, confidence-gated.
    Parsed parse(const std::string& sentence) {
        Parsed out;
        if (!available_) return out;

        // prompt hygiene mirrors training format
        std::string clean;
        clean.reserve(sentence.size());
        for (char c : sentence)
            clean += (char)(c == '\n' || c == '\r' ? ' '
                            : std::tolower((unsigned char)c));
        while (!clean.empty() && clean.back() == ' ') clean.pop_back();
        if (clean.size() > 160) return out;

        const auto t0 = std::chrono::steady_clock::now();
        const auto ids = lm_.encode("read: " + clean + "\ntriple: ");
        StamlatLM::StreamCache sc;
        lm_.stream_start(ids, sc);

        std::vector<int> utt;
        bool terminated = false;
        for (int n = 0; n < 16; ++n) {
            const int tok = lm_.stream_sample(sc, 0.f, nullptr, nullptr);
            utt.push_back(tok);
            lm_.stream_step(tok, sc);
            if (lm_.token_surface(tok) == "\n") { terminated = true; break; }
        }
        auto t1 = std::chrono::steady_clock::now();
        out.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!terminated || utt.empty()) return out;

        std::string text = lm_.decode(utt);
        while (!text.empty() && (text.back() == '\n' || text.back() == ' '))
            text.pop_back();

        // expect exactly: "<subj> <rel> <obj>"
        std::vector<std::string> parts;
        {
            std::istringstream iss(text);
            std::string w;
            while (iss >> w) parts.push_back(w);
        }
        out.reply_nll = reply_nll(ids, utt);
        out.confident = parts.size() == 3 &&
                        out.reply_nll < cfg_.nll_confidence_gate;
        if (out.confident) {
            out.subj = parts[0];
            out.rel  = parts[1];
            out.obj  = parts[2];
        }
        return out;
    }

private:
    StamlatLM lm_{default_cfg()};
    bool available_ = false;

    struct Config {
        double nll_confidence_gate = 2.2;   // same gate as the mouth
    } cfg_;

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
};

} // namespace neural
} // namespace engines
} // namespace brain3
