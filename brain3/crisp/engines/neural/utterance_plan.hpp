#pragma once
/**
 * brain3/crisp/engines/neural/utterance_plan.hpp
 *
 * UTTERANCE PLANS — the pre-verbal interface between memory and speech.
 *
 * The mouth holds NO content memories: weights encode only the mapping
 * (plan → fluent sentence). Content arrives exclusively as retrieved
 * surfaces inside the plan, and decoding is content-locked to those
 * surfaces — so what is spoken cannot outlive what is remembered.
 *
 *   plan  = { act, facts[], register }          (orchestrator retrieves)
 *   text  = linearize(plan)                     (token-friendly rendering)
 *   speak = constrained_decode(text, allowed=facts ∪ whitespace)
 */

#include <string>
#include <vector>
#include <algorithm>

#include "crisp/engines/neural/stamlat_transformer.hpp"

namespace brain3 {
namespace engines {
namespace neural {

struct UtterancePlan {
    std::string act;                       // dialogue act: greeting|identity|status
    std::vector<std::string> facts;        // retrieved surfaces (may contain distractors)
    std::string reg = "neutral";           // register: warm|neutral

    // Deterministic linearization; '|' separates multi-fact groups so word
    // tokenizer keeps surfaces intact.
    std::string linearize() const {
        std::string s = "<p> act " + act + " facts";
        for (const auto& f : facts) s += " " + f;
        s += " reg " + reg + " <r>";
        return s;
    }

    // Mechanical content-lock: the ONLY content tokens decodable are the
    // surfaces this plan carries (+ whitespace glue). Deleted memories are
    // absent here ⇒ structurally unspeakable.
    std::vector<int> content_lock_ids(const StamlatLM& lm) const {
        std::vector<int> ids;
        auto push_surface = [&](const std::string& want) {
            for (int id = lm.char_vocab_size(); id < lm.total_vocab_size(); ++id)
                if (lm.token_surface(id) == want) { ids.push_back(id); return; }
        };
        for (const auto& f : facts) push_surface(f);
        for (int id = 0; id < lm.char_vocab_size(); ++id) {
            const std::string c = lm.token_surface(id);
            if (c == " " || c == "\n") ids.push_back(id);
        }
        return ids;
    }
};

// ── Corpus inversion ─────────────────────────────────────────────────────────
// Legacy template corpora stored answers as bare slot sequences
// ("intent greeting style friendly"). Invert a line back into its plan,
// demonstrating that (plan, sentence) pairs recoverable from anything the
// orchestrator once generated from structure.
inline bool invert_template_line(const std::string& answer_line,
                                 const std::string& act,
                                 const std::string& reg,
                                 const std::vector<std::string>& class_vocab,
                                 UtterancePlan& out) {
    out.act = act;
    out.reg = reg;
    out.facts.clear();
    size_t i = 0;
    while (i < answer_line.size()) {
        if (answer_line[i] == ' ') { ++i; continue; }
        size_t j = i;
        while (j < answer_line.size() && answer_line[j] != ' ') ++j;
        std::string w = answer_line.substr(i, j - i);
        bool in_class = false;
        for (const auto& c : class_vocab) if (c == w) { in_class = true; break; }
        if (!in_class) return false;                    // not a pure slot line
        out.facts.push_back(w);
        i = j;
    }
    return !out.facts.empty();
}

} // namespace neural
} // namespace engines
} // namespace brain3
