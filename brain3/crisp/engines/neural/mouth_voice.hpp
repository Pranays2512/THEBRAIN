#pragma once
/**
 * brain3/crisp/engines/neural/mouth_voice.hpp
 *
 * VOICE POLICY — emotion → decoding parameters.
 *
 * The Mouth stops being a string formatter and becomes an expression of
 * internal state:
 *   - arousal  [0,1]  → temperature      calm ≈ 0.6 … excited ≈ 1.0
 *   - valence  [-1,1] → style-token bias positive valence boosts warm style
 *                     tokens; negative valence boosts guarded/hedged ones
 *                     (sparse additive logit bias, applied pre-softmax).
 *
 * The allow-set constraint stays hard regardless of mood: emotion may color
 * HOW the brain says something, never WHETHER the fact survives (form is
 * fuzzy, content is crisp).
 */

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "fuzzy/core/emotion.hpp"
#include "crisp/engines/neural/stamlat_transformer.hpp"

namespace brain3 {
namespace engines {
namespace neural {

struct VoicePolicy {
    float temperature = 0.8f;
    StamlatLM::LogitBias bias;   // token id → additive logit bonus
};

class VoiceMapper {
public:
    // Style lexicon as SURFACE strings; resolved against the live vocab so
    // ids stay correct across models. Unknown surfaces are ignored (they may
    // simply not be in this model's word table — char fallback still speaks).
    VoiceMapper(std::vector<std::string> warm_surfaces,
                std::vector<std::string> guarded_surfaces,
                float temp_floor = 0.6f,     // calm
                float temp_span  = 0.4f,     // added at arousal=1
                float max_bias   = 2.0f)     // logit units at |valence|=1
        : warm_(std::move(warm_surfaces)),
          guarded_(std::move(guarded_surfaces)),
          t_floor_(temp_floor), t_span_(temp_span), max_bias_(max_bias) {}

    VoicePolicy policy(const StamlatLM& lm, const brain2::EmotionState& e) const {
        VoicePolicy p;
        p.temperature = t_floor_ + t_span_ * e.arousal;

        const auto& lexicon = (e.valence >= 0.f) ? warm_ : guarded_;
        const float strength = max_bias_ * std::fabs(e.valence);
        if (strength > 0.f) {
            for (const auto& s : lexicon) {
                for (int id = lm.char_vocab_size(); id < lm.total_vocab_size(); ++id)
                    if (lm.token_surface(id) == s) { p.bias[id] += strength; break; }
            }
        }
        return p;
    }

private:
    std::vector<std::string> warm_, guarded_;
    float t_floor_, t_span_, max_bias_;
};

// Default lexicon matching the chat template vocabulary used across the
// stamlat demos/tests; extend freely per deployment. Warm covers both
// social and technical self-expression (the brain is proud of being
// cognitive/neural); guarded covers hedging and bare status reporting.
inline VoiceMapper default_voice_mapper() {
    return VoiceMapper(
        /*warm*/    {"friendly", "happy", "great", "positive", "welcome", "warm",
                     "optimal", "excellent", "cognitive", "neural"},
        /*guarded*/ {"unknown", "fallback", "ready", "processing", "state", "status"});
}

} // namespace neural
} // namespace engines
} // namespace brain3
