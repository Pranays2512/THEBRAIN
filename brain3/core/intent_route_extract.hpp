#pragma once
/**
 * brain3/core/intent_route_extract.hpp
 *
 * Bridges the learned IntentRouter verdict to deterministic slot extraction.
 * The regexes here MIRROR the corresponding branches in
 * master_orchestrator.hpp's parse_intent_to_bql — eval suite guards drift.
 * On any extraction failure the caller must fall back to the legacy chain.
 */
#include <regex>
#include <string>

#include "core/intent_router.hpp"

namespace brain3 {
namespace core {

// Pronouns/personal turns that must NEVER become knowledge lookups —
// they belong to the native mouth (chat path).
inline bool is_personal_subject(const std::string& subject) {
    static const char* kStop[] = {
        "you", "your", "ur", "u", "it", "this", "that", "i", "we",
        "me", "myself", "yourself", "him", "her", "them",
    };
    for (const auto* s : kStop) if (subject == s) return true;
    return false;
}

// Returns true and fills `out` when the router family yields a clean BQL.
inline bool route_extract(const std::string& clean_text,
                          const std::string& family,
                          std::string& out) {
    if (family == "WHAT_IF") {
        static const std::regex r(
            R"((?:what if|counterfactual:?|suppose|what happens if|imagine|simulate scenario where)\s+([\w\s]+?)\s+(?:causes|leads to|results in|affects|drives|produces|is)\s+([\w\s]+))",
            std::regex_constants::icase);
        std::smatch m;
        if (std::regex_search(clean_text, m, r)) {
            auto trim = [](std::string s) {
                s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                    [](unsigned char c){ return !std::isspace(c); }));
                s.erase(std::find_if(s.rbegin(), s.rend(),
                    [](unsigned char c){ return !std::isspace(c); }).base(), s.end());
                return s;
            };
            out = "WHAT_IF " + trim(m[1].str()) + " " + trim(m[2].str());
            return true;
        }
    }
    else if (family == "TEACH") {
        static const std::regex r(
            R"((?:teach(?: that)?|remember|learn that|store that|note that|commit to memory|record|save fact|add knowledge|absorb|keep in mind|ingest fact)\s+([\w\s]+?)\s+(?:is a|is an|has|can|causes)\s+([\w\s]+))",
            std::regex_constants::icase);
        std::smatch m;
        if (std::regex_search(clean_text, m, r)) {
            out = "TEACH " + m[1].str() + " is_a " + m[2].str();
            return true;
        }
    }
    else if (family == "LOOKUP") {
        static const std::regex what_is(
            R"((?:what is|who is|what are|tell me about|look up|query|info on|describe|details of|definition of|meaning of|facts about|search for|find)\s+(?:a\s+|an\s+|the\s+)?([\w]+))",
            std::regex_constants::icase);
        std::smatch m;
        if (std::regex_search(clean_text, m, what_is)) {
            std::string subj = m[1].str();
            std::transform(subj.begin(), subj.end(), subj.begin(), ::tolower);
            if (!is_personal_subject(subj)) {
                out = "LOOKUP " + m[1].str() + " is_a";
                return true;
            }
        }
    }
    else if (family == "ANALOGY") {
        static const std::regex r(
            R"((?:compare|analogy between|map|isomorphism between|draw parallel between|align concepts|relate structurally|transfer structure from|correspondence)\s+([\w]+)\s+(?:to|and|with|onto|versus)\s+([\w]+))",
            std::regex_constants::icase);
        std::smatch m;
        if (std::regex_search(clean_text, m, r)) {
            out = "ANALOGY " + m[1].str() + " TO " + m[2].str();
            return true;
        }
    }
    else if (family == "REFUTE") {
        static const std::regex r(
            R"((?:refute|disprove|challenge|find counterexample to|attack premise|show|falsify|is it true that all)\s+([\w\s]+))",
            std::regex_constants::icase);
        std::smatch m;
        if (std::regex_search(clean_text, m, r)) {
            out = "REFUTE " + m[1].str();
            return true;
        }
    }
    else if (family == "EXPLAIN") {
        static const std::regex r(R"((?:how to|explain|plan for|outline|strategy for|walk me through|brief on|summarize approach to)\s+([\w\s]+))",
                                  std::regex_constants::icase);
        std::smatch m;
        if (std::regex_search(clean_text, m, r)) {
            out = "EXPLAIN " + m[1].str();
            return true;
        }
    }
    return false;   // fall through to legacy parser
}

} // namespace core
} // namespace brain3
