#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <regex>
#include <functional>
#include <optional>
#include <algorithm>
#include "event_form.hpp"

namespace brain2 {
namespace events {

const std::set<std::string> PRONOUNS = {"it", "they", "he", "she", "him", "her", "them", "its", "their"};

const std::map<std::string, std::pair<std::string, std::string>> CONNECTIVES = {
    {"because",   {CAUSE, "bwd"}},
    {"so",        {CAUSE, "fwd"}},
    {"therefore", {CAUSE, "fwd"}},
    {"but",       {CONTRAST, "fwd"}},
    {"however",   {CONTRAST, "fwd"}},
    {"then",      {SEQUENCE, "fwd"}},
    {"after",     {SEQUENCE, "fwd"}}
};

const std::map<std::string, std::string> CONTRACTIONS = {
    {"didn't", "did not"}, {"don't", "do not"}, {"doesn't", "does not"}, {"won't", "will not"},
    {"can't", "can not"}, {"cannot", "can not"}, {"isn't", "is not"}, {"wasn't", "was not"},
    {"weren't", "were not"}, {"hasn't", "has not"}, {"haven't", "have not"}, {"aren't", "are not"},
    {"couldn't", "could not"}, {"wouldn't", "would not"}, {"shouldn't", "should not"}, {"never", "not"}
};

const std::set<std::string> NEG_WORDS = {"not", "no"};
const std::set<std::string> PAST_AUX = {"did", "was", "were", "had"};
const std::set<std::string> FUTURE_AUX = {"will", "shall"};

const std::map<std::string, std::string> IRREGULAR_VERBS = {
    {"ate", "eat"}, {"ran", "run"}, {"went", "go"}, {"saw", "see"}, {"made", "make"},
    {"drank", "drink"}, {"flew", "fly"}, {"caught", "catch"}, {"chased", "chase"},
    {"liked", "like"}, {"moved", "move"}, {"was", "be"}, {"were", "be"}
};

const std::set<std::string> STOP_WORDS = {
    "the", "a", "an", "this", "that", "these", "those", "did", "do", "does", "not", "no",
    "was", "were", "is", "are", "am", "be", "been", "will", "shall", "had", "has", "have",
    "to", "then", "so", "but", "because", "and", "of", "at"
};

class ContextStack {
public:
    std::vector<std::pair<std::string, std::string>> entities; // (token, type)
    std::vector<int> events;
    std::function<std::string(const std::string&)> type_of_fn;

    ContextStack(std::function<std::string(const std::string&)> type_fn = [](const std::string&){ return ""; })
        : type_of_fn(type_fn) {}

    void push_entity(const std::string& token) {
        entities.push_back({token, type_of_fn(token)});
    }

    void push_event(int eid) {
        events.push_back(eid);
    }

    std::optional<std::string> resolve(const std::string& pronoun, const std::string& want_type = "") {
        if (PRONOUNS.find(pronoun) == PRONOUNS.end()) return std::nullopt;
        
        for (auto it = entities.rbegin(); it != entities.rend(); ++it) {
            if (want_type.empty() || it->second.empty() || it->second == want_type) {
                return it->first;
            }
        }
        return std::nullopt;
    }
};

inline std::optional<std::pair<std::string, std::string>> connective_of(const std::string& token) {
    auto it = CONNECTIVES.find(token);
    if (it != CONNECTIVES.end()) return it->second;
    return std::nullopt;
}

inline std::optional<Relation> link_events(const std::vector<std::string>& tokens, int prev_eid, int cur_eid) {
    for (const auto& tok : tokens) {
        auto c = connective_of(tok);
        if (c) {
            if (c->second == "fwd") return Relation(c->first, prev_eid, cur_eid);
            return Relation(c->first, cur_eid, prev_eid);
        }
    }
    return std::nullopt;
}

// Emulates engines.store.parse_template.normalize
inline std::string normalize(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    // basic stemming for 'ed' or 's' could go here if needed
    if (s.length() > 3 && s.substr(s.length() - 2) == "ed") s = s.substr(0, s.length() - 2);
    else if (s.length() > 2 && s.back() == 's' && s[s.length()-2] != 's') s = s.substr(0, s.length() - 1);
    return s;
}

inline std::string lemma(const std::string& tok) {
    auto it = IRREGULAR_VERBS.find(tok);
    if (it != IRREGULAR_VERBS.end()) return it->second;
    return normalize(tok);
}

inline std::optional<std::string> nearest_entity(const std::vector<std::string>& seq, 
                                                 const std::set<std::string>& entities, 
                                                 std::function<bool(const std::string&)> type_of = nullptr) {
    for (const auto& t : seq) {
        if (entities.count(t) || PRONOUNS.count(t) || (type_of && type_of(t))) return t;
    }
    for (const auto& t : seq) {
        if (STOP_WORDS.count(t) == 0) return t;
    }
    return std::nullopt;
}

inline std::optional<Event> parse_event(std::string text, 
                                        const std::set<std::string>& entities, 
                                        const std::set<std::string>& verbs, 
                                        std::function<bool(const std::string&)> type_of = nullptr) {
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);
    
    for (const auto& [c, e] : CONTRACTIONS) {
        size_t pos = 0;
        while ((pos = text.find(c, pos)) != std::string::npos) {
            text.replace(pos, c.length(), e);
            pos += e.length();
        }
    }

    std::regex word_regex("[a-z_]+");
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), word_regex);
    auto words_end = std::sregex_iterator();

    std::vector<std::string> raw;
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        raw.push_back(i->str());
    }

    int polarity = POS;
    std::string tense = "present";

    for (const auto& t : raw) {
        if (NEG_WORDS.count(t)) polarity = NEG;
        if (PAST_AUX.count(t)) tense = "past";
        else if (FUTURE_AUX.count(t) && tense != "past") tense = "future";
    }

    int vi = -1;
    std::string verb = "";

    // 1. trusted lemma
    for (size_t i = 0; i < raw.size(); i++) {
        std::string lem = lemma(raw[i]);
        if (verbs.count(lem) || verbs.count(raw[i])) {
            vi = i;
            verb = lem;
            break;
        }
    }

    // 2. positional
    if (vi == -1) {
        std::vector<int> content;
        for (size_t i = 0; i < raw.size(); i++) {
            if (!STOP_WORDS.count(raw[i]) && !NEG_WORDS.count(raw[i])) content.push_back(i);
        }
        if (content.size() < 2) return std::nullopt;
        vi = content[1];
        verb = lemma(raw[vi]);
    }

    if (tense == "present") {
        if (IRREGULAR_VERBS.count(raw[vi]) || (raw[vi].length() > 2 && raw[vi].substr(raw[vi].length()-2) == "ed")) {
            tense = "past";
        }
    }

    std::vector<std::string> before_verb;
    for (int i = vi - 1; i >= 0; i--) before_verb.push_back(raw[i]);
    
    std::vector<std::string> after_verb;
    for (size_t i = vi + 1; i < raw.size(); i++) after_verb.push_back(raw[i]);

    auto agent = nearest_entity(before_verb, entities, type_of);
    auto patient = nearest_entity(after_verb, entities, type_of);

    return Event(verb, agent.value_or(""), patient.value_or(""), tense, polarity);
}

inline bool verb_trusted(const std::optional<Event>& ev, const std::set<std::string>& verbs) {
    return ev.has_value() && verbs.count(ev->verb);
}

} // namespace events
} // namespace brain2
