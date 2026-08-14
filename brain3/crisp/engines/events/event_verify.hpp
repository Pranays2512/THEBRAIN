#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <optional>
#include <cmath>
#include <algorithm>
#include "event_form.hpp"

namespace brain2 {
namespace events {

constexpr const char* ADMIT = "admit";
constexpr const char* REJECT = "reject";
constexpr const char* ABSTAIN = "abstain";

class EventStore {
public:
    std::vector<Event> events;
    std::vector<Relation> relations;
    std::map<std::tuple<std::string, std::string, std::string, std::string>, int> by_key;

    bool contradicts(const Event& ev) const {
        auto it = by_key.find(ev.key());
        return it != by_key.end() && it->second != ev.polarity;
    }

    bool has(const Event& ev) const {
        auto it = by_key.find(ev.key());
        return it != by_key.end() && it->second == ev.polarity;
    }

    void commit(const Event& ev) {
        events.push_back(ev);
        by_key[ev.key()] = ev.polarity;
    }

    void add_relation(const Relation& r) {
        relations.push_back(r);
    }
};

using ConstraintSpec = std::map<std::string, std::set<std::string>>;

// -----------------------------------------------------------------------------
// Verification (event_verify.py)
// -----------------------------------------------------------------------------

inline std::optional<bool> check_types(
    const Event& ev, 
    std::function<std::set<std::string>(const std::string&)> type_of,
    const std::map<std::string, ConstraintSpec>& constraints,
    const std::optional<std::set<std::string>>& known_verbs = std::nullopt) {
    
    auto it = constraints.find(ev.verb);
    if (it == constraints.end()) {
        if (known_verbs.has_value() && known_verbs->count(ev.verb) == 0) return std::nullopt; // open-world, hold
        return true;
    }
    
    const auto& spec = it->second;
    bool unknown = false;
    
    for (const std::string& role : {"agent", "patient"}) {
        std::string tok = (role == "agent") ? ev.agent : ev.patient;
        if (tok.empty()) continue;
        
        auto spec_it = spec.find(role);
        if (spec_it == spec.end()) continue;
        
        const auto& allowed = spec_it->second;
        if (allowed.empty()) continue;
        
        std::set<std::string> types = type_of(tok);
        if (types.empty()) {
            unknown = true;
            continue;
        }
        
        bool intersect = false;
        for (const auto& t : types) {
            if (allowed.count(t)) { intersect = true; break; }
        }
        
        if (!intersect) return false;
    }
    
    if (unknown) return std::nullopt;
    return true;
}

inline std::string classify(
    const Event& ev, 
    const EventStore& store,
    std::function<std::set<std::string>(const std::string&)> type_of,
    const std::map<std::string, ConstraintSpec>& constraints,
    const std::optional<std::set<std::string>>& known_verbs = std::nullopt) {
    
    if (store.contradicts(ev)) return REJECT;
    auto t = check_types(ev, type_of, constraints, known_verbs);
    if (t == false) return REJECT;
    if (!t.has_value()) return ABSTAIN;
    return ADMIT;
}

inline std::string admit_event(
    const Event& ev, 
    EventStore& store,
    std::function<std::set<std::string>(const std::string&)> type_of,
    const std::map<std::string, ConstraintSpec>& constraints,
    const std::optional<std::set<std::string>>& known_verbs = std::nullopt) {
    
    std::string d = classify(ev, store, type_of, constraints, known_verbs);
    if (d == ADMIT && !store.has(ev)) {
        store.commit(ev);
    }
    return d;
}

// -----------------------------------------------------------------------------
// Learning (verb_learn.py)
// -----------------------------------------------------------------------------

const std::set<std::string> ROOTS = {"living_thing", "thing", "entity", "object", "concept"};

inline std::set<std::string> generalize(const std::vector<std::set<std::string>>& closures, float frac = 1.0f) {
    std::vector<std::set<std::string>> typed;
    for (const auto& c : closures) {
        if (!c.empty()) typed.push_back(c);
    }
    if (typed.empty()) return {};
    
    std::map<std::string, int> cnt;
    for (const auto& s : typed) {
        for (const auto& t : s) cnt[t]++;
    }
    
    int need = (frac >= 1.0f) ? typed.size() : std::max(1, (int)std::ceil(frac * typed.size()));
    
    std::set<std::string> res;
    for (const auto& kv : cnt) {
        if (kv.second >= need && ROOTS.count(kv.first) == 0) {
            res.insert(kv.first);
        }
    }
    return res;
}

class VerbLearner {
public:
    std::function<std::set<std::string>(const std::string&)> type_of;
    int promote_at = 2;
    float frac = 1.0f;
    
    // verb -> vector of (agent_types, patient_types)
    std::map<std::string, std::vector<std::pair<std::set<std::string>, std::set<std::string>>>> obs;
    std::map<std::string, ConstraintSpec> constraints;
    
    VerbLearner(std::function<std::set<std::string>(const std::string&)> t_fn, int p_at = 2, float f = 1.0f)
        : type_of(t_fn), promote_at(p_at), frac(f) {}
        
    void observe(const Event& ev) {
        std::set<std::string> a = ev.agent.empty() ? std::set<std::string>() : type_of(ev.agent);
        std::set<std::string> p = ev.patient.empty() ? std::set<std::string>() : type_of(ev.patient);
        obs[ev.verb].push_back({a, p});
    }
    
    std::optional<ConstraintSpec> conjecture(const std::vector<std::pair<std::set<std::string>, std::set<std::string>>>& uses) {
        std::vector<std::set<std::string>> a_uses, p_uses;
        for (const auto& u : uses) {
            a_uses.push_back(u.first);
            p_uses.push_back(u.second);
        }
        auto a_gen = generalize(a_uses, frac);
        auto p_gen = generalize(p_uses, frac);
        
        ConstraintSpec spec;
        if (!a_gen.empty()) spec["agent"] = a_gen;
        if (!p_gen.empty()) spec["patient"] = p_gen;
        if (spec.empty()) return std::nullopt;
        return spec;
    }
    
    bool satisfies(const ConstraintSpec& spec, const std::vector<std::pair<std::set<std::string>, std::set<std::string>>>& uses) {
        for (const auto& u : uses) {
            if (spec.count("agent") && !u.first.empty()) {
                bool inter = false;
                for (const auto& t : u.first) {
                    if (spec.at("agent").count(t)) { inter = true; break; }
                }
                if (!inter) return false;
            }
            if (spec.count("patient") && !u.second.empty()) {
                bool inter = false;
                for (const auto& t : u.second) {
                    if (spec.at("patient").count(t)) { inter = true; break; }
                }
                if (!inter) return false;
            }
        }
        return true;
    }
    
    std::optional<ConstraintSpec> learn(const std::string& verb, const std::vector<std::pair<std::set<std::string>, std::set<std::string>>>& holdout = {}) {
        if (obs.count(verb) == 0 || obs[verb].size() < promote_at) return std::nullopt;
        
        auto spec = conjecture(obs[verb]);
        if (!spec.has_value()) return std::nullopt;
        
        if (!satisfies(*spec, holdout)) return std::nullopt;
        
        constraints[verb] = *spec;
        return spec;
    }
    
    std::set<std::string> acquire(const std::map<std::string, std::vector<std::pair<std::set<std::string>, std::set<std::string>>>>& holdouts = {}) {
        std::set<std::string> learned;
        for (const auto& kv : obs) {
            std::string verb = kv.first;
            if (constraints.count(verb) == 0) {
                std::vector<std::pair<std::set<std::string>, std::set<std::string>>> h;
                if (holdouts.count(verb)) h = holdouts.at(verb);
                if (learn(verb, h).has_value()) learned.insert(verb);
            }
        }
        return learned;
    }
};

} // namespace events
} // namespace brain2
