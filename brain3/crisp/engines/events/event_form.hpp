#pragma once
#include <string>
#include <tuple>
#include <stdexcept>
#include <optional>
#include <set>

namespace brain2 {
namespace events {

constexpr int POS = 1;
constexpr int NEG = -1;

constexpr const char* CAUSE = "CAUSE";
constexpr const char* CONTRAST = "CONTRAST";
constexpr const char* SEQUENCE = "SEQUENCE";

inline bool is_valid_relation_kind(const std::string& kind) {
    return kind == CAUSE || kind == CONTRAST || kind == SEQUENCE;
}

struct Event {
    std::string verb;
    std::string agent;
    std::string patient;
    std::string time; // coarse tense/marker: past|present|future|None
    int polarity = POS;
    int id = 0;

    Event() = default;
    Event(std::string v, std::string a = "", std::string p = "", std::string t = "", int pol = POS, int i = 0)
        : verb(std::move(v)), agent(std::move(a)), patient(std::move(p)), time(std::move(t)), polarity(pol), id(i) {}

    auto key() const {
        return std::make_tuple(verb, agent, patient, time);
    }

    Event negated() const {
        return Event(verb, agent, patient, time, -polarity, id);
    }

    bool operator==(const Event& o) const {
        return verb == o.verb && agent == o.agent && patient == o.patient && 
               time == o.time && polarity == o.polarity && id == o.id;
    }
};

struct Relation {
    std::string kind;
    int e1;
    int e2;

    Relation() = default;
    Relation(std::string k, int e_1, int e_2) : kind(std::move(k)), e1(e_1), e2(e_2) {
        if (!is_valid_relation_kind(kind)) {
            throw std::invalid_argument("unknown relation kind " + kind);
        }
    }
};

inline Event fact_as_event(const std::string& obj, const std::string& prop, const std::string& value, int eid = 0) {
    return Event(prop, obj, value, "present", POS, eid);
}

inline std::optional<std::tuple<std::string, std::string, std::string>> event_as_fact(const Event& ev) {
    if (ev.polarity != POS) {
        return std::nullopt;
    }
    return std::make_tuple(ev.agent, ev.verb, ev.patient);
}

} // namespace events
} // namespace brain2
