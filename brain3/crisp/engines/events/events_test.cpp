#include <iostream>
#include <iomanip>
#include "event_form.hpp"
#include "event_parse.hpp"
#include "analogy.hpp"
#include "event_verify.hpp"

using namespace brain2::events;

void test_analogy_struct() {
    std::cout << "=== analogy_struct — map two domains by STRUCTURE, not shared words ===\n\n";
    std::vector<Fact> solar = {
        {"sun", "heavier", "planet"}, 
        {"planet", "revolves", "sun"},
        {"sun", "pulls", "planet"}
    };
    std::vector<Fact> atom = {
        {"nucleus", "massive", "electron"}, 
        {"electron", "circles", "nucleus"}
    };

    auto res = align_greedy(solar, atom, 10, 42); // heuristic align
    std::cout << "  structural alignment (induced, no shared vocab):\n";
    for (const auto& kv : res.emap) std::cout << "    " << kv.first << " <-> " << kv.second << "\n";
    std::cout << "    score: " << res.score << "\n";

    auto preds = transfer_structural(solar, atom, res.emap, res.relmap);
    std::cout << "\n  TRANSFER — predicted from source:\n";
    for (const auto& p : preds) {
        std::cout << "    " << std::get<0>(p.first) << " --" << std::get<1>(p.first) << "--> " 
                  << std::get<2>(p.first) << "   (" << (p.second ? "NEW relation" : "aligned") << ")\n";
    }
}

void test_analogy_engine() {
    std::cout << "\n=== analogy_engine — structure mapping (shared vocab) ===\n\n";
    std::vector<Fact> water = {
        {"pump", "increases", "flow"},
        {"pipe", "resists", "flow"},
        {"flow", "depends_on", "pressure"},
        {"pump", "raises", "pressure"}
    };
    std::vector<Fact> elec = {
        {"battery", "increases", "current"},
        {"resistor", "resists", "current"},
        {"current", "depends_on", "voltage"}
    };
    
    auto res = map_domains_shared(water, elec);
    std::cout << "Correspondence:\n";
    for (const auto& kv : res.mapping) std::cout << "  " << kv.first << " ~ " << kv.second << "\n";
    
    std::cout << "Analogical predictions:\n";
    for (const auto& t : res.transfers) {
        auto pred = std::get<0>(t);
        auto src = std::get<1>(t);
        std::cout << "  " << std::get<0>(pred) << " " << std::get<1>(pred) << " " << std::get<2>(pred) 
                  << "     (by analogy with " << std::get<0>(src) << " " << std::get<1>(src) << " " << std::get<2>(src) << ")\n";
    }
}

void test_event_parse() {
    std::cout << "\n=== event_parse — extracting events from text ===\n\n";
    std::set<std::string> entities = {"john", "apple", "he", "it"};
    std::set<std::string> verbs = {"eat", "drop"};

    auto type_of = [&](const std::string& t){ return entities.count(t) > 0; }; 
    
    std::string text = "John didn't eat the apple";
    auto ev = parse_event(text, entities, verbs, type_of);
    if (ev) {
        std::cout << "Text: " << text << "\n";
        std::cout << "Event: verb=" << ev->verb << " agent=" << ev->agent << " patient=" << ev->patient 
                  << " tense=" << ev->time << " polarity=" << ev->polarity << "\n";
    }
}

void test_event_verify() {
    std::cout << "\n=== verb_learn — acquiring selectional constraints ===\n\n";
    auto type_of = [](const std::string& t) -> std::set<std::string> {
        if (t == "apple" || t == "sandwich") return {"food", "object"};
        if (t == "john" || t == "mary") return {"person", "living_thing"};
        return {};
    };

    VerbLearner learner(type_of, 2, 1.0f);
    
    // observe two uses of 'eat'
    learner.observe(Event("eat", "john", "apple"));
    learner.observe(Event("eat", "mary", "sandwich"));
    
    // acquire
    auto learned = learner.acquire();
    std::cout << "Learned verbs: ";
    for (const auto& v : learned) std::cout << v << " ";
    std::cout << "\n";
    
    if (learner.constraints.count("eat")) {
        std::cout << "Constraint for 'eat':\n";
        for (const auto& r : learner.constraints["eat"]["agent"]) std::cout << "  agent must be: " << r << "\n";
        for (const auto& r : learner.constraints["eat"]["patient"]) std::cout << "  patient must be: " << r << "\n";
    }
}

int main() {
    test_analogy_struct();
    test_analogy_engine();
    test_event_parse();
    test_event_verify();
    return 0;
}
