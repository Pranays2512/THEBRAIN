#include <iostream>
#include <iomanip>
#include "knowledge_engine.hpp"
#include "semantic_memory.hpp"
#include "concept_blend.hpp"

using namespace brain2::knowledge;

void test_knowledge_engine() {
    std::cout << "=== knowledge_engine — symbolic multi-hop over vectors ===\n\n";
    KnowledgeEngine kb(32);
    kb.learn("alice", "manages", "bob");
    kb.learn("bob", "manages", "carol");
    kb.learn("carol", "manages", "dave");
    
    auto res = kb.ask("alice", "manages", 3);
    std::cout << "ask(\"alice\", \"manages\", hops=3) -> " << res.first << " (conf " << res.second << ")\n";
    std::cout << "explain: " << kb.explain("alice", "manages") << "\n";
    std::cout << "knows(alice manages dave)? " << (kb.knows("alice", "manages", "dave") ? "true" : "false") << "\n";
}

void test_semantic_memory() {
    std::cout << "\n=== semantic_memory — generalization via embeddings ===\n\n";
    SemanticMemory sm(4);
    
    // Very tiny dummy glove map
    std::map<std::string, Vector> dummy_glove = {
        {"automobile", {1.0, 0.8, 0.2, 0.0}},
        {"car",        {0.9, 0.85, 0.2, 0.0}},
        {"engine",     {0.5, 0.1, 0.9, 0.8}},
        {"dog",        {0.1, 0.0, 0.5, 0.5}},
        {"tail",       {0.2, 0.0, 0.8, 0.6}}
    };
    sm.load_glove(dummy_glove);
    
    sm.learn("automobile", "has", "engine");
    sm.learn("dog", "has", "tail");
    
    // Query with 'car' which was never explicitly learned
    auto res = sm.ask("car", "has");
    std::cout << "ask(\"car\", \"has\") -> " << res.first << " (conf " << res.second << ")\n";
    
    auto sims = sm.similar("car", 2);
    std::cout << "similar('car') -> ";
    for (auto s : sims) std::cout << s << " ";
    std::cout << "\n";
}

void test_concept_blend() {
    std::cout << "\n=== concept_blend — invent novelty outside categories ===\n\n";
    std::map<std::string, Vector> concepts = {
        {"bird",    {1, 1, 0, 0}},
        {"fish",    {0, 0, 1, 1}},
        {"reptile", {0, 1, 0, 0}},
        {"insect",  {1, 0, 0, 0}}
    };
    float radius = 1.0f;
    
    auto res = propose("bird", "fish", concepts, radius, "salient");
    std::cout << "propose(bird, fish) -> [";
    for (float x : res.vector) std::cout << x << " ";
    std::cout << "]\n";
    std::cout << "novel? " << (res.novel ? "Yes" : "No") << " (nearest: " << res.nearest << ", dist: " << res.distance << ")\n";
}

int main() {
    test_knowledge_engine();
    test_semantic_memory();
    test_concept_blend();
    return 0;
}
