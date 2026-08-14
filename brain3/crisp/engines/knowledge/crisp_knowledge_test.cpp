#include <iostream>
#include <iomanip>
#include "fact_extractor.hpp"
#include "knowledge_base.hpp"
#include "concept_memory.hpp"
#include "semantic_depth.hpp"

using namespace brain2::knowledge;

void test_fact_extractor() {
    std::cout << "=== fact_extractor — learn from text ===\n\n";
    FactExtractor fe;
    std::string text = "An apple is a fruit. It is red. It grows on a tree. It has seeds. A dog is an animal. It has a tail.";
    auto triples = fe.extract(text);
    for (const auto& t : triples) {
        std::cout << "  (" << std::get<0>(t) << ", " << std::get<1>(t) << ", " << std::get<2>(t) << ")\n";
    }
}

void test_knowledge_base() {
    std::cout << "\n=== knowledge_base — track and dedupe ingestion ===\n\n";
    KnowledgeBase kb;
    std::string text = "A whale is a mammal. It lives in the ocean.";
    int n = kb.ingest_text(text);
    std::cout << "ingested " << n << " facts.\n";
    std::cout << "KB stats: " << kb.stats_facts() << " facts, " << kb.stats_entities() << " entities, " << kb.stats_relations() << " relations.\n";
}

void test_concept_memory() {
    std::cout << "\n=== concept_memory — structural tracking and promotion ===\n\n";
    ConceptMemory cm(2);
    auto varA = ShapeNode::make("A");
    auto varB = ShapeNode::make("B");
    auto shape = ShapeNode::make("*", {varA, varB}); // e.g. mass * accel
    
    std::string id = cm.register_concept(shape);
    std::cout << "registered as: " << id << " (status: " << cm.concepts[id].status << ")\n";
    
    cm.record_use(id);
    cm.record_use(id);
    std::cout << "after 2 uses, status: " << cm.concepts[id].status << "\n";
    
    // Recognize
    auto conc_expr = ShapeNode::make("*", {ShapeNode::make("mass"), ShapeNode::make("accel")});
    auto res = cm.recognize(conc_expr);
    std::cout << "recognize(mass * accel) -> " << res.first << "\n";
}

void test_semantic_depth() {
    std::cout << "\n=== semantic_depth — definition parser ===\n\n";
    std::set<std::string> known = {"mass", "speed"};
    auto parsed = learn_definition("momentum is mass times speed", known);
    if (parsed.has_value()) {
        std::cout << "learned: " << parsed->target << " = " << parsed->arg1 << " " << parsed->op << " " << parsed->arg2 << "\n";
    }
}

int main() {
    test_fact_extractor();
    test_knowledge_base();
    test_concept_memory();
    test_semantic_depth();
    return 0;
}
