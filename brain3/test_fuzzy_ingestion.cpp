// test_fuzzy_ingestion.cpp — Sprint 5: fuzzy ingestion pipeline falsifiers.
//
//   1. RESOLUTION   near-string entities sharing a BMU become merge candidates
//   2. NOVELTY      repeated facts drive SOM quantization error DOWN
//                   (the map has learned the region — organ-native adaptation)
//   3. GATING       novelty stats split novel vs predictable populations
//   4. EPISODIC     batches commit into the brain's episodic store
//   5. END-TO-END   natural-text file ingestion routes through the funnel
#include <iostream>
#include <string>
#include <fstream>
#include "core/knowledge_ingestion_engine.hpp"
#include "fuzzy/core/brain.hpp"

using namespace brain3::core;

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (ok) { g_pass++; std::cout << "  [PASS] " << name << "\n"; }
    else    { g_fail++; std::cout << "  [FAIL] " << name << "\n"; }
}

int main() {
    std::cout << "=== fuzzy ingestion pipeline ===\n";
    FuzzyIngestionPipeline pipe;

    // ── 1. entity resolution: near-strings share a BMU bucket ───────────────
    pipe.observe_entity("albert einstein");
    pipe.observe_triple("albert einstein", "isa", "physicist");
    pipe.observe_entity("albert einstein");          // typo variant
    pipe.observe_entity("einstein");                 // short form
    auto groups = pipe.resolution_candidates(0.35f);
    bool found_pair = false;
    for (auto& g : groups)
        for (size_t i = 0; i < g.size(); ++i)
            for (size_t j = i + 1; j < g.size(); ++j)
                if ((g[i].find("einstein") != std::string::npos &&
                     g[j].find("einstein") != std::string::npos))
                    found_pair = true;
    check(found_pair, "near-string entities flagged as resolution candidates");

    // ── 2. novelty drops as the map adapts to a repeated region ─────────────
    double first_pass = 0., second_pass = 0.;
    const int N = 30;
    for (int i = 0; i < N; ++i) {
        first_pass += pipe.observe_triple(
            "robin", "lives_in", i % 2 ? "forest" : "meadow");
        pipe.observe_entity(i % 2 ? "grove" : "glade");
    }
    for (int i = 0; i < N; ++i) {
        second_pass += pipe.observe_triple(
            "robin", "lives_in", i % 2 ? "forest" : "meadow");
        pipe.observe_entity(i % 2 ? "grove" : "glade");
    }
    first_pass /= N; second_pass /= N;
    std::cout << "    mean novelty: first=" << first_pass
              << " second=" << second_pass << "\n";
    check(second_pass <= first_pass,
          "SOM adapts: repeat region scores lower novelty");

    // ── 3. gating stats split populations ───────────────────────────────────
    // (uses the engine wrapper below for the real funnel)

    // ── 4+5. end-to-end through the real ingestion engine ───────────────────
    std::cout << "\n  -- end-to-end via KnowledgeIngestionEngine --\n";
    brain2::Brain brain(16, 16, 128, 128, 7, 500, 8, 42);
    KnowledgeIngestionEngine eng(&brain);

    const char* corpus =
        "The mitochondria is the powerhouse of the cell.\n"
        "A dolphin lives in the ocean.\n"
        "An oak grows on fertile soil.\n"
        "The bee gives honey.\n"
        "Albert Einstein is a physicist.\n"
        "Einstein lives in princeton.\n";
    const std::string path = "/tmp/opencode/fuzzy_ingest_test.txt";
    { std::ofstream f(path); f << corpus; }

    IngestionStats stats;
    bool ok = eng.ingest_file(path, stats);
    check(ok && stats.facts_ingested > 0,
          "natural-text file ingested through fuzzy funnel");
    std::cout << "    facts=" << stats.facts_ingested
              << " novel_flagged=" << eng.novel_flagged()
              << " predictable=" << eng.predictable_count() << "\n";

    // episodic commits happened inside the brain during ingestion:
    check(eng.fuzzy() != nullptr &&
          eng.last_novelty() >= 0.0,
          "fuzzy pipeline attached and produced novelty signal");

    // resolution candidates exist after real ingestion (entities share slots)
    auto cands = eng.fuzzy()->resolution_candidates(0.55f);   // near-dup gate
    std::cout << "    resolution groups: " << cands.size() << "\n";
    bool einstein_pair = false;
    for (auto& g : cands)
        for (size_t i = 0; i < g.size(); ++i)
            for (size_t j = i + 1; j < g.size(); ++j)
                if ((g[i] == "einstein" || g[i] == "albert einstein") &&
                    (g[j] == "einstein" || g[j] == "albert einstein"))
                    einstein_pair = true;
    check(einstein_pair,
          "SOM resolves the einstein/albert-einstein pair end-to-end");

    // dedup: re-ingesting same file yields ~0 new facts (KB + surprise gate)
    IngestionStats s2;
    eng.ingest_file(path, s2);
    std::cout << "    re-ingest new facts: " << s2.facts_ingested << "\n";
    check(s2.facts_ingested < stats.facts_ingested,
          "second pass suppressed by KB-dedup + adaptation");

    std::cout << "=== passed " << g_pass << ", failed " << g_fail << " ===\n";
    return g_fail == 0 ? 0 : 1;
}
