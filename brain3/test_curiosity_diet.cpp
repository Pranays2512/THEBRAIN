// test_curiosity_diet.cpp — Sprint 5b: contradiction quarantine + diet ordering.
//
//   1. QUARANTINE  conflicting functional fact never silently co-exists;
//                  stats.contradictions_quarantined fires; KB keeps OLD
//                  value until resolve_quarantine applies the challenger
//   2. AMNESIA LINK resolving with a new template variant changes what the
//                  plan-mouth speaks (memory -> speech, still one store)
//   3. DIET ORDER  curiosity scheduler ranks an unfamiliar fact-dense file
//                  above an already-known sparse one
#include <iostream>
#include <fstream>
#include <string>
#include "core/master_orchestrator.hpp"
#include "core/curiosity_scheduler.hpp"
#include "core/knowledge_ingestion_engine.hpp"

using namespace brain3::core;

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (ok) { g_pass++; std::cout << "  [PASS] " << name << "\n"; }
    else    { g_fail++; std::cout << "  [FAIL] " << name << "\n"; }
}

int main() {
    std::cout << "=== curiosity diet + contradiction quarantine ===\n";

    // ── setup: plan-capable mouth + orchestrator ────────────────────────────
    const char* candidates[] = {"mouth_native.bin", "../mouth_native.bin"};
    for (auto c : candidates) {
        std::FILE* pf = std::fopen(c, "rb");
        if (pf) { std::fclose(pf); setenv("BRAIN_NATIVE_MOUTH_MODEL", c, 1); break; }
    }
    MasterOrchestrator orch;

    // ── 1. quarantine on conflicting template ──────────────────────────────
    auto& F = orch.get_brain()->brainql_engine.facts;
    std::string old_reply;
    {
        auto r0 = orch.process("hello");
        old_reply = r0.natural_reply;
    }
    // challenger: same act, different canonical sentence
    orch.process("teach act:greeting responds intent salutation emotion calm");


    const auto& Q = orch.get_ingestion_engine()->quarantine();
    bool quarantined = false;
    for (auto& q : Q)
        if (q.subj == "act:greeting" && q.rel == "responds" &&
            q.obj_new.find("salutation") != std::string::npos)
            quarantined = true;
    check(quarantined, "conflicting responds-fact quarantined, not co-stored");

    // KB must STILL speak the old variant (old value kept)
    {
        auto r1 = orch.process("hello");
        check(r1.natural_reply == old_reply,
              "old template remains authoritative pre-resolution");
    }

    // ── 2. resolution flips what the mouth speaks ───────────────────────────
    {
        auto* eng = orch.get_ingestion_engine();
        size_t idx = 0;
        for (size_t i = 0; i < Q.size(); ++i)
            if (Q[i].obj_new.find("salutation") != std::string::npos) idx = i;
        check(eng->resolve_quarantine(idx), "quarantine resolved (keep new)");

        auto r2 = orch.process("hello");
        std::cout << "    post-resolution hello -> \"" << r2.natural_reply << "\"\n";
        check(r2.natural_reply.find("calm") != std::string::npos ||
              r2.natural_reply.find("salutation") != std::string::npos,
              "mouth now speaks the resolved variant (one memory, one voice)");
    }

    // ── 3. curiosity-ordered diet ───────────────────────────────────────────
    {
        // known-file: content overlapping seeded templates (low novelty)
        const std::string known = "/tmp/opencode/diet_known.txt";
        { std::ofstream f(known);
          f << "hello intent greeting style friendly\n";
          f << "identity system type cognitive\n"; }
        // fresh-file: unfamiliar domain, fact-dense
        const std::string fresh = "/tmp/opencode/diet_fresh.txt";
        { std::ofstream f(fresh);
          f << "mycelium networks transfer nutrients between trees.\n";
          f << "tardigrades survive vacuum exposure.\n";
          f << "basalt columns form from cooling lava.\n";
          f << "coral reefs shelter marine biodiversity.\n";
          f << "glaciers carve fjords over millennia.\n"; }

        // ingest known content FIRST so the live map has a baseline profile
        orch.process("hello");   // warm mouth path (no learn)
        brain3::core::KnowledgeIngestionEngine eng2(orch.get_brain());
        { brain3::core::IngestionStats st; eng2.ingest_file(known, st); }
        {   // pre-warm: known vocabulary becomes SOM-familiar
            std::ifstream kf(known);
            std::string w;
            while (kf >> w) eng2.observe_word(w);
        }
        CuriosityScheduler sched(eng2.fuzzy());   // live map baseline
        auto ranked = sched.rank({known, fresh}, 8);
        std::cout << "    ranked:\n";
        for (auto& s : ranked)
            std::cout << "      " << s.path.substr(s.path.find_last_of('/') + 1)
                      << " score=" << s.score
                      << " triples=" << s.sampled_triples
                      << " mean_nov=" << s.mean_novelty << "\n";
        std::cout << "\n";
        check(!ranked.empty() && ranked.front().path == fresh,
              "fresh fact-dense source outranks known content");
    }

    std::cout << "=== passed " << g_pass << ", failed " << g_fail << " ===\n";
    return g_fail == 0 ? 0 : 1;
}
