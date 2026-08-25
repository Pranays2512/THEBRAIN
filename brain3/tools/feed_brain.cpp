// feed_brain.cpp — ingest corpus → train graph reasoner → query
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include "core/master_orchestrator.hpp"

using namespace brain3::core;

int main(int argc, char** argv) {
    std::string corpus_path = "/tmp/opencode/massive_corpus.txt";
    if (argc > 1) corpus_path = argv[1];

    MasterOrchestrator orch;

    // Read corpus lines: "subject relation object"
    std::ifstream f(corpus_path);
    if (!f) { std::cerr << "Cannot open " << corpus_path << "\n"; return 1; }

    int ingested = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string s, r, o;
        if (!(iss >> s >> r >> o)) continue;
        orch.get_brain()->brainql_engine.learn(s, r, o);
        ++ingested;
    }
    f.close();
    std::cout << "Ingested " << ingested << " facts into BrainQL.\n";

    // Load into graph reasoner and train
    auto* gr = orch.get_graph_reasoner();
    f.open(corpus_path);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string s, r, o;
        if (!(iss >> s >> r >> o)) continue;
        gr->add_edge(gr->add_entity(s), gr->add_relation(r), gr->add_entity(o));
    }
    f.close();

    std::cout << "Graph: " << gr->entity_count() << " entities, "
              << gr->edge_count() << " edges\n";
    gr->train();
    std::cout << "Graph reasoner trained.\n";

    // Query test
    struct Query { const char* entity; };
    Query probes[] = {
        {"dog"}, {"cat"}, {"lion"}, {"eagle"}, {"hydrogen"},
        {"carbon"}, {"gravity"}, {"light"}, {"pi"}, {"paris"},
    };
    for (auto& q : probes) {
        int eid = gr->entity_id(q.entity);
        if (eid < 0) continue;
        auto qr = gr->query_stages(eid, {-1}, 1.0);
        std::cout << q.entity << " connects to ";
        for (int i = 0; i < std::min(size_t(3), qr.ranked.size()); ++i)
            std::cout << gr->entity_name(qr.ranked[i].entity)
                      << "(" << qr.ranked[i].mass << ") ";
        std::cout << "\n";
    }

    return 0;
}
