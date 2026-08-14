/*
 * demo_curriculum_training.cpp — Trains Brain3 on the massive brain_curriculum.txt
 * semantic dataset extracted from textbooks. 
 * This directly exercises the BindingMemory, SOM, and Language embedding system.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include "fuzzy/core/brain.hpp"

using namespace brain2;

// Helper to trim strings
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << "  BRAIN3: MASSIVE CURRICULUM TRAINING\n";
    std::cout << "  Ingesting semantic facts from brain2/data/brain_curriculum.txt\n";
    std::cout << "=============================================================\n\n";

    // 1. Initialize the Brain with larger memory for 18k facts
    Brain brain(1024, 64, 64, 128, 7, 20000, 8, 42);
    
    std::ifstream file("../brain2/data/brain_curriculum.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open ../brain2/data/brain_curriculum.txt\n";
        return 1;
    }

    std::string line;
    int facts_learned = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::string subj, rel, obj;
        if (line.find("ISA:") == 0) {
            std::string content = line.substr(4);
            size_t delim = content.find('|');
            if (delim != std::string::npos) {
                subj = content.substr(0, delim);
                obj = content.substr(delim + 1);
                rel = "isa";
            }
        } else if (line.find("FACT:") == 0) {
            std::string content = line.substr(5);
            size_t delim1 = content.find('|');
            if (delim1 != std::string::npos) {
                size_t delim2 = content.find('|', delim1 + 1);
                if (delim2 != std::string::npos) {
                    subj = content.substr(0, delim1);
                    rel = content.substr(delim1 + 1, delim2 - delim1 - 1);
                    obj = content.substr(delim2 + 1);
                }
            }
        }

        trim(subj); trim(rel); trim(obj);
        if (!subj.empty() && !rel.empty() && !obj.empty()) {
            auto v_subj = brain.language.encode(subj);
            auto v_rel  = brain.language.encode(rel);
            auto v_obj  = brain.language.encode(obj);

            brain.bind_triple(v_subj, v_rel, v_obj);
            facts_learned++;

            // Periodically tick to allow daydreaming and consolidation
            if (facts_learned % 100 == 0) {
                brain.tick();
            }
        }
    }
    file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "=============================================================\n";
    std::cout << "  TRAINING COMPLETE (" << std::fixed << std::setprecision(2) << duration << " seconds)\n";
    std::cout << "=============================================================\n";
    std::cout << "Total semantic facts ingested: " << facts_learned << "\n";
    std::cout << "Total SOM Neurons Minted:      " << brain.som.n_neurons << "\n\n";

    // Let's test recall!
    std::cout << "--- Testing Recall ---\n";
    auto query_fact = [&](const std::string& s, const std::string& r) {
        auto vs = brain.language.encode(s);
        auto vr = brain.language.encode(r);
        auto [v_ans, conf] = brain.binding_query(vs, vr, true, 0.1f);
        
        std::cout << "Q: " << s << " " << r << " ?\n";
        if (conf > 0.1f) {
            auto decoded = brain.language.decode(v_ans);
            std::string ans = decoded.empty() ? "[unknown]" : decoded[0].first;
            std::cout << "A: " << ans << " (conf=" << conf << ")\n";
        } else {
            std::cout << "A: [No memory found]\n";
        }
    };

    query_fact("textbook", "has");
    query_fact("printing_works", "located_in");
    query_fact("textbook", "printed_on");

    try {
        std::system("mkdir -p ./out/brain_curriculum_trained");
        brain.save_components("./out/brain_curriculum_trained");
        std::cout << "\n[Save] ✓ Semantic Brain saved to ./out/brain_curriculum_trained/\n";
    } catch (const std::exception& e) {
        std::cout << "\n[Save] Warning: " << e.what() << "\n";
    }

    return 0;
}
