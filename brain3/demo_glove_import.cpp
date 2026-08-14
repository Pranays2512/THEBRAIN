/*
 * demo_glove_import.cpp — Bootstrapping the Biological Language System
 *
 * This script ingests the massive 400,000 word GloVe 100D semantic database
 * and wires it directly into Brain3's native language system, allowing it
 * to "understand" and "speak" English natively without an LLM.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include "fuzzy/core/brain.hpp"

using namespace brain2;

int main() {
    std::cout << "=============================================================\n";
    std::cout << "  BRAIN3: LANGUAGE BOOTSTRAP (GLOVE INGESTION)\n";
    std::cout << "=============================================================\n\n";

    // 1. Initialize the Brain with 128D (Standard for Brain3 default demos)
    // We will map 100D GloVe to the first 100 dimensions of the Brain's vectors.
    int n_dims = 128;
    Brain brain(16, 16, n_dims, 128, 7, 500, 8, 42);

    std::cout << "[1] Opening GloVe database (400,000 words)...\n";
    std::ifstream file("../brain2/glove.6B.100d.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open ../brain2/glove.6B.100d.txt\n";
        return 1;
    }

    std::string line;
    int words_loaded = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "[2] Ingesting semantic vectors into Biological Core...\n";
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string word;
        ss >> word;

        std::vector<float> vec(n_dims, 0.0f);
        float val;
        int i = 0;
        while (ss >> val && i < 100) {
            vec[i] = val;
            i++;
        }

        // Only register if we successfully parsed the 100 dimensions
        if (i == 100) {
            brain.language.register_word(word, vec);
            words_loaded++;
        }

        if (words_loaded % 50000 == 0) {
            std::cout << "  Loaded " << words_loaded << " words...\n";
        }
    }
    file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "=============================================================\n";
    std::cout << "  INGESTION COMPLETE (" << std::fixed << std::setprecision(2) << duration << " seconds)\n";
    std::cout << "=============================================================\n";
    std::cout << "Total fluent words learned: " << words_loaded << "\n\n";

    // 2. Perform a native semantic recall test
    std::cout << "--- NATIVE FLUENCY TEST ---\n";
    
    auto encode = [&](const std::string& w) { return brain.language.encode(w); };
    
    // Semantic arithmetic: King - Man + Woman = ?
    auto v_king = encode("king");
    auto v_man = encode("man");
    auto v_woman = encode("woman");
    
    std::vector<float> query_vec(n_dims, 0.0f);
    for (int i = 0; i < n_dims; i++) {
        query_vec[i] = v_king[i] - v_man[i] + v_woman[i];
    }

    std::cout << "Biological Query: 'king - man + woman'\n";
    auto results = brain.language.decode(query_vec, 3, {"king", "man", "woman"});
    
    std::cout << "Brain Native Response (Top 3):\n";
    for (const auto& res : results) {
        std::cout << "  -> " << res.first << " (sim: " << res.second << ")\n";
    }

    std::cout << "\nBiological Query: 'paris - france + germany'\n";
    auto v_paris = encode("paris");
    auto v_france = encode("france");
    auto v_germany = encode("germany");
    
    std::vector<float> query_vec2(n_dims, 0.0f);
    for (int i = 0; i < n_dims; i++) {
        query_vec2[i] = v_paris[i] - v_france[i] + v_germany[i];
    }
    auto results2 = brain.language.decode(query_vec2, 3, {"paris", "france", "germany"});
    for (const auto& res : results2) {
        std::cout << "  -> " << res.first << " (sim: " << res.second << ")\n";
    }

    // 3. Save the fluent brain
    try {
        std::system("mkdir -p ./out/brain_fluent");
        brain.save_components("./out/brain_fluent");
        std::cout << "\n[Save] ✓ Fluent Biological Brain saved to ./out/brain_fluent/\n";
    } catch (const std::exception& e) {
        std::cout << "\n[Save] Warning: " << e.what() << "\n";
    }

    return 0;
}
