#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include "fuzzy/core/brain.hpp"

using namespace brain2;

int main() {
    std::cout << "=============================================================\n";
    std::cout << "  BRAIN3: LANGUAGE SEQUENCE TRAINING DEMO\n";
    std::cout << "  Training the Hierarchical Predictor and Softmax LM\n";
    std::cout << "=============================================================\n\n";

    // 1. Initialize the Brain with 128D
    Brain brain(16, 16, 128);
    
    std::cout << "Loading GloVe components from ./out/brain_fluent/ ...\n";
    try {
        brain.load_components(
            "./out/brain_fluent/predictor.bin", "./out/brain_fluent/language.bin", "./out/brain_fluent/som.bin", 
            "./out/brain_fluent/episodic.bin", "./out/brain_fluent/emotion.bin", "./out/brain_fluent/self.bin",
            "./out/brain_fluent/symbolic.bin", "./out/brain_fluent/binding.bin", "./out/brain_fluent/bg.bin",
            "./out/brain_fluent/procedures.bin", "./out/brain_fluent/hpred.bin"
        );
    } catch (const std::exception& e) {
        std::cerr << "Failed to load components: " << e.what() << "\n";
        return 1;
    }

    // 2. Load the synthetic conversational dataset
    std::ifstream file("brain_conversations.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open brain_conversations.txt\n";
        return 1;
    }

    std::vector<std::string> sentences;
    std::string line;
    std::unordered_set<int> unique_word_ids;
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            sentences.push_back(line);
            std::stringstream ss(line);
            std::string word;
            while (ss >> word) {
                if (brain.language.knows(word)) {
                    unique_word_ids.insert(brain.language.word_id(word));
                }
            }
        }
    }
    file.close();

    std::cout << "Setting active vocabulary...\n";
    brain.language.freeze_vocabulary(true);
    std::vector<int> active_indices(unique_word_ids.begin(), unique_word_ids.end());
    brain.set_active_vocab(active_indices);
    
    std::ofstream av_file("./out/brain_fluent/active_vocab.txt");
    for (int idx : active_indices) av_file << idx << "\n";
    av_file.close();

    std::cout << "Loaded " << sentences.size() << " conversational exchanges.\n";
    std::cout << "Starting massive sequence training loop...\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    int words_processed = 0;
    int num_epochs = 10;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "Starting Epoch " << (epoch + 1) << "/" << num_epochs << "...\n";
        for (size_t s = 0; s < sentences.size(); s++) {
            std::stringstream ss(sentences[s]);
            std::string word;
            
            while (ss >> word) {
                if (!brain.language.knows(word)) continue;
                
                auto vec = brain.language.encode(word);
                int wid = brain.language.word_id(word);
                
                brain.perceive(vec, wid);
                words_processed++;
            }
            
            brain.working_mem.clear();
            brain.reset_sequence();
        }
        std::cout << "Epoch " << (epoch + 1) << " Error: " << brain.predictor.last_error() << "\n";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "=============================================================\n";
    std::cout << "  TRAINING COMPLETE (" << std::fixed << std::setprecision(2) << duration << " seconds)\n";
    std::cout << "  Words processed: " << words_processed << "\n";
    std::cout << "=============================================================\n";

    // 3. Save the trained brain
    try {
        std::system("mkdir -p ./out/brain_fluent");
        brain.save_components("./out/brain_fluent");
        std::cout << "\n[Save] ✓ Fully trained brain saved to ./out/brain_fluent/\n";
    } catch (const std::exception& e) {
        std::cout << "\n[Save] Warning: " << e.what() << "\n";
    }

    return 0;
}
