#pragma once
/*
 * brain_trainer.hpp — Unified Training Pipeline for Brain3
 *
 * Provides a single, structured entry point for training all learning engines:
 *   1. Loads intuition weights from disk (if they exist).
 *   2. Feeds training examples through UnifiedProposer in epoch batches.
 *   3. After each epoch, runs the SleepEngine to consolidate learned patterns.
 *   4. Tracks routing accuracy, depth usage, and improvement over epochs.
 *   5. Saves intuition weights to disk after every epoch (checkpoint).
 *
 * Usage:
 *   BrainTrainer trainer("./brain_weights");
 *   trainer.add_example({.type="equation", .data_str="2x+4=10"}, true);
 *   trainer.train(50 epochs);
 */
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include "fuzzy/engines/synthesis/unified_proposer.hpp"
#include "crisp/engines/reasoning/reasoning_engine.hpp"
#include "crisp/engines/reasoning/sleep_engine.hpp"

namespace brain3 {
namespace training {

using engines::synthesis::Problem;
using engines::synthesis::UnifiedProposer;

struct TrainingExample {
    Problem problem;
    bool    expected_success;
};

struct EpochResult {
    int epoch;
    int total;
    int direct_hits;       // Intuition routed correctly on first try
    int fallback_hits;     // Needed fallback (wrong intuition, corrected)
    int failures;          // Could not solve at all
    double accuracy;       // direct_hits / total
    long long duration_ms;
};

class BrainTrainer {
private:
    UnifiedProposer proposer;
    brain2::reasoning::ReasoningEngine reasoning_kb;
    brain2::reasoning::SleepEngine sleep_engine;
    std::string weights_path;
    std::vector<TrainingExample> dataset;
    std::vector<EpochResult> history;

    void print_epoch(const EpochResult& r) {
        std::cout << "[Epoch " << std::setw(3) << r.epoch << "] "
                  << "Accuracy: " << std::fixed << std::setprecision(1) << (r.accuracy * 100.0) << "% "
                  << "(Direct: " << r.direct_hits
                  << " | Fallback: " << r.fallback_hits
                  << " | Failed: " << r.failures << ")"
                  << " [" << r.duration_ms << "ms]\n";
    }

public:
    explicit BrainTrainer(const std::string& weights_directory = "./brain_weights")
        : weights_path(weights_directory) {
        // Attempt to resume from last saved weights
        bool loaded = proposer.load_weights(weights_path + "/intuition.bin");
        if (!loaded) {
            std::cout << "[Trainer] No saved weights found. Starting fresh training.\n";
        }
    }

    void add_example(const Problem& p, bool expected_success = true) {
        dataset.push_back({p, expected_success});
    }

    void add_examples(const std::vector<TrainingExample>& examples) {
        for (const auto& ex : examples) dataset.push_back(ex);
    }

    // Train for N epochs over the full dataset
    // After each epoch: checkpoint weights + run SleepEngine consolidation
    void train(int epochs = 10, bool verbose = true) {
        if (dataset.empty()) {
            std::cout << "[Trainer] No training data loaded. Aborting.\n";
            return;
        }

        std::cout << "\n=== BRAIN3 TRAINING PIPELINE ===\n";
        std::cout << "Dataset: " << dataset.size() << " examples | Epochs: " << epochs << "\n";
        std::cout << "Weights path: " << weights_path << "\n\n";

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            auto t_start = std::chrono::steady_clock::now();

            // Reset per-epoch stats
            proposer.stats = {};

            // Forward + Backward pass on every training example
            for (const auto& ex : dataset) {
                proposer.solve_tracked(ex.problem);
            }

            auto t_end = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

            EpochResult r;
            r.epoch         = epoch;
            r.total         = proposer.stats.total;
            r.direct_hits   = proposer.stats.direct_success;
            r.fallback_hits = proposer.stats.fallback_success;
            r.failures      = proposer.stats.failed;
            r.accuracy      = (r.total > 0) ? (double)r.direct_hits / r.total : 0.0;
            r.duration_ms   = ms;
            history.push_back(r);

            if (verbose) print_epoch(r);

            // Checkpoint: Save weights after every epoch so training is never lost
            proposer.save_weights(weights_path + "/intuition.bin");

            // SleepEngine consolidation: compress redundant facts into rules every 5 epochs
            if (epoch % 5 == 0 && !reasoning_kb.facts.empty()) {
                std::cout << "  [Sleep] Consolidating knowledge base...\n";
                sleep_engine.sleep(reasoning_kb, 0.9, 3);
            }

            // Early stopping: if we reach 95% direct accuracy, stop training
            if (r.accuracy >= 0.95 && epoch >= 5) {
                std::cout << "\n[Trainer] Target accuracy reached at epoch " << epoch << "! Stopping early.\n";
                break;
            }
        }

        print_summary();
    }

    void print_summary() const {
        if (history.empty()) return;
        std::cout << "\n=== TRAINING SUMMARY ===\n";
        const auto& first = history.front();
        const auto& last  = history.back();
        std::cout << "Epochs run:      " << history.size() << "\n";
        std::cout << "Starting accuracy: " << std::fixed << std::setprecision(1) << (first.accuracy * 100.0) << "%\n";
        std::cout << "Final accuracy:    " << (last.accuracy * 100.0) << "%\n";
        std::cout << "Improvement:       +" << ((last.accuracy - first.accuracy) * 100.0) << "%\n";
    }

    // Expose proposer for inference after training
    UnifiedProposer& get_proposer() { return proposer; }
    brain2::reasoning::ReasoningEngine& get_knowledge_base() { return reasoning_kb; }
    const std::vector<EpochResult>& get_history() const { return history; }
};

} // namespace training
} // namespace brain3
