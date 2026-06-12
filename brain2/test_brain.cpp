#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include "core/brain.hpp"

using namespace brain2;

// Helper: euclidean squared
float vec_l2sq(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

void print_header(const std::string& title) {
    std::cout << "\n======================================================\n";
    std::cout << " " << title << "\n";
    std::cout << "======================================================\n" << std::flush;
}

int main() {
    std::cout << "Loading Brain Components...\n" << std::flush;
    Brain b(128, 128, 64, 512);
    std::string cd = "checkpoints/executive_brain";
    try {
        b.load_components(cd + "/predictor.bin", cd + "/language.bin", cd + "/som.bin", 
            cd + "/episodic.bin", cd + "/emotion.bin", cd + "/self.bin",
            cd + "/symbolic.bin", cd + "/binding.bin", cd + "/bg.bin",
            cd + "/procedures.bin", cd + "/hpred.bin"
        );
        std::cout << "Brain Loaded. Vocabulary size: " << b.language.vocab_size() << "\n" << std::flush;
    } catch(const std::exception& e) {
        std::cout << "Warning loading components: " << e.what() << "\nProceeding with untrained components...\n" << std::flush;
    }

    // Scorecard variables
    float lang_acc_clean = 0;
    float lang_noise_thresh = 0;
    float pred_loop_final_err = 0;
    float pred_heldout_ppl = 0.f;
    float pred_heldout_top1 = 0.f;
    float pred_heldout_top5 = 0.f;
    float pred_heldout_top20 = 0.f;
    float baseline_ppl = 0.f;
    float baseline_top1 = 0.f;
    float bind_acc_50 = 0.f;
    float bind_acc_200 = 0.f;
    float bind_acc_500 = 0.f;
    float l2_acc = 0.f;
    float l3_acc = 0.f;
    int dist_pass = 0, dist_total = 0;
    int active_vocab_size = 0;
    float wm_gate_rate = 0.f;
    float ep_commit_rate = 0.f;
    float speedup_factor = 0;
    float tree_som_disagreement = 0;

    // --- TEST 1: Language round-trip ---
    print_header("TEST 1 - Language Round-Trip");
    {
        std::cout << "Running clean round-trip... " << std::flush;
        int correct = 0, total = 0;
        int max_eval = 2000;
        for (const auto& w : b.language.vocab()) {
            if (total > max_eval) break;
            auto vec = b.language.encode(w);
            auto dec = b.language.best_word(vec);
            if (dec == w) correct++;
            total++;
        }
        lang_acc_clean = (float)correct / total * 100.0f;
        std::cout << std::fixed << std::setprecision(2) << lang_acc_clean << "% (" << correct << "/" << total << ")\n" << std::flush;

        std::cout << "Running noise test... " << std::flush;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float noise_std = 0.01f;
        while (noise_std < 2.0f) {
            int noise_correct = 0;
            int test_count = std::min(100, total); // REDUCED for speed
            int idx = 0;
            for (const auto& w : b.language.vocab()) {
                if (idx++ >= test_count) break;
                auto vec = b.language.encode(w);
                for(auto& val : vec) val += dist(rng) * noise_std;
                if (b.language.best_word(vec) == w) noise_correct++;
            }
            if ((float)noise_correct / test_count < 0.90f) {
                lang_noise_thresh = noise_std;
                break;
            }
            noise_std += 0.05f;
        }
        std::cout << "Threshold (90% retention): sigma = " << lang_noise_thresh << "\n\n" << std::flush;

        std::vector<std::string> test_words = {"dog", "run", "red", "water", "king"};
        std::cout << "Nearest Neighbors (k=5):\n" << std::flush;
        for (const auto& w : test_words) {
            if (b.language.knows(w)) {
                auto vec = b.language.encode(w);
                auto nn = b.language.decode(vec, 6);
                std::cout << "  " << w << " -> ";
                for (auto& n : nn) if(n.first != w) std::cout << n.first << " ";
                std::cout << "\n";
            } else {
                std::cout << "  " << w << " -> [NOT IN VOCAB]\n";
            }
        }
        std::cout << std::flush;
    }

    print_header("TEST 2 - Predictor Held-Out Perplexity (FULL path, offline)");
    {
        std::cout << "Loading training corpus to build active vocabulary... " << std::flush;
        std::map<std::string, std::map<std::string, int>> bigram;
        std::map<std::string, int> unigram;
        int total_unigrams = 0;
        std::ifstream train_f("data/train_sentences_80k.txt");
        if (train_f.is_open()) {
            std::string line, w, prev;
            while(std::getline(train_f, line)) {
                std::istringstream iss(line);
                prev = "<START>";
                while(iss >> w) {
                    bigram[prev][w]++;
                    unigram[w]++;
                    total_unigrams++;
                    prev = w;
                }
            }
        }
        
        std::vector<int> active_indices;
        for (const auto& pair : unigram) {
            if (b.language.knows(pair.first)) {
                active_indices.push_back(b.language.word_id(pair.first));
            }
        }
        b.set_active_vocab(active_indices);
        active_vocab_size = active_indices.size();
        std::cout << "Done. Active vocab size: " << active_vocab_size << "\n";

        // --- Gate rate measurement: 500-word sample with FULL error mode ---
        std::cout << "Measuring gate firing rates (200 burn-in + 500 words)... " << std::flush;
        {
            b.reset_sequence();
            std::ifstream gf("data/test_sentences_6k.txt");
            int wm_open = 0, ep_stored = 0, w_count = 0;
            int burn_in = 0;
            if (gf.is_open()) {
                std::string line;
                while (std::getline(gf, line) && w_count < 500) {
                    std::istringstream iss(line);
                    std::string w;
                    while (iss >> w && w_count < 500) {
                        if (!b.language.knows(w)) continue;
                        int wid = b.language.word_id(w);
                        auto res = b.perceive(b.language.encode(w), wid, ErrorMode::FULL);
                        
                        b.update_error_stats(b.predictor.last_error());
                        
                        if (burn_in < 200) {
                            burn_in++;
                        } else {
                            if (res.wm_passed) wm_open++;
                            if (res.episodic_stored) ep_stored++;
                            w_count++;
                        }
                    }
                }
            }
            wm_gate_rate = w_count > 0 ? (float)wm_open / w_count * 100.f : 0.f;
            ep_commit_rate = w_count > 0 ? (float)ep_stored / w_count * 100.f : 0.f;
        }

        /*
        std::cout << "Done.\n";
        std::cout << "Running held-out test... " << std::flush;

        std::ifstream f("data/test_sentences_6k.txt");
        if (f.is_open()) {
            std::string line;
            double lstm_ce_sum = 0.0, bigram_ce_sum = 0.0;
            int words_evaluated = 0;
            int top_1 = 0, top_5 = 0, top_20 = 0, bigram_top1 = 0;
            b.reset_sequence();
            
            b.predictor.set_offline(true); // Disable gradient updates
            
            while (std::getline(f, line)) {
                std::istringstream iss(line);
                std::string w, prev = "<START>";
                while (iss >> w) {
                    if (!b.language.knows(w)) continue;
                    int target_id = b.language.word_id(w);
                    auto ranked_preds = b.predictor.step(b.language.encode(prev), target_id, ErrorMode::FULL);
                    lstm_ce_sum += b.predictor.last_error();
                    
                    bool found1=false, found5=false, found20=false;
                    for (size_t rank = 0; rank < ranked_preds.size(); rank++) {
                        if (ranked_preds[rank].second == target_id) {
                            if (rank < 1) found1 = true;
                            if (rank < 5) found5 = true;
                            if (rank < 20) found20 = true;
                            break;
                        }
                    }
                    if(found1) top_1++;
                    if(found5) top_5++;
                    if(found20) top_20++;

                    float bg_prob = 1e-6f;
                    float uni_prob = (unigram.count(w) > 0) ? ((float)unigram[w] / total_unigrams) : 1e-6f;
                    if (bigram.count(prev) && bigram[prev].count(w)) {
                        int prev_total = 0;
                        for (auto& pair : bigram[prev]) prev_total += pair.second;
                        if (prev_total > 0) bg_prob = (float)bigram[prev][w] / prev_total;
                    }
                    float p_interp = 0.7f * bg_prob + 0.3f * uni_prob;
                    bigram_ce_sum += -std::log(std::max(p_interp, 1e-7f));

                    std::string bg_best = ""; int bg_max = 0;
                    if (bigram.count(prev)) {
                        for (auto& pair : bigram[prev]) {
                            if (pair.second > bg_max) { bg_max = pair.second; bg_best = pair.first; }
                        }
                    }
                    if (bg_best == w) bigram_top1++;

                    words_evaluated++;
                    prev = w;
                }
            }
            b.predictor.set_offline(false); // Re-enable gradient updates for the rest of the application
            pred_heldout_ppl = std::exp(lstm_ce_sum / words_evaluated);
            baseline_ppl = std::exp(bigram_ce_sum / words_evaluated);
            pred_heldout_top1 = (float)top_1 / words_evaluated * 100.f;
            pred_heldout_top5 = (float)top_5 / words_evaluated * 100.f;
            pred_heldout_top20 = (float)top_20 / words_evaluated * 100.f;
            baseline_top1 = (float)bigram_top1 / words_evaluated * 100.f;
        }
    }

    // --- TEST 3: Binding memory retrieval ---
    print_header("TEST 3 - Binding Memory Retrieval");
    {
        std::cout << "Running binding tests... " << std::flush;
        int max_triples = 500;
        std::vector<std::vector<float>> subs, rels, objs;
        auto vocab = b.language.vocab();
        auto it = vocab.begin();
        for(int i=0; i<max_triples; i++) {
            if (it == vocab.end()) it = vocab.begin();
            subs.push_back(b.language.encode(*it++));
            if (it == vocab.end()) it = vocab.begin();
            rels.push_back(b.language.encode(*it++));
            if (it == vocab.end()) it = vocab.begin();
            objs.push_back(b.language.encode(*it++));
        }

        auto eval_at = [&](int count) {
            int hits = 0;
            for(int i=0; i<count; i++) {
                auto res = b.binding_query(subs[i], rels[i]);
                if (b.language.best_word(res.first) == b.language.best_word(objs[i])) hits++;
            }
            return (float)hits / count * 100.0f;
        };

        for(int i=0; i<50; i++) b.bind_triple(subs[i], rels[i], objs[i]);
        float bind_acc_50 = eval_at(50);
        for(int i=50; i<200; i++) b.bind_triple(subs[i], rels[i], objs[i]);
        float bind_acc_200 = eval_at(200);
        for(int i=200; i<500; i++) b.bind_triple(subs[i], rels[i], objs[i]);
        bind_acc_500 = eval_at(500);

        std::cout << "\nCapacity Curve:\n";
        std::cout << "  Accuracy at 50 triples:  " << bind_acc_50 << "%\n";
        std::cout << "  Accuracy at 200 triples: " << bind_acc_200 << "%\n";
        std::cout << "  Accuracy at 500 triples: " << bind_acc_500 << "%\n";
        
        if(!b.language.knows("cat")) b.language.register_word("cat");
        if(!b.language.knows("animal")) b.language.register_word("animal");
        if(!b.language.knows("isa")) b.language.register_word("isa");
        if(!b.language.knows("dog")) b.language.register_word("dog");
        
        b.bind_triple(b.language.encode("dog"), b.language.encode("isa"), b.language.encode("animal"));
        auto res = b.binding_query(b.language.encode("cat"), b.language.encode("isa"));
        std::cout << "Generalization (cat isa ?): " << b.language.best_word(res.first) << " (conf=" << res.second << ")\n" << std::flush;
    }
    */

    // --- TEST 4: Logic engine ---
    print_header("TEST 4 - Logic Engine (Reasoning Depth)");
    {
        std::cout << "Simulating logic inference chains...\n";
        struct LogicProblem {
            std::vector<std::tuple<std::string, std::string, std::string>> facts;
            std::string q_subj;
            std::string q_rel;
            std::string expected;
            int hops;
        };
        
        std::vector<LogicProblem> problems = {
            // 2-hop chains
            {{{"rain", "causes", "flood"}, {"flood", "causes", "damage"}}, "rain", "causes", "damage", 2},
            {{{"sun", "causes", "heat"}, {"heat", "causes", "fire"}}, "sun", "causes", "fire", 2},
            {{{"virus", "causes", "fever"}, {"fever", "causes", "sweat"}}, "virus", "causes", "sweat", 2},
            {{{"work", "causes", "money"}, {"money", "causes", "food"}}, "work", "causes", "food", 2},
            {{{"study", "causes", "knowledge"}, {"knowledge", "causes", "power"}}, "study", "causes", "power", 2},
            {{{"ice", "causes", "slip"}, {"slip", "causes", "fall"}}, "ice", "causes", "fall", 2},
            {{{"wind", "causes", "cold"}, {"cold", "causes", "shiver"}}, "wind", "causes", "shiver", 2},
            {{{"sugar", "causes", "energy"}, {"energy", "causes", "movement"}}, "sugar", "causes", "movement", 2},
            {{{"exercise", "causes", "muscle"}, {"muscle", "causes", "strength"}}, "exercise", "causes", "strength", 2},
            {{{"sleep", "causes", "rest"}, {"rest", "causes", "health"}}, "sleep", "causes", "health", 2},
            
            // Clean 3-hop chains
            {{{"seed", "causes", "plant"}, {"plant", "causes", "flower"}, {"flower", "causes", "fruit"}}, "seed", "causes", "fruit", 3},
            {{{"spark", "causes", "fire"}, {"fire", "causes", "smoke"}, {"smoke", "causes", "alarm"}}, "spark", "causes", "alarm", 3},
            {{{"cloud", "causes", "rain"}, {"rain", "causes", "water"}, {"water", "causes", "life"}}, "cloud", "causes", "life", 3},
            {{{"thought", "causes", "idea"}, {"idea", "causes", "plan"}, {"plan", "causes", "action"}}, "thought", "causes", "action", 3},
            {{{"time", "causes", "change"}, {"change", "causes", "growth"}, {"growth", "causes", "decay"}}, "time", "causes", "decay", 3},
            
            // Distractor problems
            {{{"A", "causes", "B"}, {"A", "annoys", "D"}, {"B", "causes", "C"}}, "A", "causes", "C", 4},
            {{{"cat", "causes", "scratch"}, {"cat", "eats", "fish"}, {"scratch", "causes", "pain"}}, "cat", "causes", "pain", 4},
            {{{"dog", "causes", "bark"}, {"dog", "likes", "bone"}, {"bark", "causes", "noise"}}, "dog", "causes", "noise", 4},
            {{{"car", "causes", "travel"}, {"car", "needs", "gas"}, {"travel", "causes", "fatigue"}}, "car", "causes", "fatigue", 4},
            {{{"book", "causes", "reading"}, {"book", "has", "pages"}, {"reading", "causes", "learning"}}, "book", "causes", "learning", 4}
        };
        
        int h2_pass = 0, h2_total = 0;
        int h3_pass = 0, h3_total = 0;
        dist_pass = 0; dist_total = 0;
        
        for (const auto& p : problems) {
            b.language.register_word(p.q_subj);
            b.language.register_word(p.q_rel);
            b.language.register_word(p.expected);
            for (const auto& f : p.facts) {
                b.language.register_word(std::get<0>(f));
                b.language.register_word(std::get<1>(f));
                b.language.register_word(std::get<2>(f));
            }
        }
        
        int correct_2hop = 0, total_2hop = 0;
        int correct_3hop = 0, total_3hop = 0;
        
        for (const auto& p : problems) {
            b.binding.bindings_.clear(); // clear memory for each problem to isolate
            for (const auto& f : p.facts) {
                b.bind_triple(b.language.encode(std::get<0>(f)), 
                              b.language.encode(std::get<1>(f)), 
                              b.language.encode(std::get<2>(f)));
            }
            
            b.scratchpad.clear();
            b.scratchpad.write("subject", b.language.encode(p.q_subj), "query");
            b.scratchpad.write("relation", b.language.encode(p.q_rel), "query");
            
            b.logic_engine.execute_op(Op::CHAIN_FOLLOW, b.scratchpad);
            
            auto result = b.scratchpad.read("result");
            std::string ans = b.language.best_word(result);
            if (p.hops == 2) {
                h2_total++;
                if (ans == p.expected) h2_pass++;
            } else if (p.hops == 3) {
                h3_total++;
                if (ans == p.expected) h3_pass++;
            } else if (p.hops == 4) { // Distractors
                dist_total++;
                if (ans == p.expected) dist_pass++;
            }
        }
        
        printf("  2-Hop Chain Accuracy: %.2f%% (%d/%d)\n", h2_total > 0 ? (float)h2_pass/h2_total*100.f : 0.f, h2_pass, h2_total);
        printf("  3-Hop Chain Accuracy: %.2f%% (%d/%d)\n", h3_total > 0 ? (float)h3_pass/h3_total*100.f : 0.f, h3_pass, h3_total);
        printf("  Distractor Accuracy : %.2f%% (%d/%d)\n", dist_total > 0 ? (float)dist_pass/dist_total*100.f : 0.f, dist_pass, dist_total);
        
        l2_acc = h2_total > 0 ? (float)h2_pass / h2_total * 100.f : 0.f;
        l3_acc = h3_total > 0 ? (float)h3_pass / h3_total * 100.f : 0.f;
    }

    // --- SCORECARD ---
    print_header("FINAL SCORECARD");
    // std::cout << "1. Language Round-Trip : " << std::fixed << std::setprecision(2) << lang_acc_clean << "% (Noise threshold: " << lang_noise_thresh << ")\n";
    // std::cout << "2. Cognitive Gate Rates: WM=" << wm_gate_rate << "%, Episodic=" << ep_commit_rate << "%\n";
    // std::cout << "3. Held-Out Perplexity (over " << active_vocab_size << "-word vocab) :\n";
    // std::cout << "   LSTM  Top-1/5/20:    " << std::fixed << std::setprecision(2) << pred_heldout_top1 << "% / " << pred_heldout_top5 << "% / " << pred_heldout_top20 << "%\n";
    // std::cout << "   LSTM  Perplexity:    " << std::fixed << std::setprecision(1) << pred_heldout_ppl << "\n";
    // std::cout << "   Bigram Top-1:        " << std::fixed << std::setprecision(2) << baseline_top1 << "%\n";
    // std::cout << "   Bigram Perplexity:   " << std::fixed << std::setprecision(1) << baseline_ppl << "\n";
    // std::cout << "   LSTM vs Bigram PPL:  " << (pred_heldout_ppl < baseline_ppl ? "LSTM WINS" : "BIGRAM WINS") << "\n";
    // std::cout << "4. Binding (500 items) : " << bind_acc_500 << "%\n";
    std::cout << "5. Logic Hops          : 2-Hop " << l2_acc << "% / 3-Hop " << l3_acc << "% Accuracy.\n";
    std::cout << "   Logic Distractors   : " << (dist_total > 0 ? (float)dist_pass/dist_total*100.f : 0.f) << "% Accuracy.\n";
    // std::cout << "6. Tree-SOM exactness  : " << std::fixed << std::setprecision(2) << (100.f - tree_som_disagreement) << "%\n";
    // std::cout << "   Neighbor check (dog/run/red):";
    // for (const auto& w : std::vector<std::string>{"dog", "run", "red"}) {
    //     if (b.language.knows(w)) {
    //         auto nn = b.language.decode(b.language.encode(w), 4);
    //         std::cout << " " << w << ":";
    //         for (auto& n : nn) if (n.first != w) std::cout << n.first << ",";
    //     }
    // }
    std::cout << "\n";
    std::cout << "======================================================\n" << std::flush;

    }
    return 0;
}
