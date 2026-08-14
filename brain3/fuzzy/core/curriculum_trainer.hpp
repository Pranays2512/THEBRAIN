#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "fuzzy/core/brain.hpp"
#include "crisp/engines/reasoning/sleep_engine.hpp"

namespace brain3 {
namespace training {

struct CurriculumStats {
    int facts_ingested = 0;
    int words_registered = 0;
    int som_neurons = 0;
    double duration_sec = 0.0;
    double avg_prediction_error = 0.0;
};

class CurriculumTrainer {
private:
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

public:
    // Train a Brain instance over a curriculum file (e.g. brain_curriculum.txt or text facts)
    static CurriculumStats train(brain2::Brain& brain, 
                                 const std::string& dataset_path, 
                                 int epochs = 1, 
                                 bool consolidate_at_end = true,
                                 const std::string& checkpoint_dir = "./out/brain_fluent") {
        CurriculumStats stats;
        
        std::vector<std::string> filepaths;
        std::stringstream pss(dataset_path);
        std::string ptoken;
        while (std::getline(pss, ptoken, ',')) {
            trim(ptoken);
            if (!ptoken.empty()) filepaths.push_back(ptoken);
        }

        std::cout << "\n=============================================================\n";
        std::cout << "         🧠 BRAIN3 CURRICULUM TRAINING PIPELINE\n";
        std::cout << "=============================================================\n";
        std::cout << " Datasets (" << filepaths.size() << " files):\n";
        for (const auto& fp : filepaths) std::cout << "   • " << fp << "\n";
        std::cout << " Epochs:     " << epochs << "\n";
        std::cout << " Checkpoint: " << checkpoint_dir << "\n\n";

        struct Triple {
            std::string s, r, o;
        };
        std::vector<Triple> dataset;

        for (const auto& fp : filepaths) {
            std::ifstream file(fp);
            if (!file.is_open()) {
                std::cerr << "[CurriculumTrainer] Warning: Could not open dataset: " << fp << "\n";
                continue;
            }

            // Check for JSON dataset
            if (fp.length() >= 5 && fp.substr(fp.length() - 5) == ".json") {
                std::string full_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                file.close();

                // Simple JSON object extraction for turn pairs
                size_t pos = 0;
                while ((pos = full_content.find('{', pos)) != std::string::npos) {
                    size_t end_pos = full_content.find('}', pos);
                    if (end_pos == std::string::npos) break;
                    std::string obj_str = full_content.substr(pos + 1, end_pos - pos - 1);
                    pos = end_pos + 1;

                    std::string inp, tgt;
                    // Find "input" or "prompt"
                    size_t in_key = obj_str.find("\"input\"");
                    if (in_key == std::string::npos) in_key = obj_str.find("\"prompt\"");
                    if (in_key != std::string::npos) {
                        size_t c_pos = obj_str.find(':', in_key);
                        size_t q1 = obj_str.find('\"', c_pos);
                        size_t q2 = obj_str.find('\"', q1 + 1);
                        if (q1 != std::string::npos && q2 != std::string::npos) {
                            inp = obj_str.substr(q1 + 1, q2 - q1 - 1);
                        }
                    }

                    // Find "target" or "response"
                    size_t tgt_key = obj_str.find("\"target\"");
                    if (tgt_key == std::string::npos) tgt_key = obj_str.find("\"response\"");
                    if (tgt_key != std::string::npos) {
                        size_t c_pos = obj_str.find(':', tgt_key);
                        size_t q1 = obj_str.find('\"', c_pos);
                        size_t q2 = obj_str.find('\"', q1 + 1);
                        if (q1 != std::string::npos && q2 != std::string::npos) {
                            tgt = obj_str.substr(q1 + 1, q2 - q1 - 1);
                        }
                    }

                    trim(inp); trim(tgt);
                    if (!inp.empty() && !tgt.empty()) {
                        dataset.push_back({inp, "response", tgt});
                    }
                }
                continue;
            }

            std::string line;
            while (std::getline(file, line)) {
                if (line.empty() || line[0] == '#') continue;

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
                } else if (line.find("PROP:") == 0) {
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
                } else if (line.find("EVENT:") == 0) {
                    std::string content = line.substr(6);
                    size_t delim1 = content.find('|');
                    if (delim1 != std::string::npos) {
                        size_t delim2 = content.find('|', delim1 + 1);
                        if (delim2 != std::string::npos) {
                            rel = content.substr(0, delim1);
                            subj = content.substr(delim1 + 1, delim2 - delim1 - 1);
                            obj = content.substr(delim2 + 1);
                        }
                    }
                } else if (line.find(" is a ") != std::string::npos) {
                    size_t pos = line.find(" is a ");
                    subj = line.substr(0, pos);
                    rel = "isa";
                    obj = line.substr(pos + 6);
                } else if (line.find(" is an ") != std::string::npos) {
                    size_t pos = line.find(" is an ");
                    subj = line.substr(0, pos);
                    rel = "isa";
                    obj = line.substr(pos + 7);
                } else if (line.find(" has ") != std::string::npos) {
                    size_t pos = line.find(" has ");
                    subj = line.substr(0, pos);
                    rel = "has";
                    obj = line.substr(pos + 5);
                } else if (line.find(" can ") != std::string::npos) {
                    size_t pos = line.find(" can ");
                    subj = line.substr(0, pos);
                    rel = "can";
                    obj = line.substr(pos + 5);
                } else {
                    // Fallback: whitespace delimited 3-token line
                    std::stringstream ss(line);
                    ss >> subj >> rel >> obj;
                }

                trim(subj); trim(rel); trim(obj);
                if (!subj.empty() && !rel.empty() && !obj.empty()) {
                    dataset.push_back({subj, rel, obj});
                }
            }
            file.close();
        }

        std::cout << "[CurriculumTrainer] Loaded " << dataset.size() << " semantic facts across all corpora.\n";

        auto start_time = std::chrono::high_resolution_clock::now();
        float total_error = 0.f;
        int error_steps = 0;

        for (int ep = 1; ep <= epochs; ep++) {
            int epoch_facts = 0;
            for (const auto& t : dataset) {
                // 1. Ensure vocabulary exists in Language
                if (!brain.language.knows(t.s)) brain.language.register_word(t.s);
                if (!brain.language.knows(t.r)) brain.language.register_word(t.r);
                if (!brain.language.knows(t.o)) brain.language.register_word(t.o);

                // 2. Crisp Reasoning Engine learns fact directly
                brain.brainql_engine.learn(t.s, t.r, t.o);

                // 3. Neural binding & SOM perception
                auto vs = brain.language.encode(t.s);
                auto vr = brain.language.encode(t.r);
                auto vo = brain.language.encode(t.o);

                if (!vs.empty() && !vr.empty() && !vo.empty()) {
                    brain.bind_triple(vs, vr, vo);

                    // Sequence prediction training: [subj] -> [rel] -> target [obj]
                    brain.reset_sequence();
                    brain.perceive(vs);
                    
                    int target_id = brain.language.word_id(t.o);
                    brain.perceive(vr, target_id, brain2::ErrorMode::FULL);

                    total_error += brain.predictor.last_error();
                    error_steps++;
                    epoch_facts++;
                }

                if (epoch_facts % 2000 == 0) {
                    std::cout << "  [Epoch " << ep << "] Ingested " << epoch_facts << "/" << dataset.size() 
                              << " facts (Predictor Error: " << std::fixed << std::setprecision(4) 
                              << (error_steps > 0 ? (total_error / error_steps) : 0.f) << ")\n" << std::flush;
                    brain.tick();
                }
            }
            stats.facts_ingested += epoch_facts;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        stats.duration_sec = std::chrono::duration<double>(end_time - start_time).count();
        stats.som_neurons = brain.som.n_neurons;
        stats.words_registered = brain.language.vocab_size();
        stats.avg_prediction_error = (error_steps > 0) ? (total_error / error_steps) : 0.0;

        std::cout << "\n[CurriculumTrainer] Base Training Complete in " << std::fixed << std::setprecision(2) 
                  << stats.duration_sec << "s.\n";
        std::cout << "  • Total Facts Ingested:   " << stats.facts_ingested << "\n";
        std::cout << "  • Vocab Size Registered:  " << stats.words_registered << "\n";
        std::cout << "  • SOM Neurons Activated:  " << stats.som_neurons << "\n";
        std::cout << "  • Final Predictor Error:  " << stats.avg_prediction_error << "\n\n";

        // Consolidation & Checkpoint
        if (consolidate_at_end) {
            std::cout << "[CurriculumTrainer] Running End-of-Curriculum Sleep Consolidation...\n";
            brain2::reasoning::SleepEngine sleep_engine;
            auto report = sleep_engine.full_sleep(brain, "associative_gate.jsonl", checkpoint_dir, 0.9, 3);
            report.print();
        }

        return stats;
    }
};

} // namespace training
} // namespace brain3
