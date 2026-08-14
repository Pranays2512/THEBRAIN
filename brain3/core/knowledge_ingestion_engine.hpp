#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <cctype>

#include "../fuzzy/core/brain.hpp"
#include "algorithmic_policy_engine.hpp"

namespace brain3 {
namespace core {

struct IngestionStats {
    size_t files_processed = 0;
    size_t lines_read = 0;
    size_t facts_ingested = 0;
    size_t rules_ingested = 0;
    size_t isa_relations_ingested = 0;
    size_t domains_registered = 0;
    size_t contradictions_quarantined = 0;
    double elapsed_ms = 0.0;
    double throughput_facts_per_sec = 0.0;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"files_processed\": " << files_processed << ",\n";
        oss << "  \"lines_read\": " << lines_read << ",\n";
        oss << "  \"facts_ingested\": " << facts_ingested << ",\n";
        oss << "  \"rules_ingested\": " << rules_ingested << ",\n";
        oss << "  \"isa_relations_ingested\": " << isa_relations_ingested << ",\n";
        oss << "  \"domains_registered\": " << domains_registered << ",\n";
        oss << "  \"contradictions_quarantined\": " << contradictions_quarantined << ",\n";
        oss << "  \"elapsed_ms\": " << elapsed_ms << ",\n";
        oss << "  \"throughput_facts_per_sec\": " << throughput_facts_per_sec << "\n";
        oss << "}";
        return oss.str();
    }
};

class KnowledgeIngestionEngine {
private:
    brain2::Brain* brain_;
    std::unordered_map<std::string, std::vector<brain2::reasoning::DomainTriple>> pending_domain_triples_;

    static std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }

    static std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s;
    }

    static std::vector<std::string> split(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(trim(token));
        }
        return tokens;
    }

    static bool is_number(const std::string& s, double& out_val) {
        if (s.empty()) return false;
        try {
            size_t idx = 0;
            out_val = std::stod(s, &idx);
            return idx == s.length();
        } catch (...) {
            return false;
        }
    }

public:
    explicit KnowledgeIngestionEngine(brain2::Brain* brain) : brain_(brain) {}

    /**
     * Ingests a single file (auto-detects tagged format, pair format, json, or raw text)
     */
    bool ingest_file(const std::string& filepath, IngestionStats& stats, const std::string& override_domain = "") {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;

        std::string filename = std::filesystem::path(filepath).stem().string();
        std::string domain = override_domain.empty() ? infer_domain_from_filename(filename) : override_domain;

        std::string line;
        while (std::getline(file, line)) {
            stats.lines_read++;
            std::string trimmed = trim(line);
            if (trimmed.empty() || trimmed[0] == '#') continue;

            ingest_line(trimmed, domain, stats);
        }

        stats.files_processed++;
        return true;
    }

    /**
     * Ingests an entire directory of files
     */
    IngestionStats ingest_directory(const std::string& dirpath, const std::vector<std::string>& file_prefixes = {}) {
        IngestionStats stats;
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!std::filesystem::exists(dirpath)) return stats;

        for (const auto& entry : std::filesystem::directory_iterator(dirpath)) {
            if (entry.is_regular_file()) {
                std::string path_str = entry.path().string();
                std::string fname = entry.path().filename().string();
                std::string ext = entry.path().extension().string();

                if (ext != ".txt" && ext != ".json") continue;

                if (!file_prefixes.empty()) {
                    bool match = false;
                    for (const auto& pfx : file_prefixes) {
                        if (fname.rfind(pfx, 0) == 0) { match = true; break; }
                    }
                    if (!match) continue;
                }

                ingest_file(path_str, stats);
            }
        }

        // Commit all accumulated domain triples to AnalogyEngine
        commit_domains(stats);

        auto end_time = std::chrono::high_resolution_clock::now();
        stats.elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        if (stats.elapsed_ms > 0.0) {
            stats.throughput_facts_per_sec = (stats.facts_ingested * 1000.0) / stats.elapsed_ms;
        }

        return stats;
    }

    /**
     * Parse and route a single statement or tagged fact
     */
    void ingest_line(const std::string& line, const std::string& domain, IngestionStats& stats) {
        if (!brain_) return;

        std::string target_line = line;

        // 1. Natural language pair format: "the box has mass 5 => FACT: box | mass | 5"
        size_t arrow_pos = target_line.find("=>");
        if (arrow_pos != std::string::npos) {
            target_line = trim(target_line.substr(arrow_pos + 2));
        }

        // 2. Tagged: "FACT: subj | rel | obj"
        if (target_line.rfind("FACT:", 0) == 0) {
            std::string payload = trim(target_line.substr(5));
            auto parts = split(payload, '|');
            if (parts.size() >= 3) {
                std::string s = parts[0];
                std::string r = parts[1];
                std::string o = parts[2];
                learn_triple(s, r, o, domain, stats);
            }
            return;
        }

        // 3. Tagged: "ISA: subj | parent"
        if (target_line.rfind("ISA:", 0) == 0) {
            std::string payload = trim(target_line.substr(4));
            auto parts = split(payload, '|');
            if (parts.size() >= 2) {
                std::string s = parts[0];
                std::string parent = parts[1];
                brain_->brainql_engine.learn(s, "isa", parent);
                stats.facts_ingested++;
                stats.isa_relations_ingested++;
                pending_domain_triples_[domain].push_back({s, "isa", parent});
            }
            return;
        }

        // 4. Tagged: "LAW: name | rel | formula"
        if (target_line.rfind("LAW:", 0) == 0) {
            std::string payload = trim(target_line.substr(4));
            auto parts = split(payload, '|');
            if (parts.size() >= 3) {
                std::string name = parts[0];
                std::string rel = parts[1];
                std::string formula = parts[2];
                brain_->brainql_engine.learn(name, rel, formula);
                stats.facts_ingested++;
                stats.rules_ingested++;
                pending_domain_triples_[domain].push_back({name, rel, formula});
            }
            return;
        }

        // 5. Tagged: "EVENT:" or "PROP:" or "PART:"
        if (target_line.rfind("EVENT:", 0) == 0 || target_line.rfind("PROP:", 0) == 0 || target_line.rfind("PART:", 0) == 0) {
            size_t colon = target_line.find(':');
            std::string payload = trim(target_line.substr(colon + 1));
            auto parts = split(payload, '|');
            if (parts.size() >= 3) {
                learn_triple(parts[0], parts[1], parts[2], domain, stats);
            } else if (parts.size() == 2) {
                learn_triple(parts[0], "has_part", parts[1], domain, stats);
            }
            return;
        }

        // 6. Raw natural language extraction: "X is a Y", "X causes Y", "X has Y of Z"
        extract_from_natural_text(target_line, domain, stats);
    }

    void commit_all_domains(IngestionStats& stats) {
        commit_domains(stats);
    }

private:
    void learn_triple(const std::string& s, const std::string& r, const std::string& o, const std::string& domain, IngestionStats& stats) {
        // Learn in crisp BrainQL knowledge engine
        brain_->brainql_engine.learn(s, r, o);
        stats.facts_ingested++;

        // If object is numeric, learn in crisp numeric store
        double num_val = 0.0;
        if (is_number(o, num_val)) {
            brain_->teach_fact(s, r, num_val);
        }

        // Queue for AnalogyEngine domain registry
        if (!domain.empty()) {
            pending_domain_triples_[domain].push_back({s, r, o});
        }
    }

    void extract_from_natural_text(const std::string& text, const std::string& domain, IngestionStats& stats) {
        std::string lower = to_lower(text);

        // Pattern 1: "X is a Y" / "X is an Y"
        size_t is_a_pos = lower.find(" is a ");
        if (is_a_pos != std::string::npos && is_a_pos > 0) {
            std::string subj = trim(text.substr(0, is_a_pos));
            std::string obj = trim(text.substr(is_a_pos + 6));
            if (!subj.empty() && !obj.empty() && subj.find(' ') == std::string::npos) {
                brain_->brainql_engine.learn(subj, "isa", obj);
                stats.facts_ingested++;
                stats.isa_relations_ingested++;
                pending_domain_triples_[domain].push_back({subj, "isa", obj});
                return;
            }
        }

        // Pattern 2: "X causes Y"
        size_t causes_pos = lower.find(" causes ");
        if (causes_pos != std::string::npos && causes_pos > 0) {
            std::string subj = trim(text.substr(0, causes_pos));
            std::string obj = trim(text.substr(causes_pos + 8));
            if (!subj.empty() && !obj.empty()) {
                learn_triple(subj, "causes", obj, domain, stats);
                return;
            }
        }

        // Pattern 3: "X produces Y" or "X requires Y"
        size_t prod_pos = lower.find(" produces ");
        if (prod_pos != std::string::npos && prod_pos > 0) {
            std::string subj = trim(text.substr(0, prod_pos));
            std::string obj = trim(text.substr(prod_pos + 10));
            if (!subj.empty() && !obj.empty()) {
                learn_triple(subj, "produces", obj, domain, stats);
                return;
            }
        }
    }

    void commit_domains(IngestionStats& stats) {
        for (const auto& kv : pending_domain_triples_) {
            const std::string& domain_name = kv.first;
            const auto& triples = kv.second;
            if (triples.empty()) continue;

            brain2::reasoning::DomainModel model;
            model.name = domain_name;
            for (const auto& t : triples) {
                model.add_triple(t.subj, t.rel, t.obj);
            }
            brain_->analogy_engine.define_domain(model);
            stats.domains_registered++;
        }
        pending_domain_triples_.clear();
    }

    static std::string infer_domain_from_filename(const std::string& fname) {
        std::string l = to_lower(fname);
        if (l.find("calculus") != std::string::npos) return "calculus";
        if (l.find("mechanic") != std::string::npos || l.find("physics") != std::string::npos) return "mechanics";
        if (l.find("chemistry") != std::string::npos) return "chemistry";
        if (l.find("biology") != std::string::npos || l.find("cell") != std::string::npos) return "biology";
        if (l.find("network") != std::string::npos) return "telecom_network";
        if (l.find("database") != std::string::npos) return "database";
        if (l.find("math") != std::string::npos) return "mathematics";
        if (l.find("science") != std::string::npos) return "science";
        if (l.find("finance") != std::string::npos || l.find("market") != std::string::npos || l.find("econ") != std::string::npos) return "economics";
        if (l.find("taxonomy") != std::string::npos) return "taxonomy";
        if (l.find("kimi") != std::string::npos) return "physics_experiments";
        return "general_knowledge";
    }
};

} // namespace core
} // namespace brain3
