#pragma once
/**
 * brain3/core/curiosity_scheduler.hpp
 *
 * CURIOSITY-ORDERED DIET — the brain chooses what to read next.
 *
 * Ranks candidate sources by expected information gain:
 *   - sample sentences from each source (dry-run, nothing learned)
 *   - run extraction + SOM novelty on samples via a throwaway pipeline
 *     (fresh map per candidate would bias; instead we reuse ONE probe map
 *      seeded from the LIVE pipeline's recent-distance profile)
 *   - score = mean_novelty * log2(1 + extracted_triples)
 * High score = unfamiliar AND fact-dense = read me first.
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "fuzzy_ingestion.hpp"
#include "../crisp/engines/knowledge/fact_extractor.hpp"

namespace brain3 {
namespace core {

class CuriosityScheduler {
public:
    struct Scored { std::string path; double score; int sampled_triples; double mean_novelty; };

    explicit CuriosityScheduler(FuzzyIngestionPipeline* live = nullptr)
        : live_(live) {}

    std::vector<Scored> rank(const std::vector<std::string>& paths,
                             int max_samples_per_source = 8) {
        std::vector<Scored> out;
        // probe map carries the live pipeline's adaptation profile so
        // "already-known" regions score low even in a throwaway instance
        for (const auto& path : paths) {
            Scored s{path, 0.0, 0, 0.0};
            std::ifstream f(path);
            if (!f) { out.push_back(s); continue; }
            std::string line;
            FuzzyIngestionPipeline probe(8, 8, dim_probe_);
            int lines_sampled = 0;
            double nov_sum = 0.;
            while (std::getline(f, line) && lines_sampled < max_samples_per_source) {
                if (line.size() < 8) continue;
                for (auto& t : fact_extractor_.extract(line)) {
                    auto& [a, b, c] = t;
                    double n1 = probe.observe_triple(a, b, c);
                    ++s.sampled_triples;
                    nov_sum += n1;
                    ++lines_sampled;
                }
            }
            // grammar-less fallback: PER-WORD novelty against the LIVE map
            // (word scale = what the SOM organized; line scale washes out)
            if (s.sampled_triples == 0 && live_) {
                f.clear(); f.seekg(0);
                int words2 = 0;
                double dsum = 0.;
                while (std::getline(f, line) && words2 < 60) {
                    std::istringstream iss(line);
                    std::string w;
                    while (iss >> w && words2 < 60) {
                        dsum += live_->text_novelty(w);
                        ++words2;
                    }
                }
                std::cerr << "[diet-dbg] " << path << " words2=" << words2
                          << " dsum=" << dsum << "\n";
                if (words2 > 0) {
                    s.mean_novelty = dsum / words2 / 2.8f;
                    s.score = s.mean_novelty * 10.0;
                }
                out.push_back(s);
                continue;
            }
            s.score = s.mean_novelty *
                      std::log2(1.0 + (double)s.sampled_triples);
            out.push_back(s);
        }
        std::sort(out.begin(), out.end(),
                  [](const Scored& a, const Scored& b){ return a.score > b.score; });
        return out;
    }

private:
    FuzzyIngestionPipeline* live_;
    brain2::knowledge::FactExtractor fact_extractor_;
    static constexpr int dim_probe_ = 32;
};

} // namespace core
} // namespace brain3
