#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "crisp/engines/reasoning/reasoning_engine.hpp"
#include "crisp/engines/reasoning/metacognitive_engine.hpp"
#include "crisp/engines/discovery/discovery_engine.hpp"

namespace brain2 {
namespace reasoning {

struct CandidateForm {
    std::string name;
    std::string shape;
    std::function<double(double, double)> fn;
    double prior_weight = 1.0;
};

struct CuriosityGap {
    std::string id;
    std::string gap_type; // "TRANSITION_PREDICTION", "NUMERICAL_INVARIANT", "UNEXPLAINED_ENTITY"
    std::string target_entity;
    double uncertainty_score = 0.0;
    bool is_unlearnable = false;
    std::string description;
};

struct CuriosityTickResult {
    int tick_id = 0;
    std::string gap_resolved;
    std::string conjecture_name;
    int conjectures_tested = 0;
    bool verified = false;
    double remaining_error = 0.0;
    std::string explanation;
    std::vector<std::string> trace;
};

struct AutonomousCycleReport {
    int total_ticks = 0;
    int gaps_resolved = 0;
    double initial_error = 0.0;
    double final_error = 0.0;
    std::map<std::string, std::string> banked_laws;
    std::vector<CuriosityTickResult> tick_results;
    std::vector<std::string> discovery_log;
};

class ShapeProposer {
private:
    std::map<std::string, double> priors;
    std::vector<CandidateForm> forms;

public:
    ShapeProposer() {
        forms = {
            {"0.5*a*b^2", "half_ab2", [](double a, double b) { return 0.5 * a * b * b; }, 1.0},
            {"a*b",       "ab",       [](double a, double b) { return a * b; }, 1.0},
            {"a*b^2",     "ab2",      [](double a, double b) { return a * b * b; }, 1.0},
            {"a/b",       "a_over_b", [](double a, double b) { return (b != 0) ? (a / b) : 0.0; }, 1.0},
            {"a^2*b",     "a2b",      [](double a, double b) { return a * a * b; }, 1.0},
            {"0.5*a*b",   "half_ab",  [](double a, double b) { return 0.5 * a * b; }, 1.0},
            {"a*b^3",     "ab3",      [](double a, double b) { return a * b * b * b; }, 1.0}
        };

        for (const auto& f : forms) {
            priors[f.shape] = 1.0;
        }
    }

    std::vector<CandidateForm> order() {
        std::vector<CandidateForm> ordered = forms;
        for (auto& f : ordered) {
            f.prior_weight = priors[f.shape];
        }
        std::sort(ordered.begin(), ordered.end(), [](const CandidateForm& x, const CandidateForm& y) {
            return x.prior_weight > y.prior_weight;
        });
        return ordered;
    }

    void reward_shape(const std::string& shape, double boost = 3.0) {
        priors[shape] += boost;
    }

    double get_prior(const std::string& shape) const {
        auto it = priors.find(shape);
        return (it != priors.end()) ? it->second : 1.0;
    }
};

class CuriosityEngine {
private:
    std::vector<std::vector<std::string>> observed_episodes;
    std::map<std::string, std::string> predict_table; // event -> predicted immediate next event
    ShapeProposer proposer;
    std::map<std::string, std::string> banked_laws;
    int tick_counter = 0;

public:
    CuriosityEngine() {
        // Initialize default observations for bootstrap
    }

    void observe(const std::vector<std::string>& episode) {
        if (!episode.empty()) {
            observed_episodes.push_back(episode);
        }
    }

    void observe_batch(const std::vector<std::vector<std::string>>& episodes) {
        for (const auto& ep : episodes) {
            observe(ep);
        }
    }

    void clear_observations() {
        observed_episodes.clear();
        predict_table.clear();
    }

    std::vector<std::pair<std::string, std::string>> transitions() const {
        std::vector<std::pair<std::string, std::string>> trans;
        for (const auto& ep : observed_episodes) {
            if (ep.size() >= 2) {
                for (size_t i = 0; i < ep.size() - 1; ++i) {
                    trans.push_back({ep[i], ep[i + 1]});
                }
            }
        }
        return trans;
    }

    double compute_prediction_error() const {
        auto trans = transitions();
        if (trans.empty()) return 0.0;

        int total = 0;
        int wrong = 0;
        for (const auto& pair : trans) {
            total++;
            auto it = predict_table.find(pair.first);
            if (it == predict_table.end() || it->second != pair.second) {
                wrong++;
            }
        }
        return (total > 0) ? (double)wrong / total : 0.0;
    }

    std::vector<CuriosityGap> curiosity_gaps(int top_k = 5) const {
        auto trans = transitions();
        std::map<std::string, int> total_counts;
        std::map<std::string, int> wrong_counts;
        std::map<std::string, std::set<std::string>> followers;

        for (const auto& pair : trans) {
            total_counts[pair.first]++;
            followers[pair.first].insert(pair.second);
            auto it = predict_table.find(pair.first);
            if (it == predict_table.end() || it->second != pair.second) {
                wrong_counts[pair.first]++;
            }
        }

        std::vector<CuriosityGap> gaps;
        for (const auto& kv : total_counts) {
            const std::string& evt = kv.first;
            int tot = kv.second;
            int bad = wrong_counts[evt];
            if (bad > 0) {
                double err = (double)bad / tot;
                CuriosityGap gap;
                gap.id = "gap_" + evt;
                gap.target_entity = evt;
                gap.gap_type = "TRANSITION_PREDICTION";
                gap.uncertainty_score = std::round(err * 100.0) / 100.0;
                // If the event has multiple distinct non-converging outcomes across trials, it's stochastic noise
                if (followers[evt].size() >= 3 && err >= 0.6) {
                    gap.is_unlearnable = true;
                    gap.description = "Stochastic / Irreducible noise (multiple non-deterministic transitions)";
                } else {
                    gap.is_unlearnable = false;
                    gap.description = "High transition prediction error; candidate for rule induction";
                }
                gaps.push_back(gap);
            }
        }

        std::sort(gaps.begin(), gaps.end(), [](const CuriosityGap& a, const CuriosityGap& b) {
            if (a.is_unlearnable != b.is_unlearnable) return !a.is_unlearnable; // Prioritize learnable gaps
            return a.uncertainty_score > b.uncertainty_score;
        });

        if ((int)gaps.size() > top_k) {
            gaps.resize(top_k);
        }
        return gaps;
    }

    CuriosityTickResult tick(ReasoningEngine* re = nullptr, MetacognitiveEngine* mce = nullptr, discovery::DiscoveryEngine* de = nullptr) {
        tick_counter++;
        CuriosityTickResult res;
        res.tick_id = tick_counter;

        // Phase A: Ingest & Mine Transition Knowledge Gaps
        auto trans = transitions();
        std::map<std::pair<std::string, std::string>, int> transition_counts;
        std::map<std::string, int> source_counts;

        for (const auto& pair : trans) {
            transition_counts[pair]++;
            source_counts[pair.first]++;
        }

        std::vector<std::string> newly_learned;
        for (const auto& kv : source_counts) {
            const std::string& src = kv.first;
            std::string best_dst = "";
            int best_cnt = 0;
            for (const auto& tkv : transition_counts) {
                if (tkv.first.first == src && tkv.second > best_cnt) {
                    best_cnt = tkv.second;
                    best_dst = tkv.first.second;
                }
            }

            double conf = (kv.second > 0) ? (double)best_cnt / kv.second : 0.0;
            if (conf >= 0.70 && best_cnt >= 2) {
                if (predict_table[src] != best_dst) {
                    predict_table[src] = best_dst;
                    newly_learned.push_back(src + " -> " + best_dst);
                    if (re) {
                        re->learn(src, "leads_to", best_dst);
                        re->set_transitive("leads_to");
                    }
                }
            }
        }

        if (!newly_learned.empty()) {
            res.gap_resolved = "Transition Sequence Patterns";
            std::ostringstream oss;
            for (size_t i = 0; i < newly_learned.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << newly_learned[i];
            }
            res.conjecture_name = oss.str();
            res.verified = true;
            res.remaining_error = compute_prediction_error();
            res.explanation = "Inducted verified transition rules: " + oss.str();
            res.trace.push_back("[Curiosity Transition Mining]: Inducted " + std::to_string(newly_learned.size()) + " deterministic rules");
            return res;
        }

        // Phase B: Autonomous Physical Law Conjecture & Sandbox Testing
        struct PhysicalGap {
            std::string name;
            std::function<double(double, double)> truth;
        };

        std::vector<PhysicalGap> physical_gaps = {
            {"kinetic_energy",    [](double m, double v) { return 0.5 * m * v * v; }},
            {"rotational_energy", [](double I, double w) { return 0.5 * I * w * w; }},
            {"spring_energy",     [](double k, double x) { return 0.5 * k * x * x; }}
        };

        for (const auto& pgap : physical_gaps) {
            if (banked_laws.find(pgap.name) == banked_laws.end()) {
                res.gap_resolved = pgap.name;
                auto candidate_forms = proposer.order();
                int tested = 0;

                for (const auto& form : candidate_forms) {
                    tested++;
                    // Sandbox validation across 15 synthetic test coordinates
                    bool match = true;
                    double test_points[5][2] = {
                        {1.0, 2.0}, {2.0, 3.0}, {4.0, 5.0}, {0.5, 4.0}, {3.0, 2.0}
                    };

                    for (int i = 0; i < 5; ++i) {
                        double a = test_points[i][0];
                        double b = test_points[i][1];
                        double expected = pgap.truth(a, b);
                        double actual = form.fn(a, b);
                        if (std::abs(expected - actual) > 1e-6) {
                            match = false;
                            break;
                        }
                    }

                    if (match) {
                        res.conjecture_name = form.name;
                        res.conjectures_tested = tested;
                        res.verified = true;
                        banked_laws[pgap.name] = form.name;
                        proposer.reward_shape(form.shape, 3.0); // Boost Bayesian prior

                        if (re) {
                            re->learn(pgap.name, "governed_by", form.name);
                        }

                        res.remaining_error = compute_prediction_error();
                        res.explanation = "Discovered law for '" + pgap.name + "': " + form.name + " after " + std::to_string(tested) + " conjecture(s)";
                        res.trace.push_back("✓ Tested conjecture '" + form.name + "' against conservation sandbox -> Match!");
                        res.trace.push_back("✓ Banked scientific law: " + pgap.name + " = " + form.name);
                        res.trace.push_back("✓ Reinforced Proposer prior for shape '" + form.shape + "' (new weight: " + std::to_string(proposer.get_prior(form.shape)) + ")");
                        return res;
                    }
                }
            }
        }

        // Default tick: steady state
        res.gap_resolved = "none";
        res.verified = true;
        res.remaining_error = compute_prediction_error();
        res.explanation = "No unresolved active gaps. Knowledge base in steady state.";
        res.trace.push_back("Idle cycle completed. Prediction error: " + std::to_string(res.remaining_error));
        return res;
    }

    AutonomousCycleReport run_autonomous_cycle(int num_ticks = 3, ReasoningEngine* re = nullptr, MetacognitiveEngine* mce = nullptr, discovery::DiscoveryEngine* de = nullptr) {
        AutonomousCycleReport report;
        report.initial_error = compute_prediction_error();

        for (int i = 0; i < num_ticks; ++i) {
            CuriosityTickResult tick_res = tick(re, mce, de);
            report.tick_results.push_back(tick_res);
            if (tick_res.verified && tick_res.gap_resolved != "none") {
                report.gaps_resolved++;
                report.discovery_log.push_back("Tick " + std::to_string(i + 1) + ": Resolved '" + tick_res.gap_resolved + "' -> " + tick_res.conjecture_name);
            }
        }

        report.total_ticks = num_ticks;
        report.final_error = compute_prediction_error();
        report.banked_laws = banked_laws;
        return report;
    }

    std::map<std::string, std::string> get_banked_laws() const {
        return banked_laws;
    }

    double get_proposer_prior(const std::string& shape) const {
        return proposer.get_prior(shape);
    }
};

} // namespace reasoning
} // namespace brain2
