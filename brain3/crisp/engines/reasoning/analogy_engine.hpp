#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>

namespace brain2 {
namespace reasoning {

struct DomainTriple {
    std::string subj;
    std::string rel;
    std::string obj;

    std::string to_string() const {
        return subj + " " + rel + " " + obj;
    }
};

struct DomainModel {
    std::string name;
    std::vector<DomainTriple> triples;
    std::unordered_set<std::string> entities;
    std::unordered_set<std::string> relations;

    void add_triple(const std::string& s, const std::string& r, const std::string& o) {
        triples.push_back({s, r, o});
        entities.insert(s);
        entities.insert(o);
        relations.insert(r);
    }
};

struct CandidateInference {
    std::string target_subj;
    std::string target_rel;
    std::string target_obj;
    std::string source_origin;

    std::string to_string() const {
        return target_subj + " " + target_rel + " " + target_obj;
    }
};

struct AnalogyResult {
    bool success = false;
    std::string source_domain;
    std::string target_domain;
    double score = 0.0;
    std::map<std::string, std::string> entity_map; // source -> target
    std::vector<std::pair<std::string, std::string>> matched_triples; // (source, target)
    std::vector<CandidateInference> candidate_inferences;
    std::string explanation;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"verified\": " << (success ? "true" : "false") << ",\n";
        oss << "  \"source_domain\": \"" << source_domain << "\",\n";
        oss << "  \"target_domain\": \"" << target_domain << "\",\n";
        oss << "  \"structural_score\": " << score << ",\n";
        oss << "  \"entity_mappings\": {";
        size_t idx = 0;
        for (const auto& kv : entity_map) {
            oss << "\"" << kv.first << "\": \"" << kv.second << "\"";
            if (++idx < entity_map.size()) oss << ", ";
        }
        oss << "},\n";
        oss << "  \"matched_triples\": [\n";
        for (size_t i = 0; i < matched_triples.size(); ++i) {
            oss << "    {\"source\": \"" << matched_triples[i].first << "\", \"target\": \"" << matched_triples[i].second << "\"}";
            if (i + 1 < matched_triples.size()) oss << ",";
            oss << "\n";
        }
        oss << "  ],\n";
        oss << "  \"candidate_inferences\": [\n";
        for (size_t i = 0; i < candidate_inferences.size(); ++i) {
            oss << "    {\"derived_fact\": \"" << candidate_inferences[i].to_string() 
                << "\", \"source_origin\": \"" << candidate_inferences[i].source_origin << "\"}";
            if (i + 1 < candidate_inferences.size()) oss << ",";
            oss << "\n";
        }
        oss << "  ],\n";
        oss << "  \"explanation\": \"" << explanation << "\"\n";
        oss << "}";
        return oss.str();
    }
};

struct MatchHypothesis {
    size_t src_idx;
    size_t tgt_idx;
    std::string s_subj, s_obj;
    std::string t_subj, t_obj;
    std::string rel;
};

class AnalogyEngine {
private:
    std::unordered_map<std::string, DomainModel> domains;

public:
    AnalogyEngine() {
        preload_canonical_domains();
    }

    void preload_canonical_domains() {
        // 1. Solar System Domain
        DomainModel solar;
        solar.name = "solar_system";
        solar.add_triple("sun", "attracts", "planet");
        solar.add_triple("sun", "mass_greater", "planet");
        solar.add_triple("planet", "revolves_around", "sun");
        solar.add_triple("sun", "is_center", "solar_system");
        solar.add_triple("gravity", "causes", "planet_orbit");
        domains["solar_system"] = solar;

        // 2. Rutherford-Bohr Atomic Model
        DomainModel atom;
        atom.name = "rutherford_atom";
        atom.add_triple("nucleus", "attracts", "electron");
        atom.add_triple("nucleus", "mass_greater", "electron");
        atom.add_triple("nucleus", "is_center", "atom");
        domains["rutherford_atom"] = atom;
        domains["atom"] = atom;

        // 3. Hydraulic Water Flow Domain
        DomainModel hydraulic;
        hydraulic.name = "hydraulic_system";
        hydraulic.add_triple("pipe", "carries", "water");
        hydraulic.add_triple("pump", "causes", "pressure_difference");
        hydraulic.add_triple("pressure_difference", "causes", "water_flow");
        hydraulic.add_triple("narrow_pipe", "causes", "resistance");
        hydraulic.add_triple("valve", "controls", "water_flow");
        domains["hydraulic_system"] = hydraulic;
        domains["water_pipe"] = hydraulic;

        // 4. Electrical Circuit Domain
        DomainModel electric;
        electric.name = "electric_circuit";
        electric.add_triple("wire", "carries", "current");
        electric.add_triple("battery", "causes", "voltage_difference");
        electric.add_triple("resistor", "causes", "electrical_resistance");
        electric.add_triple("switch", "controls", "current_flow");
        domains["electric_circuit"] = electric;
        domains["circuit"] = electric;

        // 5. Biological Cell Domain
        DomainModel cell;
        cell.name = "biological_cell";
        cell.add_triple("nucleus", "contains", "genetic_code");
        cell.add_triple("ribosome", "produces", "protein");
        cell.add_triple("mitochondria", "generates", "atp_energy");
        cell.add_triple("membrane", "controls", "transport");
        domains["biological_cell"] = cell;
        domains["cell"] = cell;

        // 6. Industrial Factory Domain
        DomainModel factory;
        factory.name = "factory";
        factory.add_triple("headquarters", "contains", "blueprints");
        factory.add_triple("assembly_line", "produces", "manufactured_goods");
        factory.add_triple("powerhouse", "generates", "electric_power");
        factory.add_triple("security_gate", "controls", "transport");
        domains["factory"] = factory;

        // 7. Thermodynamics Domain
        DomainModel thermo;
        thermo.name = "thermodynamics";
        thermo.add_triple("temperature_difference", "drives", "heat_transfer");
        thermo.add_triple("thermal_insulation", "limits", "heat_transfer");
        thermo.add_triple("heat_source", "flows_to", "heat_sink");
        domains["thermodynamics"] = thermo;

        // 8. Market Economics Domain
        DomainModel econ;
        econ.name = "market_economics";
        econ.add_triple("price_difference", "drives", "trade_flow");
        econ.add_triple("import_tariff", "limits", "trade_flow");
        econ.add_triple("exporting_region", "flows_to", "importing_region");
        domains["market_economics"] = econ;
        domains["economics"] = econ;

        // 9. Organic Chemistry Synthesis Domain
        DomainModel chem;
        chem.name = "organic_synthesis";
        chem.add_triple("reactants", "transformed_by", "catalyst");
        chem.add_triple("intermediate_state", "reduces", "activation_energy");
        chem.add_triple("purification_step", "produces", "pure_target_compound");
        domains["organic_synthesis"] = chem;
        domains["chemistry"] = chem;

        // 10. Compiler Optimization Pipeline Domain
        DomainModel compiler;
        compiler.name = "compiler_pipeline";
        compiler.add_triple("source_ast", "transformed_by", "optimizer_pass");
        compiler.add_triple("intermediate_representation", "reduces", "register_pressure");
        compiler.add_triple("code_generation_step", "produces", "machine_binary");
        domains["compiler_pipeline"] = compiler;
        domains["compiler"] = compiler;

        // 11. Cardiovascular Hemodynamics Domain
        DomainModel cardio;
        cardio.name = "cardiovascular_system";
        cardio.add_triple("heart", "pumps", "blood");
        cardio.add_triple("blood_vessels", "deliver", "oxygen");
        cardio.add_triple("vascular_constriction", "increases", "flow_resistance");
        domains["cardiovascular_system"] = cardio;
        domains["cardio"] = cardio;

        // 12. Packet Switched Computer Network Domain
        DomainModel net;
        net.name = "packet_network";
        net.add_triple("network_router", "pumps", "data_packets");
        net.add_triple("fiber_links", "deliver", "bandwidth");
        net.add_triple("buffer_congestion", "increases", "network_latency");
        domains["packet_network"] = net;
        domains["network"] = net;

        // 13. Deep Transformer Architecture Domain
        DomainModel tf;
        tf.name = "transformer_architecture";
        tf.add_triple("attention_layer", "computes", "quadratic_matrix_products");
        tf.add_triple("kv_cache", "causes", "memory_bandwidth_wall");
        tf.add_triple("gpu_cluster", "consumes", "megawatt_power");
        domains["transformer_architecture"] = tf;
        domains["transformer"] = tf;
        domains["llm"] = tf;

        // 14. Neuromorphic Holographic Cortical Language Domain
        DomainModel neuro;
        neuro.name = "neuromorphic_holographic_cortex";
        neuro.add_triple("cortical_column", "computes", "holographic_vector_resonance");
        neuro.add_triple("associative_state", "enables", "constant_o_1_inference");
        neuro.add_triple("biological_brain", "consumes", "twenty_watt_power");
        domains["neuromorphic_holographic_cortex"] = neuro;
        domains["neuromorphic_cortex"] = neuro;
        domains["holographic_cortex"] = neuro;
    }

    const std::unordered_map<std::string, DomainModel>& get_domains() const {
        return domains;
    }

    void define_domain(const DomainModel& model) {
        domains[model.name] = model;
    }

    void define_domain(const std::string& name, const std::vector<DomainTriple>& triples) {
        DomainModel m;
        m.name = name;
        for (const auto& t : triples) {
            m.add_triple(t.subj, t.rel, t.obj);
        }
        domains[name] = m;
    }

    AnalogyResult map_domains(const std::string& src, const std::string& tgt) {
        return map_analogy(src, tgt);
    }

    void define_triple(const std::string& domain_name, const std::string& s, const std::string& r, const std::string& o) {
        domains[domain_name].name = domain_name;
        domains[domain_name].add_triple(s, r, o);
    }

    bool has_domain(const std::string& domain_name) const {
        return domains.find(domain_name) != domains.end();
    }

    const DomainModel* get_domain(const std::string& domain_name) const {
        auto it = domains.find(domain_name);
        if (it != domains.end()) return &it->second;
        return nullptr;
    }

    // Gentner's Structure Mapping Engine (SME)
    AnalogyResult map_analogy(const std::string& src_name, const std::string& tgt_name) {
        AnalogyResult res;
        res.source_domain = src_name;
        res.target_domain = tgt_name;

        const DomainModel* src = get_domain(src_name);
        const DomainModel* tgt = get_domain(tgt_name);

        if (!src || !tgt) {
            res.success = false;
            res.explanation = "Error: One or both domains not found.";
            return res;
        }

        // 1. Generate Match Hypotheses based on relational identity
        std::vector<MatchHypothesis> hypotheses;
        for (size_t si = 0; si < src->triples.size(); ++si) {
            const auto& st = src->triples[si];
            for (size_t ti = 0; ti < tgt->triples.size(); ++ti) {
                const auto& tt = tgt->triples[ti];
                if (st.rel == tt.rel) {
                    hypotheses.push_back({si, ti, st.subj, st.obj, tt.subj, tt.obj, st.rel});
                }
            }
        }

        if (hypotheses.empty()) {
            res.success = false;
            res.explanation = "No shared relations found between [" + src_name + "] and [" + tgt_name + "].";
            return res;
        }

        // 2. Exact Combinatorial Branch-and-Bound for Maximal Consistent Systematic Mapping
        std::vector<size_t> best_subset;
        double best_score = -1.0;

        std::vector<size_t> current_subset;
        std::map<std::string, std::string> s_to_t;
        std::map<std::string, std::string> t_to_s;
        std::set<size_t> used_src_triples;
        std::set<size_t> used_tgt_triples;

        auto evaluate_subset = [&](const std::vector<size_t>& sub) -> double {
            if (sub.empty()) return 0.0;
            double score = sub.size() * 10.0;
            // Systematicity bonus: reward shared connected entities
            for (size_t i = 0; i < sub.size(); ++i) {
                for (size_t j = i + 1; j < sub.size(); ++j) {
                    const auto& h1 = hypotheses[sub[i]];
                    const auto& h2 = hypotheses[sub[j]];
                    // Check if connected in source
                    bool connected_src = (h1.s_obj == h2.s_subj || h1.s_subj == h2.s_obj || h1.s_subj == h2.s_subj || h1.s_obj == h2.s_obj);
                    bool connected_tgt = (h1.t_obj == h2.t_subj || h1.t_subj == h2.t_obj || h1.t_subj == h2.t_subj || h1.t_obj == h2.t_obj);
                    if (connected_src && connected_tgt) {
                        score += 5.0; // Systematicity bonus
                    }
                }
            }
            return score;
        };

        auto search = [&](auto& self, size_t idx) -> void {
            if (idx == hypotheses.size()) {
                double sc = evaluate_subset(current_subset);
                if (sc > best_score) {
                    best_score = sc;
                    best_subset = current_subset;
                }
                return;
            }

            // Option 1: Try including hypotheses[idx] if consistent
            const auto& h = hypotheses[idx];
            bool can_include = (used_src_triples.find(h.src_idx) == used_src_triples.end()) &&
                               (used_tgt_triples.find(h.tgt_idx) == used_tgt_triples.end());

            if (can_include) {
                // Check 1-to-1 consistency for subj
                auto it_ss = s_to_t.find(h.s_subj);
                auto it_ts = t_to_s.find(h.t_subj);
                if ((it_ss != s_to_t.end() && it_ss->second != h.t_subj) ||
                    (it_ts != t_to_s.end() && it_ts->second != h.s_subj)) {
                    can_include = false;
                }
                // Check 1-to-1 consistency for obj
                auto it_so = s_to_t.find(h.s_obj);
                auto it_to = t_to_s.find(h.t_obj);
                if ((it_so != s_to_t.end() && it_so->second != h.t_obj) ||
                    (it_to != t_to_s.end() && it_to->second != h.s_obj)) {
                    can_include = false;
                }
            }

            if (can_include) {
                // Apply changes
                current_subset.push_back(idx);
                used_src_triples.insert(h.src_idx);
                used_tgt_triples.insert(h.tgt_idx);
                bool added_subj = false, added_obj = false;
                if (s_to_t.find(h.s_subj) == s_to_t.end()) {
                    s_to_t[h.s_subj] = h.t_subj;
                    t_to_s[h.t_subj] = h.s_subj;
                    added_subj = true;
                }
                if (s_to_t.find(h.s_obj) == s_to_t.end()) {
                    s_to_t[h.s_obj] = h.t_obj;
                    t_to_s[h.t_obj] = h.s_obj;
                    added_obj = true;
                }

                self(self, idx + 1);

                // Backtrack
                if (added_obj) {
                    s_to_t.erase(h.s_obj);
                    t_to_s.erase(h.t_obj);
                }
                if (added_subj) {
                    s_to_t.erase(h.s_subj);
                    t_to_s.erase(h.t_subj);
                }
                used_tgt_triples.erase(h.tgt_idx);
                used_src_triples.erase(h.src_idx);
                current_subset.pop_back();
            }

            // Option 2: Skip hypotheses[idx]
            self(self, idx + 1);
        };

        search(search, 0);

        // 3. Build Final Alignment Results
        std::map<std::string, std::string> final_entity_map;
        std::set<size_t> final_matched_src_indices;
        std::vector<std::pair<std::string, std::string>> final_matched_triples;

        for (size_t h_idx : best_subset) {
            const auto& h = hypotheses[h_idx];
            final_entity_map[h.s_subj] = h.t_subj;
            final_entity_map[h.s_obj] = h.t_obj;
            final_matched_src_indices.insert(h.src_idx);
            final_matched_triples.push_back({src->triples[h.src_idx].to_string(), tgt->triples[h.tgt_idx].to_string()});
        }

        // 4. Candidate Inference Projection (Gentner's Systematicity)
        // Project unmapped source triples into the target domain
        std::vector<CandidateInference> inferences;
        for (size_t si = 0; si < src->triples.size(); ++si) {
            if (final_matched_src_indices.find(si) != final_matched_src_indices.end()) continue;

            const auto& st = src->triples[si];
            auto it_s = final_entity_map.find(st.subj);
            if (it_s != final_entity_map.end()) {
                std::string proj_subj = it_s->second;
                std::string proj_rel = st.rel;
                std::string proj_obj = (final_entity_map.find(st.obj) != final_entity_map.end()) ? 
                                       final_entity_map[st.obj] : st.obj;

                // Check if already in target
                bool exists = false;
                for (const auto& tt : tgt->triples) {
                    if (tt.subj == proj_subj && tt.rel == proj_rel && tt.obj == proj_obj) {
                        exists = true;
                        break;
                    }
                }

                if (!exists) {
                    inferences.push_back({proj_subj, proj_rel, proj_obj, st.to_string()});
                }
            }
        }

        res.success = !final_entity_map.empty() && !final_matched_triples.empty();
        res.score = src->triples.empty() ? 0.0 : (double)final_matched_triples.size() / (double)std::min(src->triples.size(), tgt->triples.size());
        res.entity_map = final_entity_map;
        res.matched_triples = final_matched_triples;
        res.candidate_inferences = inferences;

        std::ostringstream exp;
        exp << "Mapped " << final_matched_triples.size() << " relational structures from [" 
            << src_name << "] to [" << tgt_name << "]. "
            << "Derived " << inferences.size() << " candidate inferences.";
        res.explanation = exp.str();

        return res;
    }
};

} // namespace reasoning
} // namespace brain2
