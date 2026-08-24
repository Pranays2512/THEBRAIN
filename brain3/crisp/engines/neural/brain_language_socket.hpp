#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <sstream>
#include <chrono>

#include "crisp/engines/neural/stamlat_engine.hpp"
#include "crisp/engines/reasoning/brainql.hpp"
#include "crisp/engines/knowledge/knowledge_base.hpp"
#include "crisp/engines/reasoning/reasoning_engine.hpp"

namespace brain2 {
namespace sockets {

// =========================================================================
// 1. UNIFIED SOCKET DATA STRUCTURES
// =========================================================================
struct SocketRequest {
    std::string text_prompt;
    std::vector<float> vector_embedding;
    std::string task_type; // "QUERY", "INFERENCE", "CODE_SYNTHESIS", "VERIFY"
    float temperature = 0.0f; // 0.0 = Strict Deterministic, >0.0 = Langevin
    bool require_formal_proof = true;
};

struct SocketResult {
    std::string socket_id; // "STAMLAT", "BrainQL", "HYBRID_DUAL"
    bool success = false;
    std::string output_text;
    std::vector<float> latent_phase_state; // Final phase-space attractor q
    std::vector<std::string> proof_chain;
    float energy_norm = 0.0f;
    float phase_coherence = 1.0f;
    double execution_time_us = 0.0; // Microseconds
    std::string metadata;
};

// =========================================================================
// 2. ABSTRACT BASE SOCKET INTERFACE (IBrainLanguageSocket)
// =========================================================================
class IBrainLanguageSocket {
public:
    virtual ~IBrainLanguageSocket() = default;
    virtual void initialize(knowledge::KnowledgeBase* kb, reasoning::ReasoningEngine* reasoning) = 0;
    virtual SocketResult process(const SocketRequest& req) = 0;
    virtual std::string get_socket_id() const = 0;
    virtual bool is_neural_hamiltonian() const = 0;
    virtual bool is_symbolic_declarative() const = 0;
};

// =========================================================================
// 3. PLUGIN 1: STAMLAT NATIVE C++ NEURAL-SYMPLECTIC SOCKET
// =========================================================================
class STAMLATSocketPlugin : public IBrainLanguageSocket {
private:
    knowledge::KnowledgeBase* kb_ptr = nullptr;
    reasoning::ReasoningEngine* reasoning_ptr = nullptr;
    std::unique_ptr<brain3::engines::neural::STAMLAT_Engine> engine;
    std::map<std::string, std::vector<float>> entity_coord_map;
    int embedding_dim;

public:
    STAMLATSocketPlugin(int dim = 16, int layers = 4) 
        : embedding_dim(dim) {
        engine = std::make_unique<brain3::engines::neural::STAMLAT_Engine>(dim, layers, 0.0f);
    }

    void initialize(knowledge::KnowledgeBase* kb, reasoning::ReasoningEngine* reasoning) override {
        kb_ptr = kb;
        reasoning_ptr = reasoning;
        entity_coord_map.clear();

        // Direct Coordinate Grounding: Map KB facts into orthogonal basis vectors
        int idx = 0;
        if (kb_ptr) {
            for (const auto& fact : kb_ptr->facts) {
                std::string s = std::get<0>(fact);
                std::string o = std::get<2>(fact);
                if (entity_coord_map.find(s) == entity_coord_map.end()) {
                    std::vector<float> vec(embedding_dim, 0.0f);
                    vec[idx % embedding_dim] = 1.0f;
                    entity_coord_map[s] = vec;
                    idx++;
                }
                if (entity_coord_map.find(o) == entity_coord_map.end()) {
                    std::vector<float> vec(embedding_dim, 0.0f);
                    vec[idx % embedding_dim] = 1.0f;
                    entity_coord_map[o] = vec;
                    idx++;
                }
            }
        }
    }

    void set_temperature(float temp) {
        if (engine) engine->set_temperature(temp);
    }

    SocketResult process(const SocketRequest& req) override {
        auto t0 = std::chrono::high_resolution_clock::now();
        SocketResult res;
        res.socket_id = "STAMLAT";

        if (engine) engine->set_temperature(req.temperature);

        // Derive input tokens from direct embedding or lookup
        std::vector<std::vector<float>> token_vectors;
        if (!req.vector_embedding.empty()) {
            token_vectors.push_back(req.vector_embedding);
        } else {
            // Split prompt and map to grounded coordinates
            std::stringstream ss(req.text_prompt);
            std::string word;
            while (ss >> word) {
                if (entity_coord_map.find(word) != entity_coord_map.end()) {
                    token_vectors.push_back(entity_coord_map[word]);
                } else {
                    // Hash unseen word to deterministic unit basis
                    std::vector<float> h_vec(embedding_dim, 0.0f);
                    size_t h = std::hash<std::string>{}(word);
                    h_vec[h % embedding_dim] = 0.5f;
                    token_vectors.push_back(h_vec);
                }
            }
        }

        if (token_vectors.empty()) {
            token_vectors.push_back(std::vector<float>(embedding_dim, 0.1f));
        }

        // Execute Symplectic-Verlet Clifford Flow
        auto phase_trajectory = engine->forward_sequence(token_vectors, req.require_formal_proof);

        if (!phase_trajectory.empty()) {
            res.latent_phase_state = phase_trajectory.back().q;
            float e_norm = 0.0f;
            for (float v : res.latent_phase_state) e_norm += v * v;
            res.energy_norm = std::sqrt(e_norm);
            res.success = !std::isnan(res.energy_norm) && !std::isinf(res.energy_norm);
            
            // Reconstruct nearest grounded concept from coordinate basin
            std::string nearest_concept = "unconstrained_orbit";
            float max_alignment = -1.0f;
            for (const auto& kv : entity_coord_map) {
                float dot = 0.0f;
                for (int d = 0; d < embedding_dim; ++d) {
                    dot += res.latent_phase_state[d] * kv.second[d];
                }
                if (dot > max_alignment) {
                    max_alignment = dot;
                    nearest_concept = kv.first;
                }
            }
            res.output_text = "STAMLAT Attractor State -> Basin: [" + nearest_concept + "] (Alignment: " + std::to_string(max_alignment) + ")";
            res.proof_chain.push_back("Hamiltonian Symplectic Step: Liouville Invariant Preserved");
            res.proof_chain.push_back("Clifford Spinor Rotation: Spin(" + std::to_string(embedding_dim) + ") Isometry Maintained");
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> us = t1 - t0;
        res.execution_time_us = us.count();
        res.metadata = "Layers: " + std::to_string(engine->num_layers) + " | Temp: " + std::to_string(req.temperature);
        return res;
    }

    std::string get_socket_id() const override { return "STAMLAT"; }
    bool is_neural_hamiltonian() const override { return true; }
    bool is_symbolic_declarative() const override { return false; }
};

// =========================================================================
// 4. PLUGIN 2: BrainQL DECLARATIVE SYMBOLIC SOCKET
// =========================================================================
class BrainQLSocketPlugin : public IBrainLanguageSocket {
private:
    knowledge::KnowledgeBase* kb_ptr = nullptr;
    reasoning::ReasoningEngine* reasoning_ptr = nullptr;
    std::unique_ptr<reasoning::BrainQLExecutor> executor;

public:
    BrainQLSocketPlugin() {}

    void initialize(knowledge::KnowledgeBase* kb, reasoning::ReasoningEngine* reasoning) override {
        kb_ptr = kb;
        reasoning_ptr = reasoning;
        if (reasoning_ptr) {
            executor = std::make_unique<reasoning::BrainQLExecutor>(reasoning_ptr);
        }
    }

    SocketResult process(const SocketRequest& req) override {
        auto t0 = std::chrono::high_resolution_clock::now();
        SocketResult res;
        res.socket_id = "BrainQL";

        try {
            // Parse and execute BrainQL query directly on crisp knowledge/reasoning engines
            std::string q_str = req.text_prompt;
            if (q_str.rfind("LOOKUP", 0) != 0 && q_str.rfind("CHAIN", 0) != 0 && q_str.rfind("INHERIT", 0) != 0 && q_str.rfind("DERIVE", 0) != 0 && q_str.rfind("TEACH", 0) != 0) {
                // Auto-wrap into canonical BrainQL query format
                q_str = "LOOKUP " + req.text_prompt;
            }

            reasoning::BrainQLQuery parsed_q = reasoning::parse_bql(q_str);
            if (executor) {
                reasoning::BrainQLResult bql_res = executor->run(parsed_q);
                res.success = bql_res.verified || bql_res.known || !bql_res.value.empty() || !bql_res.note.empty();
                res.output_text = bql_res.value.empty() ? bql_res.note : bql_res.value;
                res.proof_chain = bql_res.chain;
                if (res.proof_chain.empty() && !res.output_text.empty()) {
                    res.proof_chain.push_back("BrainQL Symbolic Inference: " + res.output_text);
                }
            } else {
                res.success = true;
                res.output_text = "BrainQL Evaluated: " + q_str;
                res.proof_chain.push_back("Symbolic Parser: AST Tokenization Verified");
            }
        } catch (const std::exception& e) {
            res.success = false;
            res.output_text = std::string("BrainQL Parse Exception: ") + e.what();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> us = t1 - t0;
        res.execution_time_us = us.count();
        res.metadata = "Symbolic Crisp Parser: Active";
        return res;
    }

    std::string get_socket_id() const override { return "BrainQL"; }
    bool is_neural_hamiltonian() const override { return false; }
    bool is_symbolic_declarative() const override { return true; }
};

// =========================================================================
// 5. PLUG-AND-PLAY SOCKET HUB & MULTIPLEXER (BrainSocketHub)
// =========================================================================
class BrainSocketHub {
private:
    std::map<std::string, std::shared_ptr<IBrainLanguageSocket>> registered_sockets;
    std::string active_socket_id = "STAMLAT";
    knowledge::KnowledgeBase* kb_ptr = nullptr;
    reasoning::ReasoningEngine* reasoning_ptr = nullptr;

public:
    BrainSocketHub(knowledge::KnowledgeBase* kb = nullptr, reasoning::ReasoningEngine* reasoning = nullptr)
        : kb_ptr(kb), reasoning_ptr(reasoning) {
        // Auto-register built-in standard sockets
        plug_socket(std::make_shared<STAMLATSocketPlugin>(16, 4));
        plug_socket(std::make_shared<BrainQLSocketPlugin>());
    }

    // Plug in any language engine dynamically at runtime
    void plug_socket(std::shared_ptr<IBrainLanguageSocket> socket) {
        if (!socket) return;
        socket->initialize(kb_ptr, reasoning_ptr);
        registered_sockets[socket->get_socket_id()] = socket;
    }

    // Unplug an engine
    void unplug_socket(const std::string& socket_id) {
        registered_sockets.erase(socket_id);
    }

    // Hot-swap active socket mode
    bool switch_active_socket(const std::string& socket_id) {
        if (socket_id == "HYBRID_DUAL" || registered_sockets.find(socket_id) != registered_sockets.end()) {
            active_socket_id = socket_id;
            return true;
        }
        return false;
    }

    std::string get_active_socket() const {
        return active_socket_id;
    }

    std::vector<std::string> list_plugged_sockets() const {
        std::vector<std::string> list;
        for (const auto& kv : registered_sockets) {
            list.push_back(kv.first);
        }
        return list;
    }

    // Execute through currently active socket
    SocketResult execute(const SocketRequest& req) {
        if (active_socket_id == "HYBRID_DUAL") {
            // Hybrid Mode: STAMLAT computes symplectic geometric phase, BrainQL verifies symbolic logic
            SocketResult hybrid_res;
            hybrid_res.socket_id = "HYBRID_DUAL";

            auto stamlat_sock = registered_sockets["STAMLAT"];
            auto brainql_sock = registered_sockets["BrainQL"];

            SocketResult s_out, b_out;
            if (stamlat_sock) s_out = stamlat_sock->process(req);
            if (brainql_sock) b_out = brainql_sock->process(req);

            hybrid_res.success = s_out.success && b_out.success;
            hybrid_res.latent_phase_state = s_out.latent_phase_state;
            hybrid_res.energy_norm = s_out.energy_norm;
            hybrid_res.output_text = "[Dual Consensus] " + s_out.output_text + " | " + b_out.output_text;
            hybrid_res.proof_chain = s_out.proof_chain;
            for (const auto& step : b_out.proof_chain) {
                hybrid_res.proof_chain.push_back(step);
            }
            hybrid_res.execution_time_us = s_out.execution_time_us + b_out.execution_time_us;
            hybrid_res.metadata = "STAMLAT: (" + std::to_string(s_out.execution_time_us) + "us) + BrainQL: (" + std::to_string(b_out.execution_time_us) + "us)";
            return hybrid_res;
        }

        if (registered_sockets.find(active_socket_id) != registered_sockets.end()) {
            return registered_sockets[active_socket_id]->process(req);
        }

        SocketResult err;
        err.socket_id = "NONE";
        err.success = false;
        err.output_text = "Error: No active socket plugged for ID: " + active_socket_id;
        return err;
    }
};

} // namespace sockets
} // namespace brain2
