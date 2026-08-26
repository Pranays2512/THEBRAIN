#pragma once
/**
 * brain3/core/mcp_server.hpp
 *
 * THE BRAIN 3: MODEL CONTEXT PROTOCOL (MCP) SERVER & SOCKET BRIDGE
 *
 * Full JSON-RPC 2.0 MCP (Model Context Protocol) implementation for The Brain 3.
 * Enables external tools, agents, IDEs, robotics, and network clients to connect
 * with The Brain over Standard I/O (stdio) or TCP Socket to take actions, execute
 * queries, audit claims, solve anomalies, and access real-time knowledge resources.
 */

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <functional>
#include <algorithm>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

#include "master_orchestrator.hpp"
#include "epistemic_logical_scrutiny_engine.hpp"
#include "../crisp/engines/math/symbolic_cas_calculator_engine.hpp"

namespace brain3 {
namespace mcp {

// Simple lightweight JSON string extractor helpers (zero external dependencies)
inline std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\b') o << "\\b";
        else if (c == '\f') o << "\\f";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if ('\x00' <= c && c <= '\x1f') o << "\\u" << std::hex << (int)c;
        else o << c;
    }
    return o.str();
}

inline std::string extract_json_field(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";

    pos += needle.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t')) pos++;

    if (pos >= json.length()) return "";

    if (json[pos] == '"') {
        pos++;
        size_t end_pos = pos;
        while (end_pos < json.length() && json[end_pos] != '"') {
            if (json[end_pos] == '\\') end_pos++;
            end_pos++;
        }
        return json.substr(pos, end_pos - pos);
    } else if (json[pos] == '{' || json[pos] == '[') {
        int depth = 0;
        char open_char = json[pos];
        char close_char = (open_char == '{') ? '}' : ']';
        size_t start_pos = pos;
        while (pos < json.length()) {
            if (json[pos] == open_char) depth++;
            else if (json[pos] == close_char) {
                depth--;
                if (depth == 0) return json.substr(start_pos, pos - start_pos + 1);
            }
            pos++;
        }
        return "";
    } else {
        size_t end_pos = pos;
        while (end_pos < json.length() && json[end_pos] != ',' && json[end_pos] != '}' && json[end_pos] != ' ' && json[end_pos] != '\n') {
            end_pos++;
        }
        return json.substr(pos, end_pos - pos);
    }
}

inline std::string format_id_json(const std::string& id) {
    if (id.empty()) return "1";
    bool is_num = true;
    for (char c : id) {
        if (!isdigit(c) && c != '-') { is_num = false; break; }
    }
    if (is_num) return id;
    if (id.front() == '"' && id.back() == '"') return id;
    return "\"" + id + "\"";
}

class MCPServer {
private:
    brain3::core::MasterOrchestrator orchestrator_;
    std::mutex brain_mutex_;
    bool is_running_ = false;
    int server_fd_ = -1;

public:
    MCPServer() = default;

    ~MCPServer() {
        stop();
    }

    /**
     * Process a single incoming JSON-RPC 2.0 MCP Message and produce a response.
     */
    std::string handle_message(const std::string& raw_json) {
        std::string method = extract_json_field(raw_json, "method");
        std::string id = extract_json_field(raw_json, "id");
        std::string id_str = format_id_json(id);

        // ── 1. MCP Initialization ───────────────────────────────────────────
        if (method == "initialize") {
            std::ostringstream resp;
            resp << "{\"jsonrpc\":\"2.0\",\"id\":" << id_str << ",\"result\":{"
                 << "\"protocolVersion\":\"2024-11-05\","
                 << "\"capabilities\":{"
                 << "\"tools\":{\"listChanged\":false},"
                 << "\"resources\":{\"subscribe\":false,\"listChanged\":false},"
                 << "\"prompts\":{\"listChanged\":false}"
                 << "},"
                 << "\"serverInfo\":{"
                 << "\"name\":\"TheBrain-3-MCP-Server\","
                 << "\"version\":\"3.0.0\","
                 << "\"description\":\"High-performance cognitive C++ MCP server with epistemic anti-overclaiming and action execution.\""
                 << "}"
                 << "}}";
            return resp.str();
        }

        if (method == "notifications/initialized") {
            return ""; // No response needed for notifications
        }

        if (method == "ping") {
            return "{\"jsonrpc\":\"2.0\",\"id\":" + id_str + ",\"result\":{}}";
        }

        // ── 2. MCP Tools List ────────────────────────────────────────────────
        if (method == "tools/list") {
            std::ostringstream resp;
            resp << "{\"jsonrpc\":\"2.0\",\"id\":" << id_str << ",\"result\":{\"tools\":["
                 // Tool 1: brain_query
                 << "{"
                 << "\"name\":\"brain_query\","
                 << "\"description\":\"Query The Brain's semantic graph, BrainQL knowledge vault, or ask a natural language question.\","
                 << "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\",\"description\":\"The question or BrainQL command (e.g. 'WHAT IS entropy', 'DERIVE E=mc^2')\"}},\"required\":[\"query\"]}"
                 << "},"
                 // Tool 2: brain_teach
                 << "{"
                 << "\"name\":\"brain_teach\","
                 << "\"description\":\"Teach The Brain a new relational fact into its persistent long-term associative memory.\","
                 << "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"subject\":{\"type\":\"string\"},\"relation\":{\"type\":\"string\"},\"object\":{\"type\":\"string\"}},\"required\":[\"subject\",\"relation\",\"object\"]}"
                 << "},"
                 // Tool 3: brain_audit_claim
                 << "{"
                 << "\"name\":\"brain_audit_claim\","
                 << "\"description\":\"Run the Epistemic Adversarial Skeptic Gate on a scientific, mathematical, or architectural claim to prevent overclaiming and verify capacity/complexity bounds.\","
                 << "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"claim\":{\"type\":\"string\",\"description\":\"The claim to rigorously scrutinize and audit against fundamental laws\"}},\"required\":[\"claim\"]}"
                 << "},"
                 // Tool 4: brain_solve_anomaly
                 << "{"
                 << "\"name\":\"brain_solve_anomaly\","
                 << "\"description\":\"Run Abductive MCTS search to relax axioms and synthesize latent operators resolving a scientific crisis.\","
                 << "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"anomaly_name\":{\"type\":\"string\",\"description\":\"Name or keyword of anomaly (e.g. 'hubble_tension', 'missing_beta_decay_momentum', 'llm_transformer_compute_memory_wall_crisis')\"}},\"required\":[\"anomaly_name\"]}"
                 << "},"
                 // Tool 5: brain_analogize
                 << "{"
                 << "\"name\":\"brain_analogize\","
                 << "\"description\":\"Compute Structure Mapping Engine (SME) systematic analogy alignment between two conceptual domains.\","
                 << "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"source_domain\":{\"type\":\"string\"},\"target_domain\":{\"type\":\"string\"}},\"required\":[\"source_domain\",\"target_domain\"]}"
                 << "},"
                 // Tool 6: brain_symbolic_cas
                 << "{"
                 << "\"name\":\"brain_symbolic_cas\","
                 << "\"description\":\"Evaluate, differentiate, or simplify mathematical expressions using The Brain's native Computer Algebra System.\","
                 << "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"},\"operation\":{\"type\":\"string\",\"enum\":[\"diff\",\"eval\",\"render\"]},\"variable\":{\"type\":\"string\"}},\"required\":[\"expression\",\"operation\"]}"
                 << "},"
                 // Tool 7: brain_action_execute
                 << "{"
                 << "\"name\":\"brain_action_execute\","
                 << "\"description\":\"Execute a real action through the Brain: 'sleep' runs sleep consolidation with rollback gates; any other string is processed as a natural-language query/teach/command by the Master Orchestrator (returns reply, engine used, verified flag).\","
                 << "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"action\":{\"type\":\"string\"},\"parameters\":{\"type\":\"object\"}},\"required\":[\"action\"]}"
                 << "},"
                 // Tool 8: brain_align_ancient_modern
                 << "{"
                 << "\"name\":\"brain_align_ancient_modern\","
                 << "\"description\":\"Compute rigorous structural isomorphisms and Gentner SME alignments between ancient philosophical/cosmological systems (Nyaya logic, Vaisheshika atomism, Samkhya dualism, Advaita Maya, Nasadiya Sukta, Bhagavad Gita, Pingala combinatorics) and modern physics, AI, and mathematics.\","
                 << "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"topic\":{\"type\":\"string\",\"description\":\"Ancient or modern topic (e.g. 'samkhya', 'quantum observer', 'nasadiya', 'advaita', 'vaisheshika', 'pingala', 'gita', 'all')\"}},\"required\":[\"topic\"]}"
                 << "},"
                 // Tool 9: brain_run_agentic_task
                 << "{"
                 << "\"name\":\"brain_run_agentic_task\","
                 << "\"description\":\"Execute an autonomous multi-step agentic ReAct & Reflexion loop to plan, call tools, self-correct, and achieve a high-level goal.\","
                 << "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"goal\":{\"type\":\"string\",\"description\":\"The high-level goal to autonomously plan and execute\"},\"max_steps\":{\"type\":\"integer\",\"description\":\"Maximum number of autonomous subtasks (default: 8)\"}},\"required\":[\"goal\"]}"
                 << "}"
                 << "]}}";
            return resp.str();
        }

        // ── 3. MCP Tools Execution (tools/call) ──────────────────────────────
        if (method == "tools/call") {
            std::string tool_name = extract_json_field(raw_json, "name");
            std::string arguments_raw = extract_json_field(raw_json, "arguments");

            std::lock_guard<std::mutex> lock(brain_mutex_);
            std::string tool_output;

            if (tool_name == "brain_query") {
                std::string q = extract_json_field(arguments_raw, "query");
                if (q.empty()) q = extract_json_field(raw_json, "query");
                auto resp = orchestrator_.process(q);
                tool_output = resp.natural_reply;
                if (tool_output.empty()) tool_output = resp.raw_output;
            } else if (tool_name == "brain_teach") {
                std::string s = extract_json_field(arguments_raw, "subject");
                std::string r = extract_json_field(arguments_raw, "relation");
                std::string o = extract_json_field(arguments_raw, "object");
                orchestrator_.get_brain()->brainql_engine.learn(s, r, o);
                tool_output = "✓ Ingested fact: (" + s + " " + r + " " + o + ") into BrainQL long-term memory.";
            } else if (tool_name == "brain_run_agentic_task") {
                std::string goal = extract_json_field(arguments_raw, "goal");
                if (goal.empty()) goal = extract_json_field(raw_json, "goal");
                auto traj = orchestrator_.get_agentic_engine()->execute_goal(goal);
                tool_output = orchestrator_.get_agentic_engine()->articulate_trajectory(traj);
            } else if (tool_name == "brain_align_ancient_modern") {
                std::string topic = extract_json_field(arguments_raw, "topic");
                if (topic.empty()) topic = extract_json_field(raw_json, "topic");
                if (topic.empty()) topic = "all";
                tool_output = orchestrator_.get_ancient_alignment_engine()->articulate_alignment(topic);
            } else if (tool_name == "brain_audit_claim") {
                std::string claim = extract_json_field(arguments_raw, "claim");
                auto scrutiny = brain3::core::EpistemicLogicalScrutinyEngine::scrutinize_claim(claim);
                std::ostringstream ss;
                ss << "Epistemic Audit Verdict: [" << scrutiny.scientific_verdict_label << "]\n"
                   << scrutiny.grounded_explanation;
                tool_output = ss.str();
            } else if (tool_name == "brain_solve_anomaly") {
                std::string anomaly = extract_json_field(arguments_raw, "anomaly_name");
                brain2::discovery::AbductiveDiscoveryEngine abductive;
                auto inv = abductive.invent_latent_concept(anomaly, 100, 4);
                if (inv.success) {
                    tool_output = "🌟 [MCTS Abductive Invention Success]:\n" + inv.proof_explanation;
                } else {
                    tool_output = "⚠️ [MCTS Search]: Search finished without reducing residual error below epsilon.";
                }
            } else if (tool_name == "brain_analogize") {
                std::string src = extract_json_field(arguments_raw, "source_domain");
                std::string tgt = extract_json_field(arguments_raw, "target_domain");
                auto mapping = orchestrator_.get_brain()->analogy_engine.map_analogy(src, tgt);
                std::ostringstream ss;
                ss << "Analogical Structural Alignment: [" << src << " => " << tgt << "]\n"
                   << "• Structural Alignment Score: " << mapping.score << "\n"
                   << "• Inferred Conjectures:\n";
                for (const auto& c : mapping.candidate_inferences) {
                    ss << "  - " << c.to_string() << "\n";
                }
                tool_output = ss.str();
            } else if (tool_name == "brain_symbolic_cas") {
                std::string expr = extract_json_field(arguments_raw, "expression");
                std::string op = extract_json_field(arguments_raw, "operation");
                std::string var = extract_json_field(arguments_raw, "variable");
                auto node = thebrain::cas::SymbolicCasCalculatorEngine::parse_expression(expr);
                if (op == "diff") {
                    auto d_node = thebrain::cas::SymbolicCasCalculatorEngine::diff(node, var);
                    tool_output = "d/d" + var + "(" + expr + ") = " + thebrain::cas::SymbolicCasCalculatorEngine::render(d_node);
                } else {
                    tool_output = "Expression: " + thebrain::cas::SymbolicCasCalculatorEngine::render(node);
                }
            } else if (tool_name == "brain_action_execute") {
                std::string act = extract_json_field(arguments_raw, "action");
                if (act.empty()) {
                    tool_output = "Error: missing required 'action' argument. No action taken.";
                } else if (act == "sleep" || act == "sleep_consolidate") {
                    tool_output = "🌙 [Sleep Consolidation]\n" + orchestrator_.sleep_consolidate();
                } else {
                    // Everything else is a genuine cognitive request routed
                    // through the Master Orchestrator's dispatch ladder.
                    auto r = orchestrator_.process(act);
                    std::ostringstream ss;
                    ss << "⚡ [Brain Action] engine=" << r.engine_used
                       << " verified=" << (r.verified ? "true" : "false")
                       << " latency_ms=" << r.latency_ms << "\n"
                       << r.natural_reply;
                    tool_output = ss.str();
                }
            } else {
                tool_output = "Error: Unknown tool name '" + tool_name + "'";
            }

            std::ostringstream resp;
            resp << "{\"jsonrpc\":\"2.0\",\"id\":" << id_str << ",\"result\":{"
                 << "\"content\":[{\"type\":\"text\",\"text\":\"" << json_escape(tool_output) << "\"}]"
                 << "}}";
            return resp.str();
        }

        // ── 4. MCP Resources (resources/list & resources/read) ──────────────
        if (method == "resources/list") {
            std::ostringstream resp;
            resp << "{\"jsonrpc\":\"2.0\",\"id\":" << id_str << ",\"result\":{\"resources\":["
                 << "{\"uri\":\"brain://theorems\",\"name\":\"Audited Scientific Theorems\",\"mimeType\":\"text/plain\"},"
                 << "{\"uri\":\"brain://axioms\",\"name\":\"Axiomatic Knowledge Vault\",\"mimeType\":\"text/plain\"},"
                 << "{\"uri\":\"brain://agent_memory\",\"name\":\"Autonomous Agent Episodic Memory Logs\",\"mimeType\":\"application/json\"},"
                 << "{\"uri\":\"brain://agentic_knowledge\",\"name\":\"Agentic AI Architectures & Cognitive Loops\",\"mimeType\":\"text/plain\"},"
                 << "{\"uri\":\"brain://ancient_philosophies\",\"name\":\"Ancient Indian Shad-Darshanas & Epistemologies\",\"mimeType\":\"text/plain\"},"
                 << "{\"uri\":\"brain://vedic_cosmology\",\"name\":\"Vedic Cosmological Hymns & Upanishads\",\"mimeType\":\"text/plain\"},"
                 << "{\"uri\":\"brain://epics_and_ancient_sciences\",\"name\":\"Ancient Epics, Math, Astronomy & Comparative Philosophies\",\"mimeType\":\"text/plain\"},"
                 << "{\"uri\":\"brain://status\",\"name\":\"Real-Time Brain Status\",\"mimeType\":\"application/json\"}"
                 << "]}}";
            return resp.str();
        }

        if (method == "resources/read") {
            std::string uri = extract_json_field(raw_json, "uri");
            std::string text_content;

            if (uri == "brain://theorems") {
                text_content = "Verified Theorems: (1) Navier-Stokes Fisher Invariant, (2) Non-Hermitian EP Quantum Memory, (3) Holographic Quantum Island Hubble Tension, (4) Holographic Linear Recurrent Accumulator (H2RL - Plate 1995/RetNet).";
            } else if (uri == "brain://axioms") {
                text_content = "Axioms: Information Conservation, Carnot Thermodynamic Limits, GSSM State-Space Expressivity Bounds, Shannon-Plate Superposition SNR scaling O(sqrt(D/N)).";
            } else if (uri == "brain://agent_memory") {
                size_t total = orchestrator_.get_agentic_engine()->get_total_cycles();
                size_t succ = orchestrator_.get_agentic_engine()->get_successful_cycles();
                text_content = "{\"total_agentic_cycles\":" + std::to_string(total) + ",\"successful_cycles\":" + std::to_string(succ) + ",\"status\":\"ONLINE_READY\"}";
            } else if (uri == "brain://agentic_knowledge") {
                text_content = "Agentic AI Architectures: (1) ReAct (Reason+Act), (2) Reflexion (Self-Correction), (3) Tree of Thoughts (ToT MCTS), (4) MemGPT (Tiered Memory Paging), (5) Model Context Protocol (MCP 2024-11-05), (6) Multi-Agent Swarms & Debate.";
            } else if (uri == "brain://ancient_philosophies") {
                text_content = "Ancient Indian Shad-Darshanas: (1) Nyaya (Logic, 4 Pramanas, 5-member syllogism, Vyapti), (2) Vaisheshika (Atomism, 7 Padarthas, Paramanu, Dvyanuka, Tryanuka), (3) Samkhya (Purusha, Prakriti, 3 Gunas, 25 Tattvas), (4) Yoga (Chitta Vritti Nirodha, Ashtanga), (5) Mimamsa (Svatah-Pramanyavada, Shabda), (6) Vedanta (Advaita Vivartavada/Brahman, Vishishtadvaita, Dvaita), plus Jain Anekantavada & Buddhist Pratityasamutpada.";
            } else if (uri == "brain://vedic_cosmology") {
                text_content = "Vedic Cosmological Hymns & Upanishads: (1) Nasadiya Sukta (Rigveda 10.129 - Creation & Agnosticism), (2) Purusha Sukta (Rigveda 10.90 - Cosmic Holographic Organism), (3) Mandukya Upanishad (4 States: Jagrat, Svapna, Sushupti, Turiya), (4) Chandogya Upanishad (Tat Tvam Asi), (5) Katha Upanishad (Chariot Allegory), (6) Brihadaranyaka Upanishad (Neti-Neti).";
            } else if (uri == "brain://epics_and_ancient_sciences") {
                text_content = "Ancient Epics & Sciences: (1) Bhagavad Gita (Nishkama Karma, Sthitaprajna, Vishvarupa, Kshetra/Kshetrajna), (2) Mahabharata (Yaksha Prashna, Sanatsujatiya), (3) Yoga Vasistha (Nested Multiverse, Simulation, Time Dilation), (4) Pingala (Binary 0/1, Pascal's Triangle Meru Prastara, Fibonacci Matrameru), (5) Aryabhata & Brahmagupta (Zero, Earth rotation, Pi approximation), (6) Comparative Taoism, Stoicism & Presocratics.";
            } else if (uri == "brain://status") {
                text_content = "{\"status\":\"ACTIVE\",\"mcp_version\":\"2024-11-05\",\"engine\":\"Brain3 Native Agentic Core with ReAct Loop & Structural Alignment\"}";
            } else {
                text_content = "Resource not found: " + uri;
            }

            std::ostringstream resp;
            resp << "{\"jsonrpc\":\"2.0\",\"id\":" << id_str << ",\"result\":{"
                 << "\"contents\":[{\"uri\":\"" << uri << "\",\"mimeType\":\"text/plain\",\"text\":\"" << json_escape(text_content) << "\"}]"
                 << "}}";
            return resp.str();
        }

        // ── 5. MCP Prompts (prompts/list & prompts/get) ─────────────────────
        if (method == "prompts/list") {
            std::ostringstream resp;
            resp << "{\"jsonrpc\":\"2.0\",\"id\":" << id_str << ",\"result\":{\"prompts\":["
                 << "{\"name\":\"epistemic_audit_prompt\",\"description\":\"Prompt to scrutinize any hypothesis against information capacity and complexity bounds.\"},"
                 << "{\"name\":\"cross_domain_synthesis_prompt\",\"description\":\"Prompt to map structural isomorphisms between two domains.\"}"
                 << "]}}";
            return resp.str();
        }

        // Default: Method not found
        return "{\"jsonrpc\":\"2.0\",\"id\":" + id_str + ",\"error\":{\"code\":-32601,\"message\":\"Method not found: " + method + "\"}}";
    }

    /**
     * Runs the MCP server in Stdio Mode (reads stdin, writes stdout).
     */
    void run_stdio() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::string response = handle_message(line);
            if (!response.empty()) {
                std::cout << response << std::endl;
            }
        }
    }

    /**
     * Starts the MCP server on a TCP socket in a background thread or blocking mode.
     */
    bool start_socket_server(int port = 9999, bool block = true) {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            std::cerr << "❌ [MCP Socket Error]: Could not create socket.\n";
            return false;
        }

        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);

        if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "❌ [MCP Socket Error]: Bind failed on port " << port << "\n";
            close(server_fd_);
            return false;
        }

        if (listen(server_fd_, 10) < 0) {
            std::cerr << "❌ [MCP Socket Error]: Listen failed.\n";
            close(server_fd_);
            return false;
        }

        is_running_ = true;
        std::cout << "🌐 [TheBrain-3 MCP Server]: Listening for external MCP connections on TCP port " << port << "...\n";

        auto loop = [this]() {
            while (is_running_) {
                sockaddr_in client_addr{};
                socklen_t client_len = sizeof(client_addr);
                int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
                if (client_fd < 0) {
                    if (!is_running_) break;
                    continue;
                }

                // Handle client in a thread
                std::thread([this, client_fd]() {
                    char buffer[8192];
                    std::string stream_buffer;

                    while (is_running_) {
                        ssize_t bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
                        if (bytes_read <= 0) break;
                        buffer[bytes_read] = '\0';
                        stream_buffer += buffer;

                        // Process complete JSON-RPC lines
                        size_t newline_pos;
                        while ((newline_pos = stream_buffer.find('\n')) != std::string::npos) {
                            std::string line = stream_buffer.substr(0, newline_pos);
                            stream_buffer.erase(0, newline_pos + 1);

                            // Strip carriage returns
                            if (!line.empty() && line.back() == '\r') line.pop_back();
                            if (line.empty()) continue;

                            std::string response = handle_message(line);
                            if (!response.empty()) {
                                response += "\n";
                                send(client_fd, response.c_str(), response.length(), 0);
                            }
                        }
                    }
                    close(client_fd);
                }).detach();
            }
        };

        if (block) {
            loop();
        } else {
            std::thread(loop).detach();
        }

        return true;
    }

    void stop() {
        is_running_ = false;
        if (server_fd_ >= 0) {
            close(server_fd_);
            server_fd_ = -1;
        }
    }
};

} // namespace mcp
} // namespace brain3
