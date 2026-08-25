// tools/swarm_train.cpp — SWARM SELF-PLAY TRAINING
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "crisp/engines/reasoning/graph_attention_reasoner.hpp"

using G = brain3::engines::reasoning::GraphAttentionReasoner;

int main(int argc, char** argv) {
    auto t0 = std::chrono::steady_clock::now();
    int num_agents = argc > 1 ? atoi(argv[1]) : 10;
    int per_agent = argc > 2 ? atoi(argv[2]) : 200;
    int train_steps = argc > 3 ? atoi(argv[3]) : 3000;

    std::cout << "=== SWARM SELF-PLAY ===\n";
    std::cout << "Agents: " << num_agents << " x " << per_agent << "\n\n";

    G g;
    auto R_relates = g.add_relation("relates_to");
    auto R_isa = g.add_relation("isa");

    // Build knowledge graph — quantum + AI + math + physics chains
    struct Edge { const char* s; const char* r; const char* t; };
    std::vector<Edge> edges;
    
    // Quantum chain
    edges.push_back({"quantum_superposition","has_property","multiple_states"});
    edges.push_back({"entanglement","links","spacetime_separated"});
    edges.push_back({"tunneling","penetrates","barriers"});
    edges.push_back({"wave_function","describes","probability_amplitude"});
    edges.push_back({"schrodinger_eq","governs","wave_function_evolution"});
    edges.push_back({"heisenberg_uncertainty","limits","measurement_precision"});
    edges.push_back({"decoherence_destroys","superposition_states","via_environment_coupling"});
    edges.push_back({"qubit_concept","is_quantum_version_of","classical_bit"});
    edges.push_back({"shors_algorithm","factors_integers","exponentially_faster"});
    edges.push_back({"grovers_algorithm","searches_database","quadratically_faster"});
    edges.push_back({"bell_inequality_violation","proves_nonlocality","of_quantum_mechanics"});
    edges.push_back({"no_cloning_theorem","forbids_copying","unknown_quantum_states"});
    
    // AI/ML chain  
    edges.push_back({"neural_network_ml","learns_by_adjusting","weights_to_minimize_error"});
    edges.push_back({"backpropagation_algorithm","computes_gradients_through","network_layers"});
    edges.push_back({"gradient_descent_opt","minimizes_loss","stepping_opposite_gradient"});
    edges.push_back({"adam_optimizer","combines_momentum_and_rmsprop","adaptive_learning_rates"});
    edges.push_back({"transformer_architecture","uses_self_attention","instead_of_recurrence"});
    edges.push_back({"attention_weights_importance","dynamically_scores","input_tokens"});
    edges.push_back({"gpt_autoregressive_model","predicts_next_token","given_context_window"});
    edges.push_back({"bert_masked_lm","understands_bidirectional_context","via_masked_tokens"});
    edges.push_back({"cnn_spatial_filters","extract_features","from_image_data"});
    edges.push_back({"lstm_gates","control_information_flow","through_timesteps"});
    edges.push_back({"reinforcement_learning_agent","maximizes","cumulative_reward_signal"});
    edges.push_back({"transfer_learning_method","reuses_pretrained_knowledge","for_new_tasks"});
    edges.push_back({"overfitting_memorizes_training_data","fails_on_new_data","generalization_failure"});
    
    // Math chain
    edges.push_back({"euler_identity_math","connects_five_constants","e_pi_i_and_minus_one"});
    edges.push_back({"gauss_bonnet_theorem","relates_curvature_to","topology_invariant"});
    edges.push_back({"goedel_incompleteness_theorem","proves_any_formal_system_has","true_but_unprovable_statements"});
    edges.push_back({"cantor_diagonal_argument","proves_reals_are","uncountable_infinity"});
    edges.push_back({"riemann_hypothesis_conjecture","concerns_zeros_of","zeta_function"});
    edges.push_back({"central_limit_theorem_stats","sample_means_approach","normal_distribution"});
    edges.push_back({"bayes_theorem_updates","posterior_beliefs_from","prior_and_likelihood"});
    edges.push_back({"eigenvalue_algebra","characterizes","linear_transformation_directions"});
    edges.push_back({"group_theory_structure","studies","symmetry_operations"});
    edges.push_back({"topology_deformation","preserves_connectivity","not_distance"});
    edges.push_back({"turing_machine_model","defines_computability","on_infinite_tape"});
    edges.push_back({"halting_problem_proof","shows_no_algorithm","decides_all_halting_cases"});
    edges.push_back({"p_vs_np_question","asks_if_verification_equals","discovery_complexity"});
    edges.push_back({"proof_by_induction","establishes_base_case","then_inductive_step"});
    edges.push_back({"modular_arithmetic_clock","numbers_wrap_around","after_reaching_modulus"});
    
    // Physics chain
    edges.push_back({"newton_first_law_motion","objects_at_rest_stay_at_rest","without_external_force"});
    edges.push_back({"newton_second_law_motion","force_equals","mass_times_acceleration_vector"});
    edges.push_back({"newton_third_law_motion","every_action_produces","equal_opposite_reaction_force"});
    edges.push_back({"gravity_law_attracts","all_masses_toward","each_other_proportionally"});
    edges.push_back({"energy_conservation_principle","total_energy_remains_constant","in_closed_systems"});
    edges.push_back({"entropy_always_increases","in_any_isolated_system","per_second_law_thermodynamics"});
    edges.push_back({"light_speed_limitation","nothing_travels_faster_than","299792458_mps_in_vacuum"});
    edges.push_back({"atom_nucleus_electrons","nucleus_contains_protons_neutrons","electrons_orbit_in_clouds"});
    edges.push_back({"radioactivity_decay_process","unstable_nuclei_emit","particles_or_radiation"});
    edges.push_back({"quantum_entanglement_pairs","measuring_one_instantly_affects","the_other_regardless_distance"});
    edges.push_back({"wave_particle_duality_light","exhibits_both_wave_and","particle_properties"});
    edges.push_back({"photoelectric_effect_photon","light_transfers_energy_in","discrete_quanta_packages"});
    
    for (auto& e : edges) {
        int h = g.add_entity(e.s);
        g.add_relation(e.r);
        int t = g.add_entity(e.t);
        int rr = g.add_relation(e.r);
        g.add_edge(h, rr, t);
    }
    (void)R_relates; (void)R_isa;

    std::cout << "Graph: " << g.entity_count() << " entities, "
              << g.edge_count() << " edges\n\n";

    // ── PHASE 1: TRAIN ────────────────────────────────────────────────────
    std::cout << "[TRAIN] Training graph reasoner...\n";
    G::TrainConfig tc;
    tc.steps = train_steps;
    tc.batch = 16;
    tc.max_path_len = 3;
    g.train(tc);

    double mrr_before = g.self_check_mrr();
    std::cout << "  Self-check MRR: " << mrr_before << "\n\n";

    // ── PHASE 2: SWARM INTERACTION ────────────────────────────────────────
    std::cout << "[SWARM] " << num_agents << " agents × "
              << per_agent << " interactions\n";

    std::mt19937 rng(20260825);
    std::uniform_int_distribution<int> pick_entity(0, g.entity_count()-1);
    std::uniform_int_distribution<int> pick_agent_action(0, 4);

    for (int agent = 0; agent < num_agents; ++agent) {
        int correct = 0, total = 0;
        for (int i = 0; i < per_agent; ++i) {
            int src = pick_entity(rng);
            auto qr = g.query_stages(src, {-1}, 1.0);
            if (!qr.ranked.empty()) ++correct;
            ++total;

            // Teach: add cross-links occasionally
            if (i % 20 == 19 && qr.ranked.size() >= 2) {
                int t1 = qr.ranked[0].entity;
                int t2 = qr.ranked[qr.ranked.size()-1].entity;
                if (t1 != t2 && t1 < g.entity_count() && t2 < g.entity_count())
                    g.add_edge(t1, R_relates, t2);
            }
        }
        std::cout << "  Agent " << agent << ": answered "
                  << correct << "/" << total << "\n";
    }

    // ── PHASE 3: RETRAIN ON ACCUMULATED GRAPH ─────────────────────────────
    std::cout << "\n[RETRAIN] Consolidating accumulated knowledge...\n";
    G::TrainConfig rc;
    rc.steps = train_steps / 2;
    rc.batch = 16;
    g.train(rc);
    double mrr_after = g.self_check_mrr();
    std::cout << "  MRR after consolidation: " << mrr_after << "\n";

    auto t1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nDone in " << elapsed << "s\n";
    return 0;
}
