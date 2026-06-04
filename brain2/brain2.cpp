/*
 * brain2.cpp — pybind11 bindings for Brain v2
 *
 * Exposes all C++ brain components to Python.
 * Python is used ONLY for: tests, training loops, and user interface.
 * All computation stays in C++.
 *
 * Build: see CMakeLists.txt
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/attention.hpp"
#include "core/brain.hpp"
#include "core/emotion.hpp"
#include "core/episodic.hpp"
#include "core/imagination.hpp"
#include "core/language.hpp"
#include "core/predictor.hpp"
#include "core/reasoning.hpp"
#include "core/scratchpad.hpp"
#include "core/self_model.hpp"
#include "core/som.hpp"
#include "core/symbolic.hpp"
#include "core/working_mem.hpp"
// Brain V3
#include "core/basal_ganglia.hpp"
#include "core/binding_memory.hpp"
#include "core/global_workspace.hpp"
#include "core/hierarchical_predictor.hpp"
#include "core/predictive_coding.hpp"
#include "core/procedural_memory.hpp"

namespace py = pybind11;
using namespace brain2;

// Helper: numpy array → std::vector<float>
static std::vector<float> to_vec(py::array_t<float, py::array::c_style> arr) {
  auto buf = arr.request();
  if (buf.ndim != 1)
    throw std::invalid_argument("Expected 1D float array");
  const float *ptr = static_cast<const float *>(buf.ptr);
  return std::vector<float>(ptr, ptr + buf.shape[0]);
}

// Helper: std::vector<float> → numpy array
static py::array_t<float> to_np(const std::vector<float> &v) {
  py::array_t<float> arr((py::ssize_t)v.size());
  std::copy(v.begin(), v.end(), arr.mutable_data());
  return arr;
}

PYBIND11_MODULE(brain2, m) {
  m.doc() = "Brain v2 — general neural brain, C++ core";

  // ── SOM ─────────────────────────────────────────────────────────
  py::class_<SOM>(m, "SOM")
      .def(py::init<int, int, int, float, float, float, unsigned>(),
           py::arg("rows"), py::arg("cols"), py::arg("n_dims"),
           py::arg("init_lr") = 0.15f, py::arg("lr_decay") = 0.9998f,
           py::arg("radius_decay") = 0.9999f, py::arg("seed") = 42u)
      .def(
          "find_bmu",
          [](const SOM &s, py::array_t<float, py::array::c_style> arr) {
            return s.find_bmu(to_vec(arr));
          },
          "Find best matching unit for input vector")
      .def(
          "activation_map",
          [](const SOM &s, py::array_t<float, py::array::c_style> arr) {
            return to_np(s.activation_map(to_vec(arr)));
          },
          "Return full activation map (n_neurons,) normalized to [0,1]")
      .def(
          "update",
          [](SOM &s, py::array_t<float, py::array::c_style> arr, int bmu,
             float reward_mod) { s.update(to_vec(arr), bmu, reward_mod); },
          py::arg("input"), py::arg("bmu"), py::arg("reward_mod") = 1.0f,
          "Update SOM weights toward input centered on bmu")
      .def(
          "neuron_weights",
          [](const SOM &s, int i) { return to_np(s.neuron_weights(i)); },
          "Get weight vector for neuron i")
      .def("grid_dist", &SOM::grid_dist,
           "2D grid distance between neurons i and j")
      .def("save", &SOM::save, "Save SOM to binary file")
      .def_static(
          "load", [](const std::string &p) { return SOM::load(p); },
          "Load SOM from binary file")
      .def_property_readonly("rows", [](const SOM &s) { return s.rows; })
      .def_property_readonly("cols", [](const SOM &s) { return s.cols; })
      .def_property_readonly("n_neurons",
                             [](const SOM &s) { return s.n_neurons; })
      .def_property_readonly("n_dims", [](const SOM &s) { return s.n_dims; })
      .def_property_readonly("step", &SOM::step)
      .def_property_readonly("lr", &SOM::lr)
      .def_property_readonly("radius", &SOM::radius);

  // ── Predictor ────────────────────────────────────────────────────
  py::class_<Predictor>(m, "Predictor")
      .def(py::init<int, int, float, unsigned>(), py::arg("input_dim"),
           py::arg("hidden_dim") = 256, py::arg("lr") = 0.001f,
           py::arg("seed") = 42u)
      .def(
          "step",
          [](Predictor &p, py::array_t<float, py::array::c_style> inp,
             py::object actual_obj) -> py::array_t<float> {
            auto x = to_vec(inp);
            if (actual_obj.is_none()) {
              return to_np(p.step(x, nullptr));
            } else {
              auto a = to_vec(
                  actual_obj.cast<py::array_t<float, py::array::c_style>>());
              return to_np(p.step(x, &a));
            }
          },
          py::arg("input"), py::arg("actual") = py::none(),
          "Predict next activation. If actual given: compute error + update.")
      .def(
          "train_sequence",
          [](Predictor &p, const std::vector<std::vector<float>> &inputs,
             const std::vector<float> &target,
             int n_bptt) { return p.train_sequence(inputs, target, n_bptt); },
          py::arg("inputs"), py::arg("target"), py::arg("n_bptt") = -1,
          "Train sequence with answer-only loss and N-step BPTT")
      .def("reset", &Predictor::reset,
           "Reset LSTM hidden/cell state (start of new sequence)")
      .def("set_offline", &Predictor::set_offline,
           "Set offline mode (imagination — no weight updates)")
      .def("save", &Predictor::save)
      .def_static("load",
                  [](const std::string &p) { return Predictor::load(p); })
      .def_property_readonly("last_error", &Predictor::last_error)
      .def_property_readonly("is_offline", &Predictor::is_offline)
      .def_property("lr", &Predictor::lr,
                    [](Predictor &p, float v) { p.set_lr(v); })
      .def_property_readonly("input_dim",
                             [](const Predictor &p) { return p.input_dim; })
      .def_property_readonly("hidden_dim",
                             [](const Predictor &p) { return p.hidden_dim; });

  // ── EpisodicMemory ───────────────────────────────────────────────
  py::class_<EpisodicMemory>(m, "EpisodicMemory")
      .def(py::init<int, int, float>(), py::arg("n_dims"),
           py::arg("max_episodes") = 2000, py::arg("surprise_threshold") = 0.3f)
      .def(
          "observe",
          [](EpisodicMemory &em, py::array_t<float, py::array::c_style> arr) {
            em.observe(to_vec(arr));
          },
          "Add activation frame to current building episode")
      .def("commit", &EpisodicMemory::commit, py::arg("prediction_error"), py::arg("payload") = std::vector<float>{},
           "Commit episode if error > threshold. Returns True if stored.")
      .def(
          "retrieve",
          [](const EpisodicMemory &em,
             py::array_t<float, py::array::c_style> arr) -> py::object {
            auto *ep = em.retrieve(to_vec(arr));
            if (!ep)
              return py::none();
            py::list frames;
            std::function<void(const EpisodeNode&)> flatten = [&](const EpisodeNode& node) {
                if (node.children.empty()) {
                    std::vector<float> f_float(node.summary_spike.size(), 0.0f);
                    for (size_t i = 0; i < node.summary_spike.size(); ++i) {
                        if (node.summary_spike[i]) f_float[i] = 1.0f;
                    }
                    frames.append(to_np(f_float));
                } else {
                    for (const auto& child : node.children) flatten(child);
                }
            };
            flatten(ep->root);
            return frames;
          },
          "Retrieve most similar episode as list of activation arrays")
      .def(
          "retrieve_topk",
          [](const EpisodicMemory &em,
             py::array_t<float, py::array::c_style> arr, int k) {
            auto r = em.retrieve_topk(to_vec(arr), k);
            py::list out;
            for (auto &[sim, idx] : r) {
              py::tuple t = py::make_tuple(sim, idx);
              out.append(t);
            }
            return out;
          },
          py::arg("query"), py::arg("k") = 3)
      .def("consolidate", &EpisodicMemory::consolidate,
           py::arg("similarity_threshold") = 0.85f,
           "Consolidate similar episodes into prototypes (call during rest)")
      .def("save", &EpisodicMemory::save)
      .def_static("load",
                  [](const std::string &p) { return EpisodicMemory::load(p); })
      .def_property_readonly("episode_count", &EpisodicMemory::episode_count)
      .def_property_readonly("prototype_count",
                             &EpisodicMemory::prototype_count)
      .def_property_readonly("step", &EpisodicMemory::step)
      .def_property(
          "surprise_threshold",
          [](const EpisodicMemory &em) { return em.surprise_threshold; },
          [](EpisodicMemory &em, float v) { em.surprise_threshold = v; },
          "Adaptive surprise threshold — write from Python to update C++ side");

  // ── WorkingMemory ────────────────────────────────────────────────
  py::class_<WorkingMemory>(m, "WorkingMemory")
      .def(py::init<int, int, float>(), py::arg("n_dims"),
           py::arg("capacity") = 7, py::arg("decay_rate") = 0.95f)
      .def(
          "gate",
          [](WorkingMemory &wm, py::array_t<float, py::array::c_style> arr,
             float salience) { return wm.gate(to_vec(arr), salience); },
          py::arg("activation"), py::arg("salience") = 0.f,
          "Insert activation into working memory. Returns True if inserted.")
      .def("tick", &WorkingMemory::tick, "Decay all slots one time step")
      .def(
          "context",
          [](const WorkingMemory &wm) { return to_np(wm.context()); },
          "Weighted mean of all active slots")
      .def(
          "most_active",
          [](const WorkingMemory &wm) { return to_np(wm.most_active()); },
          "Vector of most active slot")
      .def(
          "boost_salience",
          [](WorkingMemory &wm, py::array_t<float, py::array::c_style> arr,
             float amount) { wm.boost_salience(to_vec(arr), amount); },
          py::arg("vec"), py::arg("amount"))
      .def("clear", &WorkingMemory::clear)
      .def("activations",
           [](const WorkingMemory &wm) { return to_np(wm.activations()); })
      .def("save", &WorkingMemory::save)
      .def_static("load",
                  [](const std::string &p) { return WorkingMemory::load(p); })
      .def_property_readonly("size", &WorkingMemory::size)
      .def_property_readonly("empty", &WorkingMemory::empty)
      .def_property_readonly("n_dims",
                             [](const WorkingMemory &w) { return w.n_dims; })
      .def_property_readonly("capacity",
                             [](const WorkingMemory &w) { return w.get_base_capacity(); });

  // ── Language ─────────────────────────────────────────────────────
  py::class_<Language>(m, "Language")
      .def(py::init<int, float>(), py::arg("n_dims"), py::arg("lr") = 0.05f)
      .def(
          "register_word",
          [](Language &l, const std::string &w, py::object vec_obj) {
            if (vec_obj.is_none())
              l.register_word(w);
            else
              l.register_word(
                  w,
                  to_vec(
                      vec_obj.cast<py::array_t<float, py::array::c_style>>()));
          },
          py::arg("word"), py::arg("initial_vec") = py::none())
      .def("encode", [](Language &l,
                        const std::string &w) { return to_np(l.encode(w)); })
      .def(
          "decode",
          [](const Language &l, py::array_t<float, py::array::c_style> arr,
             int k) {
            auto r = l.decode(to_vec(arr), k);
            py::list out;
            for (auto &[w, s] : r)
              out.append(py::make_tuple(w, s));
            return out;
          },
          py::arg("concept_vec"), py::arg("k") = 5)
      .def("best_word",
           [](const Language &l, py::array_t<float, py::array::c_style> arr) {
             return l.best_word(to_vec(arr));
           })
      .def("hear",
           [](Language &l, const std::string &w,
              py::array_t<float, py::array::c_style> arr) {
             l.hear(w, to_vec(arr));
           })
      .def(
          "speak",
          [](const Language &l, py::list concept_seq, float min_sim) {
            std::vector<std::vector<float>> seqs;
            for (auto &item : concept_seq)
              seqs.push_back(
                  to_vec(item.cast<py::array_t<float, py::array::c_style>>()));
            return l.speak(seqs, min_sim);
          },
          py::arg("concept_seq"), py::arg("min_sim") = 0.f)
      .def("knows", &Language::knows)
      .def("familiarity", &Language::familiarity)
      .def("frequency", &Language::frequency)
      .def("vocab", &Language::vocab)
      .def("save", &Language::save)
      .def_static("load",
                  [](const std::string &p) { return Language::load(p); })
      .def_property_readonly("vocab_size", &Language::vocab_size)
      .def_property_readonly("n_dims",
                             [](const Language &l) { return l.n_dims; });

  // ── Simulation struct ────────────────────────────────────────────
  py::class_<Simulation>(m, "Simulation")
      .def_readonly("coherence", &Simulation::coherence)
      .def_readonly("valence", &Simulation::valence)
      .def_readonly("completed", &Simulation::completed)
      .def_property_readonly("frames", [](const Simulation &s) {
        py::list out;
        for (const auto &f : s.frames)
          out.append(to_np(f));
        return out;
      });

  // ── Imagination ──────────────────────────────────────────────────
  py::class_<Imagination>(m, "Imagination")

      .def(py::init([](Predictor *p, int max_steps) {
             return std::make_unique<Imagination>(p, max_steps);
           }),
           py::arg("predictor"), py::arg("max_steps") = 20,
           py::keep_alive<1, 2>()) // keep predictor alive
      .def(
          "simulate",
          [](Imagination &im, py::array_t<float, py::array::c_style> arr,
             int steps) { return im.simulate(to_vec(arr), steps); },
          py::arg("start_state"), py::arg("steps") = -1)
      .def(
          "dream",
          [](Imagination &im, int n_dreams, int steps_per_dream, py::list seeds,
             unsigned seed) {
            std::vector<std::vector<float>> sv;
            for (auto &item : seeds)
              sv.push_back(
                  to_vec(item.cast<py::array_t<float, py::array::c_style>>()));
            return im.dream(n_dreams, steps_per_dream, sv, seed);
          },
          py::arg("n_dreams"), py::arg("steps_per_dream") = 10,
          py::arg("seeds") = py::list(), py::arg("seed") = 42u)
      .def(
          "evaluate",
          [](const Imagination &im, const Simulation &s,
             py::array_t<float, py::array::c_style> arr, float threshold) {
            return im.evaluate(s, to_vec(arr), threshold);
          },
          py::arg("sim"), py::arg("goal_state"), py::arg("threshold") = 0.8f)
      .def(
          "extract_frames",
          [](const Imagination &im, const std::vector<Simulation> &sims,
             float min_coh) {
            auto frames = im.extract_frames(sims, min_coh);
            py::list out;
            for (const auto &f : frames)
              out.append(to_np(f));
            return out;
          },
          py::arg("sims"), py::arg("min_coherence") = 0.4f)
      .def_property_readonly("has_predictor", &Imagination::has_predictor)
      .def_property_readonly("n_dims",
                             [](const Imagination &i) { return i.n_dims; })
      .def_property_readonly("max_steps",
                             [](const Imagination &i) { return i.max_steps; });

  // ── EmotionState struct ──────────────────────────────────────────
  py::class_<EmotionState>(m, "EmotionState")
      .def_readonly("valence", &EmotionState::valence)
      .def_readonly("arousal", &EmotionState::arousal);

  // ── EmotionEvent struct ──────────────────────────────────────────
  py::class_<EmotionEvent>(m, "EmotionEvent")
      .def(py::init<float, float, float>(), py::arg("valence_delta"),
           py::arg("arousal_delta"), py::arg("intensity") = 1.0f)
      .def_readwrite("valence_delta", &EmotionEvent::valence_delta)
      .def_readwrite("arousal_delta", &EmotionEvent::arousal_delta)
      .def_readwrite("intensity", &EmotionEvent::intensity);

  // ── Emotion ──────────────────────────────────────────────────────
  py::class_<Emotion>(m, "Emotion")
      .def(py::init<float, float>(), py::arg("decay_rate") = 0.05f,
           py::arg("peak_decay") = 0.01f)
      .def("trigger", &Emotion::trigger)
      .def("from_prediction_error", &Emotion::from_prediction_error,
           py::arg("error"))
      .def("from_reward", &Emotion::from_reward, py::arg("reward"))
      .def("tick", &Emotion::tick)
      .def("reset", &Emotion::reset)
      .def("state", &Emotion::state)
      .def("save", &Emotion::save)
      .def_static("load", [](const std::string &p) { return Emotion::load(p); })
      .def_property_readonly("salience", &Emotion::salience)
      .def_property_readonly("lr_modulator", &Emotion::lr_modulator)
      .def_property_readonly("attention_modulator",
                             &Emotion::attention_modulator)
      .def_property_readonly("approach_mode", &Emotion::approach_mode)
      .def_property_readonly("avoidance_mode", &Emotion::avoidance_mode)
      .def_property_readonly("inertia", &Emotion::inertia)
      .def_property_readonly("peak_valence", &Emotion::peak_valence)
      .def_property_readonly("peak_arousal", &Emotion::peak_arousal)
      .def_property(
          "valence", [](const Emotion &e) { return e.valence; },
          [](Emotion &e, float v) { e.valence = v; })
      .def_property(
          "arousal", [](const Emotion &e) { return e.arousal; },
          [](Emotion &e, float v) { e.arousal = v; });

  // ── AttentionResult struct ───────────────────────────────────────
  py::class_<AttentionResult>(m, "AttentionResult")
      .def_readonly("passed", &AttentionResult::passed)
      .def_readonly("score", &AttentionResult::score)
      .def_readonly("threshold", &AttentionResult::threshold)
      .def_readonly("focus_bmu", &AttentionResult::focus_bmu);

  // ── Attention ────────────────────────────────────────────────────
  py::class_<Attention>(m, "Attention")
      .def(py::init<int, float, float>(), py::arg("n_neurons"),
           py::arg("decay_rate") = 0.1f, py::arg("base_threshold") = 0.3f)
      .def(
          "gate",
          [](Attention &a, py::array_t<float, py::array::c_style> arr,
             float novelty, float arousal_modulator) {
            return a.gate(to_vec(arr), novelty, arousal_modulator);
          },
          py::arg("activation_map"), py::arg("novelty"),
          py::arg("arousal_modulator") = 0.75f)
      .def("set_top_down",
           [](Attention &a, py::array_t<float, py::array::c_style> arr) {
             a.set_top_down(to_vec(arr));
           })
      .def("clear_top_down", &Attention::clear_top_down)
      .def("tick", &Attention::tick)
      .def("reset", &Attention::reset)
      .def("saliency_map",
           [](const Attention &a) { return to_np(a.saliency_map()); })
      .def("save", &Attention::save)
      .def_static("load",
                  [](const std::string &p) { return Attention::load(p); })
      .def_property_readonly("focus_neuron", &Attention::focus_neuron)
      .def_property_readonly("mean_saliency", &Attention::mean_saliency)
      .def_property_readonly("threshold", &Attention::threshold)
      .def_property_readonly("n_neurons",
                             [](const Attention &a) { return a.n_neurons; });

  // ── InternalState struct ─────────────────────────────────────────
  py::class_<InternalState>(m, "InternalState")
      .def(py::init<>())
      .def_readwrite("valence", &InternalState::valence)
      .def_readwrite("arousal", &InternalState::arousal)
      .def_readwrite("salience", &InternalState::salience)
      .def_readwrite("pred_error", &InternalState::pred_error)
      .def_readwrite("wm_load", &InternalState::wm_load)
      .def_readwrite("attention_focus", &InternalState::attention_focus)
      .def_readwrite("mean_saliency", &InternalState::mean_saliency)
      .def_readwrite("approach", &InternalState::approach)
      .def_readwrite("avoidance", &InternalState::avoidance)
      .def_readwrite("arousal_trend", &InternalState::arousal_trend);

  // ── SelfModel ────────────────────────────────────────────────────
  py::class_<SelfModel>(m, "SelfModel")
      .def(py::init<int, unsigned>(), py::arg("n_self_neurons") = 16,
           py::arg("seed") = 42u)
      .def("observe", &SelfModel::observe)
      .def("current_concept", &SelfModel::current_concept)
      .def("drift", &SelfModel::drift)
      .def("mean_recent_error", &SelfModel::mean_recent_error)
      .def("arousal_trend", &SelfModel::arousal_trend)
      .def("identity", [](const SelfModel &sm) { return to_np(sm.identity()); })
      .def("concept_weights",
           [](const SelfModel &sm, int i) {
             return to_np(sm.concept_weights(i));
           })
      .def("save", &SelfModel::save)
      .def_static("load",
                  [](const std::string &p) { return SelfModel::load(p); })
      .def_property_readonly("obs_count", &SelfModel::obs_count)
      .def_property_readonly("n_self_neurons", [](const SelfModel &sm) {
        return sm.n_self_neurons;
      });

  // ── SymbolOp enum ────────────────────────────────────────────────
  py::enum_<SymbolOp>(m, "SymbolOp")
      .value("NONE", SymbolOp::NONE)
      .value("ADD", SymbolOp::ADD)
      .value("SUBTRACT", SymbolOp::SUBTRACT)
      .value("MULTIPLY", SymbolOp::MULTIPLY)
      .value("COMPARE", SymbolOp::COMPARE)
      .value("SEQUENCE", SymbolOp::SEQUENCE)
      .value("NEGATE", SymbolOp::NEGATE)
      .export_values();

  // ── Symbolic ─────────────────────────────────────────────────────
  py::class_<Symbolic>(m, "Symbolic")
      .def(py::init<int>(), py::arg("n_dims"))
      .def(
          "bind",
          [](Symbolic &s, const std::string &sym, py::object vec_obj,
             SymbolOp op, const std::string &category) {
            if (vec_obj.is_none())
              s.bind(sym, {}, op, category);
            else
              s.bind(
                  sym,
                  to_vec(
                      vec_obj.cast<py::array_t<float, py::array::c_style>>()),
                  op, category);
          },
          py::arg("symbol"), py::arg("vec") = py::none(),
          py::arg("op") = SymbolOp::NONE, py::arg("category") = "")
      .def("lookup",
           [](Symbolic &s, const std::string &sym) {
             return to_np(s.lookup(sym));
           })
      .def("apply",
           [](Symbolic &s, const std::string &op_sym,
              py::array_t<float, py::array::c_style> a,
              py::array_t<float, py::array::c_style> b) {
             return to_np(s.apply(op_sym, to_vec(a), to_vec(b)));
           })
      .def("nearest_symbol",
           [](const Symbolic &s, py::array_t<float, py::array::c_style> arr) {
             return s.nearest_symbol(to_vec(arr));
           })
      .def("seed_math_symbols", &Symbolic::seed_math_symbols)
      .def("knows", &Symbolic::knows)
      .def("symbols", &Symbolic::symbols)
      .def("save", &Symbolic::save)
      .def_static("load",
                  [](const std::string &p) { return Symbolic::load(p); })
      .def_property_readonly("symbol_count", &Symbolic::symbol_count)
      .def_property_readonly("n_dims",
                             [](const Symbolic &s) { return s.n_dims; });

  // ── PerceiveResult ───────────────────────────────────────────────
  py::class_<PerceiveResult>(m, "PerceiveResult")
      .def_readonly("bmu", &PerceiveResult::bmu)
      .def_readonly("prediction_error", &PerceiveResult::prediction_error)
      .def_readonly("attention_passed", &PerceiveResult::attention_passed)
      .def_readonly("episodic_stored", &PerceiveResult::episodic_stored)
      .def_readonly("valence", &PerceiveResult::valence)
      .def_readonly("arousal", &PerceiveResult::arousal)
      .def_readonly("salience", &PerceiveResult::salience)
      .def_readonly("self_concept", &PerceiveResult::self_concept);

  // ── ThinkResult ──────────────────────────────────────────────────
  py::class_<ThinkResult>(m, "ThinkResult")
      .def_readonly("words", &ThinkResult::words)
      .def_readonly("coherence", &ThinkResult::coherence)
      .def_property_readonly("concepts", [](const ThinkResult &t) {
        py::list out;
        for (const auto &c : t.concepts)
          out.append(to_np(c));
        return out;
      });

  // ── Brain ────────────────────────────────────────────────────────
  py::class_<Brain>(m, "Brain")
      .def(py::init<int, int, int, int, int, int, int, unsigned>(),
           py::arg("som_rows"), py::arg("som_cols"), py::arg("n_dims"),
           py::arg("hidden_dim") = 256, py::arg("wm_capacity") = 7,
           py::arg("episodic_max") = 2000, py::arg("self_neurons") = 16,
           py::arg("seed") = 42u)
      .def("perceive",
           [](Brain &b, py::array_t<float, py::array::c_style> arr) {
             return b.perceive(to_vec(arr));
           })
      .def("hear", &Brain::hear)
      .def("reset_sequence", &Brain::reset_sequence)
      .def("load_components", &Brain::load_components,
           py::arg("predictor_path"), py::arg("language_path"),
           py::arg("som_path"), py::arg("episodic_path"),
           py::arg("emotion_path"), py::arg("self_path"),
           py::arg("symbolic_path"),
           // V3 & V4 optional
           py::arg("binding_path") = "", py::arg("bg_path") = "",
           py::arg("procedures_path") = "", py::arg("hpred_path") = "",
           py::arg("decoder_path") = "")
      .def("save_components", &Brain::save_components, py::arg("directory"))
      .def("load_bg", &Brain::load_bg, py::arg("path"))
      .def("save_bg", &Brain::save_bg, py::arg("path"))
      .def("think", &Brain::think, py::arg("steps") = 5)
      .def("reason", &Brain::reason, py::arg("goal_word"), py::arg("max_steps") = 10, py::arg("epsilon") = 0.0f)
      .def("start_reasoning", &Brain::start_reasoning)
      .def("force_reason_step", &Brain::force_reason_step, py::arg("op_idx"), py::arg("goal_word"))
      .def("reason_step", &Brain::reason_step, py::arg("goal_word"), py::arg("epsilon") = 0.0f)
      .def("direct_reason_step", &Brain::direct_reason_step, py::arg("goal_word"))
      .def("cognitive_step", &Brain::cognitive_step, py::arg("input_text"))
      .def("speak",
          [](Brain &b, std::vector<std::vector<float>> concepts, float min_sim) {
            return b.speak(concepts, min_sim);
          },
          py::arg("concepts"), py::arg("min_sim") = 0.f)
      .def("dream", &Brain::dream, py::arg("n_dreams") = 20,
           py::arg("steps_per_dream") = 15)
      .def("daydream", &Brain::daydream, "Run unsupervised predictive coding on imagination")
      .def(
          "imagine_goal",
          [](Brain &b, py::array_t<float, py::array::c_style> start,
             py::array_t<float, py::array::c_style> goal, int steps) {
            return b.imagine_goal(to_vec(start), to_vec(goal), steps);
          },
          py::arg("start"), py::arg("goal"), py::arg("steps") = 20)
      .def("commit_episode", [](Brain &b, float err, const std::vector<float>& payload) {
              return b.commit_episode(err, payload);
          }, py::arg("err"), py::arg("payload") = std::vector<float>{})
      .def("get_last_episode", [](Brain &b) -> std::vector<float> {
              // Retrieve the most recent episode from EpisodicMemory
              return b.episodic.get_last_episode();
          })
      .def("get_spoken_words", &Brain::get_spoken_words)
      .def("clear_spoken_words", &Brain::clear_spoken_words)
      .def(
          "symbol",
          [](Brain &b, const std::string &sym) { return to_np(b.symbol(sym)); })
      .def("learn_word",
           [](Brain &b, const std::string &word) {
               b.language.register_word(word);
               b.symbolic.bind(word);
           },
           "Manually register a new word into language and symbolic tables")
      .def("symbolic_op",
           [](Brain &b, const std::string &op,
              py::array_t<float, py::array::c_style> a,
              py::array_t<float, py::array::c_style> b_arr) {
             return to_np(b.symbolic_op(op, to_vec(a), to_vec(b_arr)));
           })
      // Direct component access
      .def_property_readonly(
          "som", [](Brain &b) -> SOM & { return b.som; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "predictor", [](Brain &b) -> Predictor & { return b.predictor; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "episodic", [](Brain &b) -> EpisodicMemory & { return b.episodic; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "working_mem",
          [](Brain &b) -> WorkingMemory & { return b.working_mem; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "language", [](Brain &b) -> Language & { return b.language; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "emotion", [](Brain &b) -> Emotion & { return b.emotion; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "attention", [](Brain &b) -> Attention & { return b.attention; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "self_model", [](Brain &b) -> SelfModel & { return b.self_model; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "symbolic_table", [](Brain &b) -> Symbolic & { return b.symbolic; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "scratchpad", [](Brain &b) -> Scratchpad & { return b.scratchpad; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "reasoning",
          [](Brain &b) -> ReasoningEngine & { return b.reasoning; },
          py::return_value_policy::reference_internal)
      // V3 methods
      .def(
          "bind_triple",
          [](Brain &b, py::array_t<float, py::array::c_style> s,
             py::array_t<float, py::array::c_style> r,
             py::array_t<float, py::array::c_style> o) {
            b.bind_triple(to_vec(s), to_vec(r), to_vec(o));
          },
          "Explicitly bind (subject, relation, object) triple")
      .def(
          "binding_query",
          [](Brain &b, py::array_t<float, py::array::c_style> subj,
             py::array_t<float, py::array::c_style> rel, bool want_object,
             float threshold) {
            auto result = b.binding_query(to_vec(subj), to_vec(rel), want_object, threshold);
            return py::make_tuple(to_np(result.first), result.second);
          },
          py::arg("subj"), py::arg("rel"), py::arg("want_object") = true,
          py::arg("threshold") = 0.5f)
      .def(
          "analogy_op",
          [](Brain &b, py::array_t<float, py::array::c_style> a,
             py::array_t<float, py::array::c_style> c) {
            return to_np(b.analogy_op(to_vec(a), to_vec(c)));
          },
          py::arg("subj"), py::arg("rel"),
          "Perform structure mapping analogy given (subj, rel)")
      .def("reinforce_bg", &Brain::reinforce_bg, py::arg("reward"),
           "Reinforce last BG op-chain using TD(lambda)")
      .def(
          "consolidate_procedure",
          [](Brain &b, std::vector<int> ops, const std::string &name) {
            b.consolidate_procedure(ops, name);
          },
          py::arg("ops"), py::arg("name") = "")
      // V3 component accessors
      .def_property_readonly(
          "binding", [](Brain &b) -> BindingMemory & { return b.binding; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "bg_controller",
          [](Brain &b) -> BasalGanglia & { return b.bg_controller; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "procedures",
          [](Brain &b) -> ProceduralMemory & { return b.procedures; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "h_predictor",
          [](Brain &b) -> HierarchicalPredictor & { return b.h_predictor; },
          py::return_value_policy::reference_internal)
      .def_property_readonly("step", &Brain::step)
      .def_property_readonly("initialized", &Brain::initialized);

  // ── PossibilityNode ──────────────────────────────────────────────
  py::class_<PossibilityNode>(m, "PossibilityNode")
      .def_readonly("id", &PossibilityNode::id)
      .def_readonly("parent_id", &PossibilityNode::parent_id)
      .def_readonly("h_cost", &PossibilityNode::h_cost)
      .def_readonly("children", &PossibilityNode::children)
      .def_property_readonly("state", [](const PossibilityNode &node) {
          return to_np(node.state);
      });

  // ── Scratchpad ───────────────────────────────────────────────────
  py::class_<Scratchpad>(m, "Scratchpad")
      .def(py::init<int>(), py::arg("n_dims"))
      .def(
          "write",
          [](Scratchpad &s, const std::string &name,
             py::array_t<float, py::array::c_style> arr,
             const std::string &tag) { s.write(name, to_vec(arr), tag); },
          py::arg("name"), py::arg("vec"), py::arg("tag") = "")
      .def("read", [](Scratchpad &s,
                      const std::string &name) { return to_np(s.read(name)); })
      .def("read_prev",
           [](Scratchpad &s, const std::string &name) {
             return to_np(s.read_prev(name));
           })
      .def("has", &Scratchpad::has)
      .def("similarity", &Scratchpad::similarity)
      .def("delta", &Scratchpad::delta)
      .def("push",
           [](Scratchpad &s, py::array_t<float, py::array::c_style> arr) {
             s.push(to_vec(arr));
           })
      .def("pop", [](Scratchpad &s) { return to_np(s.pop()); })
      .def("peek", [](Scratchpad &s) { return to_np(s.peek()); })
      .def("copy", &Scratchpad::copy)
      .def(
          "accumulate",
          [](Scratchpad &s, const std::string &name,
             py::array_t<float, py::array::c_style> arr,
             float alpha) { s.accumulate(name, to_vec(arr), alpha); },
          py::arg("name"), py::arg("vec"), py::arg("alpha") = 0.5f)
      .def("erase", &Scratchpad::erase)
      .def("clear", &Scratchpad::clear)
      .def("slot_names", &Scratchpad::slot_names)
      .def("tag", &Scratchpad::tag)
      .def("write_count", &Scratchpad::write_count)
      .def_property_readonly("slot_count", &Scratchpad::slot_count)
      .def_property_readonly("stack_size", &Scratchpad::stack_size)
      .def_property_readonly("n_dims",
                             [](const Scratchpad &s) { return s.n_dims; })
      // Tree Methods
      .def("start_tree", [](Scratchpad &s, py::array_t<float, py::array::c_style> arr) {
          return s.start_tree(to_vec(arr));
      })
      .def("branch", [](Scratchpad &s, py::array_t<float, py::array::c_style> arr, float h_cost) {
          return s.branch(to_vec(arr), h_cost);
      })
      .def("move_to", &Scratchpad::move_to)
      .def("move_to_best_child", &Scratchpad::move_to_best_child)
      .def("current_tree_state", [](const Scratchpad &s) {
          return to_np(s.current_tree_state());
      })
      .def("clear_tree", &Scratchpad::clear_tree)
      .def("get_tree", &Scratchpad::get_tree)
      .def("current_node", &Scratchpad::current_node);

  // ── ReasoningStep struct ─────────────────────────────────────────
  py::class_<ReasoningStep>(m, "ReasoningStep")
      .def(py::init<>())
      .def(py::init([](const std::string &inp, const std::string &op,
                       const std::string &arg, const std::string &out,
                       float cond_sim, const std::string &cond_slot) {
             return ReasoningStep{inp, op, arg, out, cond_sim, cond_slot};
           }),
           py::arg("input_slot"), py::arg("op_symbol"),
           py::arg("arg_slot") = "", py::arg("output_slot") = "result",
           py::arg("condition_sim") = 0.f, py::arg("condition_slot") = "")
      .def_readwrite("input_slot", &ReasoningStep::input_slot)
      .def_readwrite("op_symbol", &ReasoningStep::op_symbol)
      .def_readwrite("arg_slot", &ReasoningStep::arg_slot)
      .def_readwrite("output_slot", &ReasoningStep::output_slot)
      .def_readwrite("condition_sim", &ReasoningStep::condition_sim)
      .def_readwrite("condition_slot", &ReasoningStep::condition_slot);

  // ── ReasoningResult struct ───────────────────────────────────────
  py::class_<ReasoningResult>(m, "ReasoningResult")
      .def_readonly("converged", &ReasoningResult::converged)
      .def_readonly("steps_taken", &ReasoningResult::steps_taken)
      .def_readonly("final_delta", &ReasoningResult::final_delta)
      .def_readonly("trace", &ReasoningResult::trace)
      .def_property_readonly("outputs", [](const ReasoningResult &r) {
        py::list out;
        for (const auto &v : r.outputs)
          out.append(to_np(v));
        return out;
      });

  // ── ReasoningEngine ──────────────────────────────────────────────
  py::class_<ReasoningEngine>(m, "ReasoningEngine")
      .def(py::init([](Symbolic *sym, int n_dims, int max_steps, float conv_thr,
                       py::object pred_obj) {
             Predictor *pred =
                 pred_obj.is_none() ? nullptr : pred_obj.cast<Predictor *>();
             return std::make_unique<ReasoningEngine>(sym, n_dims, max_steps,
                                                      conv_thr, pred);
           }),
           py::arg("symbolic"), py::arg("n_dims"), py::arg("max_steps") = 20,
           py::arg("convergence_threshold") = 0.01f,
           py::arg("predictor") = py::none(), py::keep_alive<1, 2>(),
           py::keep_alive<1, 6>())
      .def("reason",
           [](ReasoningEngine &re, const std::vector<ReasoningStep> &steps,
              Scratchpad &pad) { return re.reason(steps, pad); })
      .def("solve_binary",
           [](ReasoningEngine &re, const std::string &op,
              py::array_t<float, py::array::c_style> a,
              py::array_t<float, py::array::c_style> b, Scratchpad &pad) {
             return to_np(re.solve_binary(op, to_vec(a), to_vec(b), pad));
           })
      .def(
          "infer",
          [](ReasoningEngine &re, py::list premises_list,
             const std::vector<ReasoningStep> &chain, Scratchpad &pad) {
            std::vector<std::pair<std::string, std::vector<float>>> premises;
            for (auto &item : premises_list) {
              auto tup = item.cast<py::tuple>();
              premises.push_back(
                  {tup[0].cast<std::string>(),
                   to_vec(
                       tup[1].cast<py::array_t<float, py::array::c_style>>())});
            }
            return to_np(re.infer(premises, chain, pad));
          })
      .def(
          "loop_until_convergence",
          [](ReasoningEngine &re, const ReasoningStep &step, Scratchpad &pad,
             int max_iters) {
            return re.loop_until_convergence(step, pad, max_iters);
          },
          py::arg("step"), py::arg("pad"), py::arg("max_iters") = -1)
      .def_property_readonly("has_symbolic", &ReasoningEngine::has_symbolic)
      .def_property_readonly("has_predictor", &ReasoningEngine::has_predictor)
      .def_property_readonly("n_dims",
                             [](const ReasoningEngine &r) { return r.n_dims; });

  // ── PredictiveCodingLayer (V3) ───────────────────────────────────
  py::class_<PredictiveCodingLayer>(m, "PredictiveCodingLayer")
      .def(py::init<int, float, float>(), py::arg("n_dims"),
           py::arg("threshold") = 0.05f, py::arg("lr") = 0.01f)
      .def("compute",
           [](PredictiveCodingLayer &pc,
              py::array_t<float, py::array::c_style> arr) {
             return to_np(pc.compute(to_vec(arr)));
           })
      .def("update", &PredictiveCodingLayer::update)
      .def_property_readonly("should_propagate",
                             &PredictiveCodingLayer::should_propagate)
      .def_property_readonly("error_norm", [](const PredictiveCodingLayer &pc) {
        return pc.error_norm;
      });

  // ── BindingMemory (V3) ───────────────────────────────────────────
  py::class_<BindingMemory>(m, "BindingMemory")
      .def(py::init<int, int>(), py::arg("n_dims"),
           py::arg("max_bindings") = 1000)
      .def("bind",
           [](BindingMemory &bm, py::array_t<float, py::array::c_style> s,
              py::array_t<float, py::array::c_style> r,
              py::array_t<float, py::array::c_style> o) {
             bm.bind(to_vec(s), to_vec(r), to_vec(o));
           })
      .def(
          "query",
          [](BindingMemory &bm, py::array_t<float, py::array::c_style> a,
             py::array_t<float, py::array::c_style> b, bool want_object, float threshold, int depth) {
            auto [vec, conf] = bm.query(to_vec(a), to_vec(b), want_object, threshold, depth);
            return py::make_tuple(to_np(vec), conf);
          },
          py::arg("a"), py::arg("b"), py::arg("want_object") = true, py::arg("threshold") = 0.3f, py::arg("depth") = 3)
      .def(
          "query_all",
          [](BindingMemory &bm, py::array_t<float, py::array::c_style> a, float threshold) {
              auto res = bm.query_all(to_vec(a), threshold);
              // convert std::vector<std::vector<float>> to list of numpy arrays
              py::list py_res;
              for (const auto& v : res) {
                  py_res.append(to_np(v));
              }
              return py_res;
          },
          py::arg("a"), py::arg("threshold") = 0.5f)
      .def("save", &BindingMemory::save)
      .def_static("load",
                  [](const std::string &p) { return BindingMemory::load(p); })
      .def_property_readonly("size", &BindingMemory::size);

  // ── GlobalWorkspace (V3) ─────────────────────────────────────────
  py::class_<GlobalWorkspace>(m, "GlobalWorkspace")
      .def(py::init<int>(), py::arg("n_dims"))
      .def("bid",
           [](GlobalWorkspace &gw, int module_id, float salience,
              py::array_t<float, py::array::c_style> rep) {
             gw.bid(module_id, salience, to_vec(rep));
           })
      .def("compete", &GlobalWorkspace::compete)
      .def("is_winner", &GlobalWorkspace::is_winner)
      .def("broadcast",
           [](const GlobalWorkspace &gw) { return to_np(gw.broadcast()); })
      .def_property_readonly("winner_id", &GlobalWorkspace::winner_id);

  // ── Op enum (V3) ─────────────────────────────────────────────────
  py::enum_<Op>(m, "Op")
      .value("READ", Op::READ)
      .value("WRITE", Op::WRITE)
      .value("MATH_SUB", Op::MATH_SUB)
      .value("MATH_DIV", Op::MATH_DIV)
      .value("COMPARE", Op::COMPARE)
      .value("BIND_QUERY", Op::BIND_QUERY)
      .value("RETRIEVE", Op::RETRIEVE)
      .value("ANALOGY", Op::ANALOGY)
      .value("HALT", Op::HALT)
      .export_values();

  // ── BasalGanglia (V3) ────────────────────────────────────────────
  py::class_<BasalGanglia>(m, "BasalGanglia")
      .def(py::init<int, float, unsigned>(), py::arg("n_dims"),
           py::arg("lr") = 0.001f, py::arg("seed") = 42u)
      .def(
          "select_op",
          [](BasalGanglia &bg, py::array_t<float, py::array::c_style> ctx,
             py::array_t<float, py::array::c_style> goal, bool greedy) {
            auto act = bg.select_op(to_vec(ctx), to_vec(goal), greedy);
            return (int)act.op;
          },
          py::arg("ctx"), py::arg("goal"), py::arg("greedy") = false)
      .def("reinforce", &BasalGanglia::reinforce, py::arg("final_reward"), py::arg("gamma") = 0.99f, py::arg("lambda_") = 0.95f)
      .def("clear_traces", &BasalGanglia::clear_traces)
      .def("save", &BasalGanglia::save)
      .def_static("load",
                  [](const std::string &p) { return BasalGanglia::load(p); });

  // ── ProceduralMemory (V3) ────────────────────────────────────────
  py::class_<ProceduralMemory>(m, "ProceduralMemory")
      .def("size", &ProceduralMemory::size)
      .def("save", &ProceduralMemory::save)
      .def("retrieve", [](ProceduralMemory& pm, py::array_t<float, py::array::c_style> context) {
          auto* p = pm.retrieve(to_vec(context));
          py::list py_ops;
          if (p) {
              for (auto op : p->steps) py_ops.append((int)op);
          }
          return py_ops;
      }, py::arg("context"));

  // ── HierarchicalPredictor (V3) ───────────────────────────────────
  py::class_<HierarchicalPredictor>(m, "HierarchicalPredictor")
      .def(py::init<int, int, int, unsigned>(), py::arg("n_dims"),
           py::arg("chunk_h") = 128, py::arg("episode_h") = 64,
           py::arg("seed") = 42u)
      .def("observe",
           [](HierarchicalPredictor &hp,
              py::array_t<float, py::array::c_style> act) {
             hp.observe(to_vec(act));
           })
      .def("reset", &HierarchicalPredictor::reset)
      .def("save", &HierarchicalPredictor::save)
      .def_static(
          "load",
          [](const std::string &p) { return HierarchicalPredictor::load(p); })
      .def_property_readonly("last_error_chunk",
                             &HierarchicalPredictor::last_error_chunk)
      .def_property_readonly("last_error_episode",
                             &HierarchicalPredictor::last_error_episode);
}
