/*
 * brain2.cpp — pybind11 bindings for Brain v2
 *
 * Exposes all C++ brain components to Python.
 * Python is used ONLY for: tests, training loops, and user interface.
 * All computation stays in C++.
 *
 * Build: see CMakeLists.txt
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "core/som.hpp"
#include "core/predictor.hpp"
#include "core/episodic.hpp"
#include "core/working_mem.hpp"
#include "core/language.hpp"
#include "core/imagination.hpp"
#include "core/emotion.hpp"
#include "core/attention.hpp"

namespace py = pybind11;
using namespace brain2;

// Helper: numpy array → std::vector<float>
static std::vector<float> to_vec(py::array_t<float, py::array::c_style> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::invalid_argument("Expected 1D float array");
    const float* ptr = static_cast<const float*>(buf.ptr);
    return std::vector<float>(ptr, ptr + buf.shape[0]);
}

// Helper: std::vector<float> → numpy array
static py::array_t<float> to_np(const std::vector<float>& v) {
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
             py::arg("init_lr")      = 0.15f,
             py::arg("lr_decay")     = 0.9998f,
             py::arg("radius_decay") = 0.9999f,
             py::arg("seed")         = 42u)
        .def("find_bmu",
             [](const SOM& s, py::array_t<float, py::array::c_style> arr) {
                 return s.find_bmu(to_vec(arr));
             }, "Find best matching unit for input vector")
        .def("activation_map",
             [](const SOM& s, py::array_t<float, py::array::c_style> arr) {
                 return to_np(s.activation_map(to_vec(arr)));
             }, "Return full activation map (n_neurons,) normalized to [0,1]")
        .def("update",
             [](SOM& s, py::array_t<float, py::array::c_style> arr,
                int bmu, float reward_mod) {
                 s.update(to_vec(arr), bmu, reward_mod);
             }, py::arg("input"), py::arg("bmu"), py::arg("reward_mod") = 1.0f,
             "Update SOM weights toward input centered on bmu")
        .def("neuron_weights",
             [](const SOM& s, int i) { return to_np(s.neuron_weights(i)); },
             "Get weight vector for neuron i")
        .def("grid_dist", &SOM::grid_dist,
             "2D grid distance between neurons i and j")
        .def("save", &SOM::save, "Save SOM to binary file")
        .def_static("load",
             [](const std::string& p) { return SOM::load(p); },
             "Load SOM from binary file")
        .def_property_readonly("rows",      [](const SOM& s){ return s.rows; })
        .def_property_readonly("cols",      [](const SOM& s){ return s.cols; })
        .def_property_readonly("n_neurons", [](const SOM& s){ return s.n_neurons; })
        .def_property_readonly("n_dims",    [](const SOM& s){ return s.n_dims; })
        .def_property_readonly("step",      &SOM::step)
        .def_property_readonly("lr",        &SOM::lr)
        .def_property_readonly("radius",    &SOM::radius);

    // ── Predictor ────────────────────────────────────────────────────
    py::class_<Predictor>(m, "Predictor")
        .def(py::init<int, int, float, unsigned>(),
             py::arg("input_dim"),
             py::arg("hidden_dim") = 256,
             py::arg("lr")         = 0.001f,
             py::arg("seed")       = 42u)
        .def("step",
             [](Predictor& p,
                py::array_t<float, py::array::c_style> inp,
                py::object actual_obj) -> py::array_t<float> {
                 auto x = to_vec(inp);
                 if (actual_obj.is_none()) {
                     return to_np(p.step(x, nullptr));
                 } else {
                     auto a = to_vec(actual_obj.cast<py::array_t<float,
                                     py::array::c_style>>());
                     return to_np(p.step(x, &a));
                 }
             }, py::arg("input"), py::arg("actual") = py::none(),
             "Predict next activation. If actual given: compute error + update.")
        .def("reset",      &Predictor::reset,
             "Reset LSTM hidden/cell state (start of new sequence)")
        .def("set_offline", &Predictor::set_offline,
             "Set offline mode (imagination — no weight updates)")
        .def("save",       &Predictor::save)
        .def_static("load",
             [](const std::string& p) { return Predictor::load(p); })
        .def_property_readonly("last_error", &Predictor::last_error)
        .def_property_readonly("is_offline", &Predictor::is_offline)
        .def_property("lr",
             &Predictor::lr,
             [](Predictor& p, float v) { p.set_lr(v); })
        .def_property_readonly("input_dim",  [](const Predictor& p){ return p.input_dim; })
        .def_property_readonly("hidden_dim", [](const Predictor& p){ return p.hidden_dim; });

    // ── EpisodicMemory ───────────────────────────────────────────────
    py::class_<EpisodicMemory>(m, "EpisodicMemory")
        .def(py::init<int, int, float>(),
             py::arg("n_dims"),
             py::arg("max_episodes")       = 2000,
             py::arg("surprise_threshold") = 0.3f)
        .def("observe",
             [](EpisodicMemory& em, py::array_t<float, py::array::c_style> arr) {
                 em.observe(to_vec(arr));
             }, "Add activation frame to current building episode")
        .def("commit",  &EpisodicMemory::commit,
             py::arg("prediction_error"),
             "Commit episode if error > threshold. Returns True if stored.")
        .def("retrieve",
             [](const EpisodicMemory& em,
                py::array_t<float, py::array::c_style> arr) -> py::object {
                 auto* ep = em.retrieve(to_vec(arr));
                 if (!ep) return py::none();
                 py::list frames;
                 for (const auto& f : ep->frames) frames.append(to_np(f));
                 return frames;
             }, "Retrieve most similar episode as list of activation arrays")
        .def("retrieve_topk",
             [](const EpisodicMemory& em,
                py::array_t<float, py::array::c_style> arr, int k) {
                 auto r = em.retrieve_topk(to_vec(arr), k);
                 py::list out;
                 for (auto& [sim, idx] : r) {
                     py::tuple t = py::make_tuple(sim, idx);
                     out.append(t);
                 }
                 return out;
             }, py::arg("query"), py::arg("k") = 3)
        .def("consolidate", &EpisodicMemory::consolidate,
             py::arg("similarity_threshold") = 0.85f,
             "Consolidate similar episodes into prototypes (call during rest)")
        .def("save", &EpisodicMemory::save)
        .def_static("load",
             [](const std::string& p) { return EpisodicMemory::load(p); })
        .def_property_readonly("episode_count",   &EpisodicMemory::episode_count)
        .def_property_readonly("prototype_count", &EpisodicMemory::prototype_count)
        .def_property_readonly("step",            &EpisodicMemory::step);

    // ── WorkingMemory ────────────────────────────────────────────────
    py::class_<WorkingMemory>(m, "WorkingMemory")
        .def(py::init<int, int, float>(),
             py::arg("n_dims"),
             py::arg("capacity")   = 7,
             py::arg("decay_rate") = 0.95f)
        .def("gate",
             [](WorkingMemory& wm, py::array_t<float, py::array::c_style> arr,
                float salience) {
                 return wm.gate(to_vec(arr), salience);
             }, py::arg("activation"), py::arg("salience") = 0.f,
             "Insert activation into working memory. Returns True if inserted.")
        .def("tick",     &WorkingMemory::tick,
             "Decay all slots one time step")
        .def("context",
             [](const WorkingMemory& wm) { return to_np(wm.context()); },
             "Weighted mean of all active slots")
        .def("most_active",
             [](const WorkingMemory& wm) { return to_np(wm.most_active()); },
             "Vector of most active slot")
        .def("boost_salience",
             [](WorkingMemory& wm, py::array_t<float, py::array::c_style> arr,
                float amount) { wm.boost_salience(to_vec(arr), amount); },
             py::arg("vec"), py::arg("amount"))
        .def("clear",       &WorkingMemory::clear)
        .def("activations",
             [](const WorkingMemory& wm) { return to_np(wm.activations()); })
        .def("save",        &WorkingMemory::save)
        .def_static("load",
             [](const std::string& p) { return WorkingMemory::load(p); })
        .def_property_readonly("size",  &WorkingMemory::size)
        .def_property_readonly("empty", &WorkingMemory::empty)
        .def_property_readonly("n_dims",    [](const WorkingMemory& w){ return w.n_dims; })
        .def_property_readonly("capacity",  [](const WorkingMemory& w){ return w.capacity; });

    // ── Language ─────────────────────────────────────────────────────
    py::class_<Language>(m, "Language")
        .def(py::init<int, float>(),
             py::arg("n_dims"),
             py::arg("lr") = 0.05f)
        .def("register_word",
             [](Language& l, const std::string& w,
                py::object vec_obj) {
                 if (vec_obj.is_none())
                     l.register_word(w);
                 else
                     l.register_word(w,
                         to_vec(vec_obj.cast<py::array_t<float,
                                py::array::c_style>>()));
             }, py::arg("word"), py::arg("initial_vec") = py::none())
        .def("encode",
             [](const Language& l, const std::string& w) {
                 return to_np(l.encode(w));
             })
        .def("decode",
             [](const Language& l,
                py::array_t<float, py::array::c_style> arr, int k) {
                 auto r = l.decode(to_vec(arr), k);
                 py::list out;
                 for (auto& [w, s] : r)
                     out.append(py::make_tuple(w, s));
                 return out;
             }, py::arg("concept_vec"), py::arg("k") = 5)
        .def("best_word",
             [](const Language& l,
                py::array_t<float, py::array::c_style> arr) {
                 return l.best_word(to_vec(arr));
             })
        .def("hear",
             [](Language& l, const std::string& w,
                py::array_t<float, py::array::c_style> arr) {
                 l.hear(w, to_vec(arr));
             })
        .def("speak",
             [](const Language& l,
                py::list concept_seq, float min_sim) {
                 std::vector<std::vector<float>> seqs;
                 for (auto& item : concept_seq)
                     seqs.push_back(to_vec(item.cast<
                         py::array_t<float, py::array::c_style>>()));
                 return l.speak(seqs, min_sim);
             }, py::arg("concept_seq"), py::arg("min_sim") = 0.f)
        .def("knows",       &Language::knows)
        .def("familiarity", &Language::familiarity)
        .def("frequency",   &Language::frequency)
        .def("vocab",       &Language::vocab)
        .def("save",        &Language::save)
        .def_static("load",
             [](const std::string& p) { return Language::load(p); })
        .def_property_readonly("vocab_size", &Language::vocab_size)
        .def_property_readonly("n_dims",
             [](const Language& l){ return l.n_dims; });

    // ── Simulation struct ────────────────────────────────────────────
    py::class_<Simulation>(m, "Simulation")
        .def_readonly("coherence",  &Simulation::coherence)
        .def_readonly("valence",    &Simulation::valence)
        .def_readonly("completed",  &Simulation::completed)
        .def_property_readonly("frames",
             [](const Simulation& s) {
                 py::list out;
                 for (const auto& f : s.frames) out.append(to_np(f));
                 return out;
             });

    // ── Imagination ──────────────────────────────────────────────────
    py::class_<Imagination>(m, "Imagination")
        .def(py::init([](Predictor* p, int max_steps) {
                 return std::make_unique<Imagination>(p, max_steps);
             }),
             py::arg("predictor"), py::arg("max_steps") = 20,
             py::keep_alive<1, 2>())  // keep predictor alive
        .def("simulate",
             [](Imagination& im,
                py::array_t<float, py::array::c_style> arr,
                int steps) {
                 return im.simulate(to_vec(arr), steps);
             }, py::arg("start_state"), py::arg("steps") = -1)
        .def("dream",
             [](Imagination& im, int n_dreams, int steps_per_dream,
                py::list seeds, unsigned seed) {
                 std::vector<std::vector<float>> sv;
                 for (auto& item : seeds)
                     sv.push_back(to_vec(item.cast<
                         py::array_t<float, py::array::c_style>>()));
                 return im.dream(n_dreams, steps_per_dream, sv, seed);
             }, py::arg("n_dreams"), py::arg("steps_per_dream") = 10,
                py::arg("seeds") = py::list(), py::arg("seed") = 42u)
        .def("evaluate",
             [](const Imagination& im, const Simulation& s,
                py::array_t<float, py::array::c_style> arr,
                float threshold) {
                 return im.evaluate(s, to_vec(arr), threshold);
             }, py::arg("sim"), py::arg("goal_state"),
                py::arg("threshold") = 0.8f)
        .def("extract_frames",
             [](const Imagination& im,
                const std::vector<Simulation>& sims, float min_coh) {
                 auto frames = im.extract_frames(sims, min_coh);
                 py::list out;
                 for (const auto& f : frames) out.append(to_np(f));
                 return out;
             }, py::arg("sims"), py::arg("min_coherence") = 0.4f)
        .def_property_readonly("has_predictor", &Imagination::has_predictor)
        .def_property_readonly("n_dims",
             [](const Imagination& i){ return i.n_dims; })
        .def_property_readonly("max_steps",
             [](const Imagination& i){ return i.max_steps; });

    // ── EmotionState struct ──────────────────────────────────────────
    py::class_<EmotionState>(m, "EmotionState")
        .def_readonly("valence", &EmotionState::valence)
        .def_readonly("arousal", &EmotionState::arousal);

    // ── EmotionEvent struct ──────────────────────────────────────────
    py::class_<EmotionEvent>(m, "EmotionEvent")
        .def(py::init<float, float, float>(),
             py::arg("valence_delta"), py::arg("arousal_delta"),
             py::arg("intensity") = 1.0f)
        .def_readwrite("valence_delta", &EmotionEvent::valence_delta)
        .def_readwrite("arousal_delta", &EmotionEvent::arousal_delta)
        .def_readwrite("intensity",     &EmotionEvent::intensity);

    // ── Emotion ──────────────────────────────────────────────────────
    py::class_<Emotion>(m, "Emotion")
        .def(py::init<float, float>(),
             py::arg("decay_rate")  = 0.05f,
             py::arg("peak_decay")  = 0.01f)
        .def("trigger",               &Emotion::trigger)
        .def("from_prediction_error", &Emotion::from_prediction_error,
             py::arg("error"))
        .def("from_reward",           &Emotion::from_reward,
             py::arg("reward"))
        .def("tick",                  &Emotion::tick)
        .def("reset",                 &Emotion::reset)
        .def("state",                 &Emotion::state)
        .def("save",                  &Emotion::save)
        .def_static("load",
             [](const std::string& p) { return Emotion::load(p); })
        .def_property_readonly("salience",           &Emotion::salience)
        .def_property_readonly("lr_modulator",       &Emotion::lr_modulator)
        .def_property_readonly("attention_modulator",&Emotion::attention_modulator)
        .def_property_readonly("approach_mode",      &Emotion::approach_mode)
        .def_property_readonly("avoidance_mode",     &Emotion::avoidance_mode)
        .def_property_readonly("inertia",            &Emotion::inertia)
        .def_property_readonly("peak_valence",       &Emotion::peak_valence)
        .def_property_readonly("peak_arousal",       &Emotion::peak_arousal)
        .def_property("valence",
             [](const Emotion& e){ return e.valence; },
             [](Emotion& e, float v){ e.valence = v; })
        .def_property("arousal",
             [](const Emotion& e){ return e.arousal; },
             [](Emotion& e, float v){ e.arousal = v; });

    // ── AttentionResult struct ───────────────────────────────────────
    py::class_<AttentionResult>(m, "AttentionResult")
        .def_readonly("passed",    &AttentionResult::passed)
        .def_readonly("score",     &AttentionResult::score)
        .def_readonly("threshold", &AttentionResult::threshold)
        .def_readonly("focus_bmu", &AttentionResult::focus_bmu);

    // ── Attention ────────────────────────────────────────────────────
    py::class_<Attention>(m, "Attention")
        .def(py::init<int, float, float>(),
             py::arg("n_neurons"),
             py::arg("decay_rate")      = 0.1f,
             py::arg("base_threshold")  = 0.3f)
        .def("gate",
             [](Attention& a,
                py::array_t<float, py::array::c_style> arr,
                float novelty, float arousal_modulator) {
                 return a.gate(to_vec(arr), novelty, arousal_modulator);
             }, py::arg("activation_map"), py::arg("novelty"),
                py::arg("arousal_modulator") = 0.75f)
        .def("set_top_down",
             [](Attention& a, py::array_t<float, py::array::c_style> arr) {
                 a.set_top_down(to_vec(arr));
             })
        .def("clear_top_down", &Attention::clear_top_down)
        .def("tick",           &Attention::tick)
        .def("reset",          &Attention::reset)
        .def("saliency_map",
             [](const Attention& a) { return to_np(a.saliency_map()); })
        .def("save",           &Attention::save)
        .def_static("load",
             [](const std::string& p) { return Attention::load(p); })
        .def_property_readonly("focus_neuron",  &Attention::focus_neuron)
        .def_property_readonly("mean_saliency", &Attention::mean_saliency)
        .def_property_readonly("threshold",     &Attention::threshold)
        .def_property_readonly("n_neurons",
             [](const Attention& a){ return a.n_neurons; });
}
