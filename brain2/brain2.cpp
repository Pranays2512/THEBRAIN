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
}
