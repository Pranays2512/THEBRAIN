#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace py = pybind11;
namespace brain2 {

struct SearchNode {
    double f;
    double g;
    long long tie;
    py::object state;
    py::list path;

    bool operator<(const SearchNode& other) const {
        if (f != other.f) return f > other.f;
        return tie > other.tie;
    }
};

struct PyObjectHash {
    size_t operator()(const py::object& obj) const {
        return py::hash(obj);
    }
};

struct PyObjectEqual {
    bool operator()(const py::object& lhs, const py::object& rhs) const {
        return lhs.equal(rhs);
    }
};

inline py::tuple solve_astar(py::object problem, int max_nodes) {
    if (max_nodes < 1) {
        throw std::invalid_argument("max_nodes must be a positive integer");
    }

    auto initial_func = problem.attr("initial");
    auto moves_func = problem.attr("moves");
    auto is_goal_func = problem.attr("is_goal");
    auto heuristic_func = problem.attr("heuristic");
    auto key_func = problem.attr("key");

    py::object start = initial_func();
    long long tie_counter = 0;
    int expanded = 0;

    std::priority_queue<SearchNode> frontier;
    py::list empty_path;
    double start_h = py::cast<double>(heuristic_func(start));
    frontier.push({start_h, 0.0, tie_counter++, start, empty_path});

    std::unordered_map<py::object, double, PyObjectHash, PyObjectEqual> best;
    py::object start_key = key_func(start);
    best[start_key] = 0.0;

    while (!frontier.empty() && expanded < max_nodes) {
        SearchNode curr = frontier.top();
        frontier.pop();

        if (py::cast<bool>(is_goal_func(curr.state))) {
            return py::make_tuple(curr.path, curr.g, expanded);
        }

        expanded++;

        py::object moves = moves_func(curr.state);
        for (auto item : moves) {
            py::tuple move = item.cast<py::tuple>();
            py::object label = move[0];
            py::object nxt = move[1];
            double cost = move[2].cast<double>();

            if (cost < 0) {
                throw std::invalid_argument("negative step cost breaks A* optimality");
            }

            double ng = curr.g + cost;
            py::object k = key_func(nxt);

            auto it = best.find(k);
            if (it != best.end() && it->second <= ng) {
                continue; // Prune
            }

            best[k] = ng;
            double nxt_h = py::cast<double>(heuristic_func(nxt));
            
            py::list next_path;
            for (auto p : curr.path) {
                next_path.append(p);
            }
            next_path.append(py::make_tuple(label, nxt));

            frontier.push({ng + nxt_h, ng, tie_counter++, nxt, next_path});
        }
    }

    return py::make_tuple(py::none(), py::none(), expanded);
}

} // namespace brain2
