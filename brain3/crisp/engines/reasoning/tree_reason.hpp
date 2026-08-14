#pragma once
#include <vector>
#include <string>
#include <map>
#include <queue>
#include <functional>
#include <memory>
#include <tuple>

namespace brain2 {
namespace reasoning {

// Generalized A* search over arbitrary states
template<typename State, typename Hash = std::hash<State>>
class SearchProblem {
public:
    virtual ~SearchProblem() = default;
    virtual State initial() const = 0;
    virtual bool is_goal(const State& s) const = 0;
    virtual double heuristic(const State& s) const { return 0.0; }
    virtual double novelty(const State& s) const { return 0.0; }
    
    // returns vector of {label, next_state, step_cost}
    virtual std::vector<std::tuple<std::string, State, double>> moves(const State& s) const = 0;
};

template<typename State, typename Hash = std::hash<State>>
struct SearchResult {
    bool solved;
    std::vector<std::pair<std::string, State>> path;
    double cost;
    int nodes_expanded;
};

template<typename State, typename Hash = std::hash<State>>
SearchResult<State, Hash> solve_astar(const SearchProblem<State, Hash>& problem, int max_nodes = 500000) {
    struct Node {
        double f, g;
        int tie;
        State state;
        std::vector<std::pair<std::string, State>> path;
        
        bool operator>(const Node& other) const {
            if (f != other.f) return f > other.f;
            return tie > other.tie;
        }
    };
    
    State start = problem.initial();
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> frontier;
    std::map<State, double> best;
    
    int tie_counter = 0;
    frontier.push({problem.heuristic(start), 0.0, tie_counter++, start, {}});
    best[start] = 0.0;
    
    int expanded = 0;
    while (!frontier.empty() && expanded < max_nodes) {
        Node curr = frontier.top();
        frontier.pop();
        
        if (problem.is_goal(curr.state)) {
            return {true, curr.path, curr.g, expanded};
        }
        
        expanded++;
        for (const auto& [label, nxt, cost] : problem.moves(curr.state)) {
            double ng = curr.g + cost;
            if (best.count(nxt) && best[nxt] <= ng) continue;
            
            best[nxt] = ng;
            auto next_path = curr.path;
            next_path.push_back({label, nxt});
            
            frontier.push({ng + problem.heuristic(nxt), ng, tie_counter++, nxt, next_path});
        }
    }
    
    return {false, {}, 0.0, expanded};
}

} // namespace reasoning
} // namespace brain2
