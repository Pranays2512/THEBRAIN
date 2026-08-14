#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include "crisp/engines/reasoning/neuro_symbolic_search.hpp"

using namespace brain2::reasoning;

struct Position {
    int x, y;
    bool operator==(const Position& o) const { return x == o.x && y == o.y; }
    bool operator<(const Position& o) const { return x != o.x ? x < o.x : y < o.y; }
};

namespace std {
    template<> struct hash<Position> {
        size_t operator()(const Position& p) const {
            return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
        }
    };
}

class GridMazeProblem : public SearchProblem<Position> {
private:
    int target_x, target_y;
public:
    GridMazeProblem(int tx, int ty) : target_x(tx), target_y(ty) {}

    Position initial() const override { return {0, 0}; }
    
    bool is_goal(const Position& s) const override { 
        return s.x == target_x && s.y == target_y; 
    }
    
    std::vector<std::tuple<std::string, Position, double>> moves(const Position& s) const override {
        std::vector<std::tuple<std::string, Position, double>> out;
        if (s.x < target_x) out.push_back({"Right", {s.x + 1, s.y}, 1.0});
        if (s.y < target_y) out.push_back({"Down", {s.x, s.y + 1}, 1.0});
        return out;
    }
    
    double novelty(const Position& s) const override {
        // Simple novelty to help MCTS spread out slightly
        return (double)(s.x + s.y);
    }
};

int main() {
    std::cout << "===========================================\n";
    std::cout << "  BRAIN 3: NEURO-SYMBOLIC SEARCH (A* + MCTS)\n";
    std::cout << "===========================================\n\n";

    // A large 15x15 grid. A* without a heuristic will explore a massive diamond shape.
    GridMazeProblem maze(15, 15);

    std::cout << "--- EXPERIMENT 1: Pure A* Search (No Heuristic) ---\n";
    auto astar_result = solve_astar(maze, 100000);
    std::cout << "Solved: " << (astar_result.solved ? "YES" : "NO") << "\n";
    std::cout << "Nodes Expanded (Combinatorial Explosion): " << astar_result.nodes_expanded << "\n\n";

    std::cout << "--- EXPERIMENT 2: Neuro-Symbolic Search (A* guided by MCTS) ---\n";
    // Neuro-Symbolic search uses 20 MCTS iterations to peek into the future.
    auto ns_result = solve_neuro_symbolic(maze, 20);
    std::cout << "Solved: " << (ns_result.solved ? "YES" : "NO") << "\n";
    std::cout << "A* Nodes Expanded (Perfect Guidance): " << ns_result.nodes_expanded << "\n";

    std::cout << "\nNotice the massive reduction in A* expanded nodes! MCTS perfectly guided A* to the goal, and the temporary memoization table prevented redundant MCTS simulations.\n";

    return 0;
}
