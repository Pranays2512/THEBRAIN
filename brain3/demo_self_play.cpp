#include <iostream>
#include <string>
#include "crisp/engines/reasoning/chess_domain.hpp"
#include "crisp/engines/reasoning/monte_carlo_tree.hpp"

using namespace brain2::reasoning;

void print_board(const ChessState& s) {
    std::cout << "\n  a b c d e f g h\n";
    for (int r = 0; r < 8; ++r) {
        std::cout << 8 - r << " ";
        for (int c = 0; c < 8; ++c) {
            std::cout << s.board[r * 8 + c] << " ";
        }
        std::cout << 8 - r << "\n";
    }
    std::cout << "  a b c d e f g h\n\n";
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "  BRAIN 3: MCTS vs MCTS SELF-PLAY (MAX CAPABILITY)\n";
    std::cout << "===========================================\n";

    ChessProblem env;
    ChessState state = env.initial();

    // High capability config
    MonteCarloConfig cfg;
    cfg.iterations = 5000;
    cfg.rollout_depth = 20;
    cfg.exploration = 1.5;
    cfg.goal_reward = 10000.0;

    int turns = 0;
    const int MAX_TURNS = 150;

    print_board(state);

    while (!env.is_goal(state) && turns < MAX_TURNS) {
        auto legal_moves = env.moves(state);
        if (legal_moves.empty()) {
            std::cout << "Stalemate or no moves left.\n";
            break;
        }

        std::cout << "Turn " << (turns + 1) << " | " << (state.white_turn ? "White" : "Black") << " is thinking...\n";
        
        class CurrentChess : public ChessProblem {
            ChessState start;
        public:
            CurrentChess(ChessState s) : start(s) {}
            ChessState initial() const override { return start; }
        };
        
        CurrentChess current_prob(state);
        auto result = solve_mcts(current_prob, cfg);
        
        if (!result.path.empty()) {
            std::string chosen_move = result.path[0].first;
            std::cout << "=> Brain chooses: " << chosen_move << " (simulations=" << result.simulations << ", reward=" << result.reward << ")\n";
            state = result.path[0].second;
        } else {
            std::cout << "=> Brain fallback to random move: " << std::get<0>(legal_moves[0]) << "\n";
            state = std::get<1>(legal_moves[0]);
        }

        print_board(state);
        turns++;
    }

    std::cout << "\n=== GAME OVER ===\n";
    if (env.is_goal(state)) {
        if (!state.white_turn) std::cout << "White wins! Captured Black's King in " << turns << " turns.\n";
        else std::cout << "Black wins! Captured White's King in " << turns << " turns.\n";
    } else if (turns >= MAX_TURNS) {
        std::cout << "Draw! Reached maximum turns (" << MAX_TURNS << ") without a capture.\n";
    } else {
        std::cout << "Draw! No moves left (Stalemate).\n";
    }

    return 0;
}
