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
    std::cout << "  BRAIN 3: GENERAL PURPOSE MCTS PLAYING CHESS\n";
    std::cout << "===========================================\n";
    std::cout << "The Brain does not have a chess engine.\n";
    std::cout << "It is using its general Monte Carlo Tree Search\n";
    std::cout << "to explore the Chess Environment.\n\n";

    ChessProblem env;
    ChessState state = env.initial();

    // MCTS config for Brain
    MonteCarloConfig cfg;
    cfg.iterations = 2000;
    cfg.rollout_depth = 15;
    cfg.exploration = 2.0; // Encourage exploring tactics
    cfg.goal_reward = 10000.0; // Win!

    while (!env.is_goal(state)) {
        print_board(state);
        
        auto legal_moves = env.moves(state);
        if (legal_moves.empty()) {
            std::cout << "Stalemate or no moves left.\n";
            break;
        }

        if (state.white_turn) {
            std::cout << "Your move (e.g. e2e4): ";
            std::string user_move;
            std::cin >> user_move;
            
            bool found = false;
            for (const auto& m : legal_moves) {
                if (std::get<0>(m) == user_move) {
                    state = std::get<1>(m);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "Invalid move. Try again.\n";
                continue;
            }
        } else {
            std::cout << "Brain is thinking (MCTS scanning thousands of futures)...\n";
            
            // To apply MCTS from current state, we make a temp problem
            // that starts from this state.
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
                std::cout << "Brain chooses: " << chosen_move << "\n";
                state = result.path[0].second;
            } else {
                // Fallback to random if MCTS totally fails to find anything
                std::cout << "Brain fallback to random move: " << std::get<0>(legal_moves[0]) << "\n";
                state = std::get<1>(legal_moves[0]);
            }
        }
    }

    std::cout << "\nGAME OVER.\n";
    print_board(state);
    if (!state.white_turn) std::cout << "You win! You captured the Brain's King.\n";
    else std::cout << "The Brain wins! It captured your King.\n";

    return 0;
}
