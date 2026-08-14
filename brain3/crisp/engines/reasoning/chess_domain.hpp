#pragma once
#include "crisp/engines/reasoning/tree_reason.hpp"
#include <vector>
#include <string>
#include <tuple>
#include <iostream>

namespace brain2 {
namespace reasoning {

struct ChessState {
    char board[64];
    bool white_turn;
    
    bool operator==(const ChessState& o) const {
        if (white_turn != o.white_turn) return false;
        for (int i = 0; i < 64; ++i) if (board[i] != o.board[i]) return false;
        return true;
    }
};

} // namespace reasoning
} // namespace brain2

namespace std {
    template<> struct hash<brain2::reasoning::ChessState> {
        size_t operator()(const brain2::reasoning::ChessState& s) const {
            size_t h = s.white_turn ? 1 : 0;
            for(int i = 0; i < 64; ++i) h = h * 31 + s.board[i];
            return h;
        }
    };
}

namespace brain2 {
namespace reasoning {

// Simplified Chess Environment for MCTS
// Goal is simply capturing the opponent's King.
class ChessProblem : public SearchProblem<ChessState, std::hash<ChessState>> {
public:
    ChessState initial() const override {
        ChessState s;
        s.white_turn = true;
        std::string start_board = 
            "rnbqkbnr"
            "pppppppp"
            "........"
            "........"
            "........"
            "........"
            "PPPPPPPP"
            "RNBQKBNR";
        for (int i = 0; i < 64; ++i) s.board[i] = start_board[i];
        return s;
    }

    bool is_goal(const ChessState& s) const override {
        // Did the previous player capture the current player's king?
        char target_king = s.white_turn ? 'K' : 'k';
        for (int i = 0; i < 64; ++i) {
            if (s.board[i] == target_king) return false;
        }
        return true; // King is dead, previous player wins
    }

    double novelty(const ChessState& s) const override {
        return 0.0;
    }

    double heuristic(const ChessState& s) const override {
        // Evaluate material from perspective of PREVIOUS player (who just moved)
        // because we want MCTS to maximize the state it just transitioned into.
        int w_score = 0, b_score = 0;
        for (int i = 0; i < 64; ++i) {
            char c = s.board[i];
            int val = 0;
            if (c == 'P' || c == 'p') val = 10;
            else if (c == 'N' || c == 'n') val = 30;
            else if (c == 'B' || c == 'b') val = 30;
            else if (c == 'R' || c == 'r') val = 50;
            else if (c == 'Q' || c == 'q') val = 90;
            else if (c == 'K' || c == 'k') val = 900;
            
            // Center control bonus
            int r = i / 8, col = i % 8;
            if (r >= 3 && r <= 4 && col >= 3 && col <= 4) val += 2;

            if (c >= 'A' && c <= 'Z') w_score += val;
            else if (c >= 'a' && c <= 'z') b_score += val;
        }
        
        // If it's black's turn now, white just moved, so evaluate from white's perspective
        if (!s.white_turn) return w_score - b_score;
        else return b_score - w_score;
    }

    std::vector<std::tuple<std::string, ChessState, double>> moves(const ChessState& s) const override {
        std::vector<std::tuple<std::string, ChessState, double>> res;
        if (is_goal(s)) return res; // Game over

        auto is_mine = [&](char c) {
            if (c == '.') return false;
            return s.white_turn ? (c >= 'A' && c <= 'Z') : (c >= 'a' && c <= 'z');
        };
        auto is_enemy = [&](char c) {
            if (c == '.') return false;
            return s.white_turn ? (c >= 'a' && c <= 'z') : (c >= 'A' && c <= 'Z');
        };
        auto on_board = [](int r, int c) { return r >= 0 && r < 8 && c >= 0 && c < 8; };

        auto add_move = [&](int from, int to, std::string label) {
            ChessState ns = s;
            char piece = ns.board[from];
            ns.board[to] = piece;
            ns.board[from] = '.';
            
            // Promotion
            if (piece == 'P' && to / 8 == 0) ns.board[to] = 'Q';
            if (piece == 'p' && to / 8 == 7) ns.board[to] = 'q';

            ns.white_turn = !s.white_turn;
            res.push_back({label, ns, 1.0});
        };

        for (int i = 0; i < 64; ++i) {
            char p = s.board[i];
            if (!is_mine(p)) continue;
            
            int r = i / 8;
            int c = i % 8;

            auto slide = [&](int dr, int dc) {
                int nr = r + dr, nc = c + dc;
                while (on_board(nr, nc)) {
                    char dest = s.board[nr * 8 + nc];
                    if (is_mine(dest)) break;
                    add_move(i, nr * 8 + nc, "");
                    if (is_enemy(dest)) break;
                    nr += dr; nc += dc;
                }
            };
            auto step = [&](int dr, int dc) {
                int nr = r + dr, nc = c + dc;
                if (on_board(nr, nc) && !is_mine(s.board[nr * 8 + nc])) {
                    add_move(i, nr * 8 + nc, "");
                }
            };

            char p_lower = std::tolower(p);
            if (p_lower == 'n') {
                int dr[] = {-2, -2, -1, -1, 1, 1, 2, 2};
                int dc[] = {-1, 1, -2, 2, -2, 2, -1, 1};
                for(int k=0; k<8; ++k) step(dr[k], dc[k]);
            } else if (p_lower == 'k') {
                for(int dr=-1; dr<=1; ++dr)
                for(int dc=-1; dc<=1; ++dc)
                    if(dr!=0 || dc!=0) step(dr, dc);
            } else if (p_lower == 'r') {
                slide(-1,0); slide(1,0); slide(0,-1); slide(0,1);
            } else if (p_lower == 'b') {
                slide(-1,-1); slide(-1,1); slide(1,-1); slide(1,1);
            } else if (p_lower == 'q') {
                slide(-1,0); slide(1,0); slide(0,-1); slide(0,1);
                slide(-1,-1); slide(-1,1); slide(1,-1); slide(1,1);
            } else if (p_lower == 'p') {
                int dir = s.white_turn ? -1 : 1;
                int start_row = s.white_turn ? 6 : 1;
                // Move forward 1
                if (on_board(r + dir, c) && s.board[(r + dir) * 8 + c] == '.') {
                    add_move(i, (r + dir) * 8 + c, "");
                    // Move forward 2
                    if (r == start_row && s.board[(r + dir*2) * 8 + c] == '.') {
                        add_move(i, (r + dir*2) * 8 + c, "");
                    }
                }
                // Captures
                if (on_board(r + dir, c - 1) && is_enemy(s.board[(r + dir) * 8 + c - 1]))
                    add_move(i, (r + dir) * 8 + c - 1, "");
                if (on_board(r + dir, c + 1) && is_enemy(s.board[(r + dir) * 8 + c + 1]))
                    add_move(i, (r + dir) * 8 + c + 1, "");
            }
        }
        
        // Relabel moves gracefully
        auto sq_name = [](int idx) {
            int r = idx / 8, c = idx % 8;
            return std::string(1, 'a' + c) + std::to_string(8 - r);
        };
        for(auto& m : res) {
            ChessState ns = std::get<1>(m);
            // find diff to create label like e2e4
            int from = -1, to = -1;
            for (int i=0; i<64; ++i) {
                if (s.board[i] != '.' && ns.board[i] == '.') from = i;
                if (s.board[i] != ns.board[i] && ns.board[i] != '.') to = i;
            }
            if (from != -1 && to != -1) {
                std::get<0>(m) = sq_name(from) + sq_name(to);
            }
        }

        return res;
    }
};

} // namespace reasoning
} // namespace brain2
