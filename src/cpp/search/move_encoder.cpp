#include "move_encoder.hpp"
#include <stdexcept>
#include <map>

namespace hyperion {
namespace engine {

namespace {

// Constants matching move_encoder.py
constexpr int QUEEN_MOVE_PLANES = 56;
constexpr int KNIGHT_MOVE_PLANES = 8;
constexpr int UNDERPROMOTION_MOVE_PLANES = 9;
constexpr int TOTAL_MOVE_PLANES = QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES + UNDERPROMOTION_MOVE_PLANES; // Should be 73

//Maps for direction lookups
// Queen-like move directions
const std::map<int, int> QUEEN_DIRECTION_MAP = {
    {8, 0}, {9, 1}, {1, 2}, {-7, 3}, {-8, 4}, {-9, 5}, {-1, 6}, {7, 7}
};

// Knight move directions
const std::map<int, int> KNIGHT_DIRECTION_MAP = {
    {17, 0}, {10, 1}, {-6, 2}, {-15, 3}, {-17, 4}, {-10, 5}, {6, 6}, {15, 7}
};

// Underpromotion piece types
const std::map<core::piece_type_e, int> UNDERPROMOTION_MAP = {
    {core::P_KNIGHT, 0}, {core::P_BISHOP, 1}, {core::P_ROOK, 2}
};

} // end anonymous namespace

int get_policy_index(const core::Move& move, const core::Position& pos) {
    using namespace core;

    int start_sq = static_cast<int>(move.from_sq);
    int end_sq = static_cast<int>(move.to_sq);
    int delta = end_sq - start_sq;
    int move_type_idx = -1;

    //Knight Moves
    if (move.piece_moved == P_KNIGHT) {
        auto it = KNIGHT_DIRECTION_MAP.find(delta);
        if (it != KNIGHT_DIRECTION_MAP.end()) {
            move_type_idx = QUEEN_MOVE_PLANES + it->second;
        }
    }
    //Case 2: Underpromotions
    else if (move.is_promotion() && move.get_promotion_piece() != P_QUEEN) {
        int color = pos.get_side_to_move();
        int direction_offset = -1;

        if (color == WHITE) {
            if (delta == 8) direction_offset = 0; // N
            else if (delta == 7) direction_offset = 1; // NW
            else if (delta == 9) direction_offset = 2; // NE
        } else { // BLACK
            if (delta == -8) direction_offset = 0; // S
            else if (delta == -9) direction_offset = 1; // SW
            else if (delta == -7) direction_offset = 2; // SE
        }
        
        auto promo_it = UNDERPROMOTION_MAP.find(move.get_promotion_piece());
        if (promo_it != UNDERPROMOTION_MAP.end() && direction_offset != -1) {
            move_type_idx = QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES + 
                            (promo_it->second * 3 + direction_offset);
        }
    }
    // All other moves (Queen-like)
    else {
        int start_file = start_sq % 8;
        int start_rank = start_sq / 8;
        int end_file = end_sq % 8;
        int end_rank = end_sq / 8;

        int dist = std::max(std::abs(start_file - end_file), std::abs(start_rank - end_rank));
        if (dist > 0) {
            int dir_delta = delta / dist;
            auto it = QUEEN_DIRECTION_MAP.find(dir_delta);
            if (it != QUEEN_DIRECTION_MAP.end()) {
                move_type_idx = it->second * 7 + (dist - 1);
            }
        }
    }
    
    if (move_type_idx == -1) {
        throw std::runtime_error("Could not encode move: " + move_to_uci_string(move));
    }

    return start_sq * TOTAL_MOVE_PLANES + move_type_idx;
}

} // namespace engine
} // namespace hyperion