#include "eval.hpp"
#include "../core/movegen.hpp"
#include "../core/bitboard.hpp"
#include "../core/constants.hpp"
#include "../core/position.hpp"
#include "../core/move.hpp"
#include "search.hpp"
#include <vector>
#include <random>

namespace hyperion {
namespace engine {

// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
// For fun I Will add a temporary static evaluation to see how good I can make this. a lot of this will be refactored but hey, the source code is on github so who cares if I cange it
// Static evaluation for fun:
/*
// --- Piece Values ---
constexpr int PAWN_VALUE   = 100;
constexpr int KNIGHT_VALUE = 320;
constexpr int BISHOP_VALUE = 330;
constexpr int ROOK_VALUE   = 500;
constexpr int QUEEN_VALUE  = 900;
constexpr int KING_VALUE   = 20000;

using core::P_PAWN;
using core::P_KNIGHT;
using core::P_BISHOP;
using core::P_ROOK;
using core::P_QUEEN;
using core::P_KING;
using core::WHITE;
using core::BLACK;

const int piece_values[] = {
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE
};

const int pawn_pst[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,  
     10,  10,  10, -10, -10,  10,  10,  10,  
     10,  10,  20,  30,  30,  20,  10,  10,  
     20,  25,  30,  50,  50,  30,  25,  20,  
     30,  40,  50,  60,  60,  50,  40,  30,  
     50,  60,  70,  80,  80,  70,  60,  50,  
    120, 130, 140, 150, 150, 140, 130, 120,  
      0,   0,   0,   0,   0,   0,   0,   0  
};
const int knight_pst[64]={-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,5,5,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50};
const int bishop_pst[64]={-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20};
const int rook_pst[64]={0,0,0,5,5,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,0,0,0,0,0};
const int queen_pst[64]={-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,0,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20};
const int king_pst[64]={20,30,10,0,0,10,30,20,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20};

const int* piece_square_tables[] = {
    pawn_pst,
    knight_pst,
    bishop_pst,
    rook_pst,
    queen_pst,
    king_pst
};
*/
//--
/* static_evaluate */
//--
// Calculates a static evaluation score for a given board position without looking ahead.
// The function iterates through each piece type for both White and Black. For each piece
// on the board, it adds (for White) or subtracts (for Black) two values: the material
// value of the piece (e.g., Queen = 900) and a positional bonus from a Piece-Square
// Table (PST). The PST rewards pieces for being on strategically advantageous squares.
// The final score is returned from the perspective of the side to move, a common
// practice in negamax-style search algorithms. This means a positive score is always
// advantageous for the current player.
/*
double static_evaluate(const core::Position& pos) {
    int score = 0;
    
    // Loop from Pawn to King
    for (int p_type_idx = P_PAWN; p_type_idx <= P_KING; ++p_type_idx) {
        core::piece_type_e p_type = static_cast<core::piece_type_e>(p_type_idx);
        
        // Get the PST for the current piece type
        const int* pst = piece_square_tables[p_type_idx];
        
        // --- White pieces ---
        core::bitboard_t white_pieces = pos.get_pieces(p_type, WHITE);
        while(white_pieces) {
            int sq = core::pop_lsb(white_pieces); // Use int for easier indexing
            // Add material value
            score += piece_values[p_type_idx];
            // Add positional value from the PST
            score += pst[sq];
        }

        // --- Black pieces ---
        core::bitboard_t black_pieces = pos.get_pieces(p_type, BLACK);
        while(black_pieces) {
            int sq = core::pop_lsb(black_pieces);
            // Subtract material value
            score -= piece_values[p_type_idx];
            // Subtract positional value from the PST, using the mirrored square index
            score -= pst[sq ^ 56]; // (black pawn on e7 (sq=52) uses white pawn pst for e2 (52^56=12))
        }
    }

    // Return score from the perspective of the side to move
    // If it's White's turn, a positive score is good for White.
    // If it's Black's turn, a positive score is good for White, so its bad for Black (-score).
    return (pos.get_side_to_move() == WHITE) ? static_cast<double>(score) : -static_cast<double>(score);
}
*/
//--
/* limited_depth_playout */
//--
// Simulates a short, random game (a "playout") from a given starting position to
// quickly estimate the game's outcome. It plays a fixed number of pseudo-legal moves
// (defined by `MAX_PLAYOUT_DEPTH`), selecting a move randomly at each step. This
// simulation modifies a local copy of the `position` object. If the simulation
// encounters a checkmate, stalemate, or a draw by the 50-move rule, it returns
// an immediate score (-1.0 for a loss, 0.0 for a draw). After the playout reaches
// its depth limit, it calls `static_evaluate` on the final position. The result is
// then normalized to a value between -1.0 and 1.0 using `std::tanh`, making it
// suitable for use in a Monte Carlo Tree Search (MCTS) algorithm.
/*
double limited_depth_playout(core::Position position, std::mt19937& gen) {
    core::MoveGenerator move_gen;
    std::vector<core::Move> move_list;
    const int MAX_PLAYOUT_DEPTH = 20; // Simulate 20 moves (10 per side) deep

    // We need to know who the player was at the *start* of the simulation
    // to correctly interpret the final static evaluation score.
    int starting_player = position.get_side_to_move();

    for (int depth = 0; depth < MAX_PLAYOUT_DEPTH; ++depth) {
        move_list.clear();
        move_gen.generate_pseudo_legal_moves(position, move_list);

        if (move_list.empty()) {
            // Checkmate is a huge loss for the current player
            if (position.is_king_in_check(position.get_side_to_move())) {
                 // The player whose turn it is got checkmated.
                 // This is the worst possible outcome for them.
                 return -1.0; 
            } else {
                // Stalemate is a draw.
                return 0.0;
            }
        }
        
        if (position.halfmove_clock >= 100) {
            return 0.0;
        }

        std::uniform_int_distribution<> distrib(0, move_list.size() - 1);
        const core::Move& random_move = move_list[distrib(gen)];
        position.make_move(random_move);
    }
    
    // After the depth limit, do a static evaluation
    double final_score = static_evaluate(position);
    
    // The static_evaluate function returns the score from the perspective of the side whose turn it is
    // at the *end* of the playout. We need to flip it if the player has changed.
    if (position.get_side_to_move() != starting_player) {
        final_score = -final_score;
    }

    // Normalize the score to be between -1 and 1 for MCTS.
    // tanh is a great function for this. We scale the score so that +/- 3 pawns is a near certain win/loss.
    return std::tanh(final_score / (PAWN_VALUE * 3.0));
}
*/
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================


//--
/* random_playout */
//--
// Simulates a complete game from a given position by making random moves for both sides
// This function is the core of the "simulation" phase in Monte Carlo Tree Searc, the point of this whole thing
// The simulation ends when a terminal state (checkmate, stalemate, or 50-move rule draw) is reached
    //  position: The board state from which the random playout will begin. It is passed by value to avoid modifying the original
    //  gen: A reference to a Mersenne Twister random number generator for selecting moves
    // The result of the game from the perspective of the starting player: 1.0 for a win, -1.0 for a loss, and 0.0 for a draw
double random_playout(core::Position position, std::mt19937& gen) {
    core::MoveGenerator move_gen;
    std::vector<core::Move> move_list;
    // Store the side to move at the beginning of the playout to correctly evaluate the final score
    int initial_player = position.get_side_to_move();

    // The main game loop for the random simulation
    while (true) {
        move_list.clear();
        // Generate all legal moves for the current player
        move_gen.generate_legal_moves(position, move_list);

        //Check for Game Over conditions
        if (move_list.empty()) {
            // No legal moves available
            if (position.is_in_check()) {
                // If the current player is in check and has no moves, it's checkmate
                // A loss for the current player (-1.0), a win for the other (+1.0)
                return (position.get_side_to_move() == initial_player) ? -1.0 : 1.0;
            } else {
                // If the current player is not in check and has no moves, it's a stalemate
                return 0.0;
            }
        }
        
        //Check for draw by the 50-move rule
        // The game is a draw if 50 full moves (100 half-moves) occur without a capture or pawn move
        if (position.halfmove_clock >= 100) {
            return 0.0;
        }

        //Pick and play a random move
        // Create a uniform distribution to select a random move index
        std::uniform_int_distribution<> distrib(0, move_list.size() - 1);
        // Select a random move from the list of legal moves
        const core::Move& random_move = move_list[distrib(gen)];
        // Apply the chosen move to the board to advance the position
        position.make_move(random_move);
    }
    // This line is un reachable as the loop only terminates via a return, but it prevents compiler warnings
    return 0.0;
}

} // namespace engine
} // namespace hyperion