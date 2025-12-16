#ifndef HYPERION_CORE_CONSTANTS_HPP
#define HYPERION_CORE_CONSTANTS_HPP
#include <cstdint>

namespace hyperion {
namespace core {
    
using bitboard_t = uint64_t;

// Mailbox empty position val
const int EMPTY_MAILBOX_VAL = -1;

// Constants for pieces
const char EMPTY_SQUARE_CHAR = '.';
const char W_PAWN = 'P', W_KNIGHT = 'N', W_BISHOP = 'B', W_ROOK = 'R', W_QUEEN = 'Q', W_KING = 'K';
const char B_PAWN = 'p', B_KNIGHT = 'n', B_BISHOP = 'b', B_ROOK = 'r', B_QUEEN = 'q', B_KING = 'k';

// Square definitions
const int A1 = 0, B1 = 1, C1 = 2, D1 = 3, E1 = 4, F1 = 5, G1 = 6, H1 = 7;
const int A2 = 8, B2 = 9, C2 = 10, D2 = 11, E2 = 12, F2 = 13, G2 = 14, H2 = 15;
const int A3 = 16, B3 = 17, C3 = 18, D3 = 19, E3 = 20, F3 = 21, G3 = 22, H3 = 23;
const int A4 = 24, B4 = 25, C4 = 26, D4 = 27, E4 = 28, F4 = 29, G4 = 30, H4 = 31;
const int A5 = 32, B5 = 33, C5 = 34, D5 = 35, E5 = 36, F5 = 37, G5 = 38, H5 = 39;
const int A6 = 40, B6 = 41, C6 = 42, D6 = 43, E6 = 44, F6 = 45, G6 = 46, H6 = 47;
const int A7 = 48, B7 = 49, C7 = 50, D7 = 51, E7 = 52, F7 = 53, G7 = 54, H7 = 55;
const int A8 = 56, B8 = 57, C8 = 58, D8 = 59, E8 = 60, F8 = 61, G8 = 62, H8 = 63;
constexpr int NUM_SQUARES = 64; // all of these are global int consts

enum class square_e {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    NO_SQ = 64
               
};

//Color
const int WHITE = 0;
const int BLACK = 1;

//Castling Constants
const int WK_CASTLE_FLAG = 1; // 0001
const int WQ_CASTLE_FLAG = 2; // 0010
const int BK_CASTLE_FLAG = 4; // 0100
const int BQ_CASTLE_FLAG = 8; // 1000


// Different Piece Types
enum piece_type_e {
    P_PAWN = 0,
    P_KNIGHT,
    P_BISHOP,
    P_ROOK,
    P_QUEEN,
    P_KING,
    P_NONE, // This is gonna be for an empty or invalid piece type
    NUM_PIECE_TYPES = 6 // Number of actual piece types (Pawn to King)
};

//Rank and File Helpers (derived from square indices)
// 0-indexed ranks (Rank 1 is 0, Rank 8 is 7)
inline int get_rank_idx(square_e s) {
    if (s == square_e::NO_SQ) return -1;
    return static_cast<int>(s) / 8;
}

inline int get_file_idx(square_e s) {
    if (s == square_e::NO_SQ) return -1;
    return static_cast<int>(s) % 8;
}

// Specific rank indices for convenience
const int RANK_1_IDX = 0; // A1..-->..H1
const int RANK_2_IDX = 1; // A2..-->..H2
const int RANK_3_IDX = 2;
const int RANK_4_IDX = 3;
const int RANK_5_IDX = 4;
const int RANK_6_IDX = 5;
const int RANK_7_IDX = 6; // A7..-->..H7
const int RANK_8_IDX = 7; // A8..-->..H8

// Helper to check if a square index is valid (0-63)
inline bool is_valid_sq_idx(int sq_idx) {
    return sq_idx >= 0 && sq_idx < NUM_SQUARES;
}

// This is for pawn bitboards:
const bitboard_t RANK_1_BB = 0x00000000000000FFULL; // Squares A1 to H1
const bitboard_t RANK_2_BB = 0x000000000000FF00ULL; // Squares A2 to H2
const bitboard_t RANK_3_BB = 0x0000000000FF0000ULL;
const bitboard_t RANK_4_BB = 0x00000000FF000000ULL;
const bitboard_t RANK_5_BB = 0x000000FF00000000ULL;
const bitboard_t RANK_6_BB = 0x0000FF0000000000ULL;
const bitboard_t RANK_7_BB = 0x00FF000000000000ULL; // Squares A7 to H7
const bitboard_t RANK_8_BB = 0xFF00000000000000ULL; // Squares A8 to H8

// Files
const bitboard_t FILE_A_BB = 0x0101010101010101ULL;
const bitboard_t FILE_B_BB = FILE_A_BB << 1;
const bitboard_t FILE_C_BB = FILE_A_BB << 2;
const bitboard_t FILE_D_BB = FILE_A_BB << 3;
const bitboard_t FILE_E_BB = FILE_A_BB << 4;
const bitboard_t FILE_F_BB = FILE_A_BB << 5;
const bitboard_t FILE_G_BB = FILE_A_BB << 6;
const bitboard_t FILE_H_BB = 0x8080808080808080ULL;

// For pawn captures to avoid wrap-around
const bitboard_t NOT_FILE_A_BB = ~FILE_A_BB;
const bitboard_t NOT_FILE_H_BB = ~FILE_H_BB;

} // namespace Core
} // namespace Hyperion

#endif