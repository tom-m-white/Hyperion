// Assumed content of Hyperion/src/cpp/core/move.hpp
#ifndef HYPERION_CORE_MOVE_HPP
#define HYPERION_CORE_MOVE_HPP
#include <string>

#include "constants.hpp" // For piece_type_e, square_e

namespace hyperion {
namespace core {

//--
/* MoveFlag */
//--
// Enum representing special properties or flags associated with a chess move.
// Uses a uint16_t
// Each enumerator represents a distinct bit, allowing combinations of flags.
// this is important becuase a move can be a CAPTURE and a PROMOTION simultaneously.
// PROMOTION_PIECE_SHIFT and PROMOTION_PIECE_MASK are used to embed the promoted piece type within the flags. <-- needs to be implemented in movegen.cpp later.
enum MoveFlag : uint16_t { // Using uint16_t to save space if there are many moves
    NORMAL_MOVE = 0,
    CAPTURE = 1 << 0,
    PROMOTION = 1 << 1,
    EN_PASSANT_CAPTURE = 1 << 2, // Specifically for en passant captures
    CASTLING_KINGSIDE = 1 << 3,
    CASTLING_QUEENSIDE = 1 << 4,
    DOUBLE_PAWN_PUSH = 1 << 5, // For setting the en_passant_square flag
    PROMOTION_PIECE_SHIFT = 6,
    PROMOTION_PIECE_MASK = (P_KNIGHT | P_BISHOP | P_ROOK | P_QUEEN) << PROMOTION_PIECE_SHIFT // Mask to extract piece type
};

//-----------------------------------
// Move operators
// These are overloaded and to better explain why each one is overloaded I added why under the defenition
//----------------------------------



//--
/* operator| (MoveFlag, MoveFlag) */
//--
// Overloads the bitwise OR operator for combining two MoveFlag values.
// WHY: Standard C++ enums (especially enum classes, and often plain enums too) don't directly support bitwise operations
//      in a way that results in the same enum type. The result of a bitwise operation on enum values is typically
//      promoted to its underlying integer type (e.g., uint16_t in this case).
// Without this overload, combining flags would require verbose and error-prone manual casting back to MoveFlag every time:
//   MoveFlag combined = static_cast<MoveFlag>(static_cast<uint16_t>(flag1) | static_cast<uint16_t>(flag2));
// BENEFIT: This overload provides a convenient, highly readable, and type-safe way to combine flags.
// The expression (flag1 | flag2) now directly yields a MoveFlag type, making the code cleaner,
// more intuitive (as OR is the natural way to combine flags), and less prone to casting errors.
//
inline MoveFlag operator|(MoveFlag a, MoveFlag b) {
    return static_cast<MoveFlag>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}

//--
/* operator& (MoveFlag, MoveFlag) */
//--
// Overloads the bitwise AND operator, typically used for checking if a specific flag is set within a composite MoveFlag value.
// WHY: Similar to operator|, applying bitwise AND to enums would result in an integer type without an overload.
//      This operation is essential for testing flag presence, e.g., checking `(current_flags & CAPTURE)`.
// Without this overload, the result of `(flag1 & flag2)` would be an integer, and while usable in boolean contexts
// (e.g., `if (static_cast<uint16_t>(flags) & static_cast<uint16_t>(SOME_MASK))`), having the operator return MoveFlag
// maintains consistency and type safety within the MoveFlag system.
// BENEFIT: Ensures that operations involving ANDing MoveFlags remain within the MoveFlag type system.
// This allows for more complex flag manipulations if ever needed, while also providing a clean way to check flags
// (usually in conjunction with a comparison, e.g., `(flags & MASK) != NORMAL_MOVE` or `(flags & MASK) == MASK`).
inline MoveFlag operator&(MoveFlag a, MoveFlag b) {
    return static_cast<MoveFlag>(static_cast<uint16_t>(a) & static_cast<uint16_t>(b));
}

//--
/* operator|= (MoveFlag&, MoveFlag) */
//--
// Overloads the bitwise OR assignment operator for adding a flag (or flags) to an existing MoveFlag variable.
// WHY: This operator provides syntactic sugar and efficiency for a very common flag operation: setting a new bit.
// Without it, one would have to write:
//   my_flags = my_flags | new_flag;
// Or, even more verbosely if strict typing and no `operator|` were in place:
//   my_flags = static_cast<MoveFlag>(static_cast<uint16_t>(my_flags) | static_cast<uint16_t>(new_flag));
// BENEFIT: This overload allows for the more natural, concise, and often more efficient syntax:
//   my_flags |= new_flag;
// It directly modifies the MoveFlag variable in place, which is idiomatic for C++ bitmask manipulation
// and enhances overall code readability and conciseness.
inline MoveFlag& operator|=(MoveFlag& a, MoveFlag b) {
    a = a | b;
    return a;
}

/* Move */
// Structure representing a chess move.
// Contains information about the source and destination squares, the piece moved,
// any captured piece, and special move flags that describe the nature of the move.

struct Move {
    square_e from_sq;
    square_e to_sq;
    piece_type_e piece_moved;     // Type of the piece that moved
    piece_type_e piece_captured;  // Type of the captured piece (P_NONE if no capture)
    MoveFlag flags;               // Flags for special moves (capture, promotion, castling, etc.)

    /* Move::Move (constructor) */
    // Default constructor for a Move object.
    // Initializes the move with optional source square, destination square, piece moved,
    // piece captured, and flags.
    // Defaults to a NO_SQ -> NO_SQ move with P_NONE pieces and NORMAL_MOVE flags if no arguments are provided.
    Move(square_e from = square_e::NO_SQ, square_e to = square_e::NO_SQ,
         piece_type_e pm = P_NONE, piece_type_e pc = P_NONE, MoveFlag f = NORMAL_MOVE)
        : from_sq(from), to_sq(to), piece_moved(pm), piece_captured(pc), flags(f) {}

    /* Move::is_capture */
    // Checks if the move is a capture.
    // Returns true if the CAPTURE bit is set in the flags member, false otherwise.
    bool is_capture() const { return (flags & CAPTURE) != NORMAL_MOVE; }

    /* Move::is_promotion */
    // Checks if the move is a promotion.
    // Returns true if the PROMOTION bit is set in the flags member, false otherwise.
    bool is_promotion() const { return (flags & PROMOTION) != NORMAL_MOVE; }

    /* Move::is_en_passant */
    // Checks if the move is an en passant capture.
    // Returns true if the EN_PASSANT_CAPTURE bit is set in the flags member, false otherwise.
    bool is_en_passant() const { return (flags & EN_PASSANT_CAPTURE) != NORMAL_MOVE; }

    /* Move::is_castling */
    // Checks if the move is any type of castling (kingside or queenside).
    // Returns true if either CASTLING_KINGSIDE or CASTLING_QUEENSIDE bit is set, false otherwise.
    bool is_castling() const { return (flags & (CASTLING_KINGSIDE | CASTLING_QUEENSIDE)) != NORMAL_MOVE; }

    /* Move::is_kingside_castle */
    // Checks if the move is a kingside castling.
    // Returns true if the CASTLING_KINGSIDE bit is set in the flags member, false otherwise.
    bool is_kingside_castle() const { return (flags & CASTLING_KINGSIDE) != NORMAL_MOVE; }

    /* Move::is_queenside_castle */
    // Checks if the move is a queenside castling.
    // Returns true if the CASTLING_QUEENSIDE bit is set in the flags member, false otherwise.
    bool is_queenside_castle() const { return (flags & CASTLING_QUEENSIDE) != NORMAL_MOVE; }

    /* Move::is_double_pawn_push */
    // Checks if the move is a double pawn push (a pawn moving two squares forward from its starting rank).
    // Returns true if the DOUBLE_PAWN_PUSH bit is set in the flags member, false otherwise.
    bool is_double_pawn_push() const { return (flags & DOUBLE_PAWN_PUSH) != NORMAL_MOVE; }

    /* Move::get_promotion_piece */
    // Retrieves the piece type to which a pawn is promoted, if the move is a promotion.
    // This  Extracts the promotion piece type from the bits reserved in the flags member
    // (using PROMOTION_PIECE_SHIFT and PROMOTION_PIECE_MASK).
    // Returns P_NONE if the move is not a promotion or if the promotion piece is not validly set.
    piece_type_e get_promotion_piece() const {
        if (!is_promotion()) return P_NONE;
        // Extract piece type from flags
        return static_cast<piece_type_e>((static_cast<uint16_t>(flags) >> PROMOTION_PIECE_SHIFT));
    }

    /* Move::make_normal */
    // Static factory method to create a standard (non-capture, non-special) move.
    // Takes the from square, to square, and the type of the piece that moved.
    // Returns a Move object initialized with NORMAL_MOVE flags and P_NONE for captured piece.

    static Move make_normal(square_e from, square_e to, piece_type_e moved_piece) {
        return Move(from, to, moved_piece, P_NONE, NORMAL_MOVE);
    }

    /* Move::make_capture */
    // Static factory method to create a capture move.
    // Takes the from square, to square, the piece moved, and the piece captured.
    // Returns a Move object initialized with the CAPTURE flag.
    static Move make_capture(square_e from, square_e to, piece_type_e moved_piece, piece_type_e captured_piece) {
        return Move(from, to, moved_piece, captured_piece, CAPTURE);
    }

    /* Move::make_promotion */
    // Static factory method to create a promotion move (which can also be a capture).
    // Takes the from square, to square, the pawn moved (piece_moved), the piece it promotes to,
    // a boolean indicating if it's a capture, and the captured piece type (if it is a capture).
    // Sets the PROMOTION flag and embeds the promoted_to piece type into the flags.
    // Also sets the CAPTURE flag if is_cap is true.
    static Move make_promotion(square_e from, square_e to, piece_type_e moved_piece, piece_type_e promoted_to, bool is_cap, piece_type_e captured_if_cap = P_NONE) {
        MoveFlag flag = PROMOTION;
        if(is_cap) flag |= CAPTURE;
        flag |= static_cast<MoveFlag>(static_cast<uint16_t>(promoted_to) << PROMOTION_PIECE_SHIFT);
        return Move(from, to, moved_piece, is_cap ? captured_if_cap : P_NONE, flag);
    }
};

// Helper to convert a square index (0-63) to algebraic notation ("a1"-"h8")
inline std::string square_to_algebraic(int sq) {
    if (sq < 0 || sq > 63) return "-";
    char file = 'a' + (sq % 8);
    char rank = '1' + (sq / 8);
    return {file, rank};
}

// Helper function to convert our Move object to a UCI-compliant string
inline std::string move_to_uci_string(const Move& move) {
    std::string uci_move = square_to_algebraic(static_cast<int>(move.from_sq)) +
                           square_to_algebraic(static_cast<int>(move.to_sq));

    if (move.is_promotion()) {
        switch (move.get_promotion_piece()) {
            case P_QUEEN:  uci_move += 'q'; break;
            case P_ROOK:   uci_move += 'r'; break;
            case P_BISHOP: uci_move += 'b'; break;
            case P_KNIGHT: uci_move += 'n'; break;
            default: break;
        }
    }
    return uci_move;
}
} // namespace core
} // namespace hyperion
#endif // HYPERION_CORE_MOVE_HPP