#include "position.hpp"
#include "movegen.hpp"
#include "move.hpp"
#include "constants.hpp"
#include "bitboard.hpp" 
#include "zobrist.hpp"  

#include <iostream>
#include <vector>
#include <string>
#include <chrono> 
#include <numeric> 
#include <algorithm>


std::string move_to_simple_str(const hyperion::core::Move& move) {
    return hyperion::core::square_to_algebraic(move.from_sq) +
           hyperion::core::square_to_algebraic(move.to_sq);
}

// Perft function: recursively counts nodes to a certain depth
uint64_t perft(hyperion::core::Position& pos, int depth, hyperion::core::MoveGenerator& move_gen) {
    if (depth == 0) {
        return 1ULL; // A single leaf node
    }

    //std::vector<hyperion::core::Move> moves;
    //move_gen.generate_legal_moves(pos, moves);
    std::vector<hyperion::core::Move> pseudo_moves;
    move_gen.generate_pseudo_legal_moves(pos, pseudo_moves);
    uint64_t nodes = 0;
    int original_side_to_move = pos.get_side_to_move(); // The player whose move we are testing

    /*
    if (depth == 1) {
        return static_cast<uint64_t>(moves.size()); // Optimization for depth 1
    }

    uint64_t nodes = 0;
    for (const hyperion::core::Move& move : moves) {
        pos.make_move(move);
        nodes += perft(pos, depth - 1, move_gen);
        pos.unmake_move(move); // Essential to unmake the move
    }
    return nodes;
    */
    if (depth == 1) {
        for (const hyperion::core::Move& pseudo_m : pseudo_moves) {
            pos.make_move(pseudo_m);
            // Check if the king OF THE PLAYER WHO JUST MOVED (original_side_to_move)
            // is now in check.
            if (!pos.is_king_in_check(original_side_to_move)) {
                nodes++;
            }
            pos.unmake_move(pseudo_m);
        }
        return nodes;
    }

        // For depths > 1
    for (const hyperion::core::Move& pseudo_m : pseudo_moves) {
        // Your Position::make_move already pushes to history_stack
        pos.make_move(pseudo_m);

            // Check if the king OF THE PLAYER WHO JUST MOVED (original_side_to_move)
            // is now in check.
            // pos.is_king_in_check(color) checks if that color's king is attacked by the opponent.
            // After make_move, pos.side_to_move has flipped. So, the opponent of the
            // original_side_to_move is now pos.get_side_to_move().
            // We need to check if original_side_to_move's king is attacked by the NEW side_to_move.
        if (!pos.is_king_in_check(original_side_to_move)) {
            nodes += perft(pos, depth - 1, move_gen);
        }

        pos.unmake_move(pseudo_m); // Essential to unmake the move
    }
    return nodes;
}

/*
// Perft Divide: runs perft for each move from the root and sums them up
// This is useful for debugging, as it shows node counts for each individual starting move.
void perft_divide(hyperion::core::Position& pos, int depth, hyperion::core::MoveGenerator& move_gen, bool verbose = true) {
    if (depth == 0) {
        if (verbose) std::cout << "Perft Divide at depth 0: 1 node" << std::endl;
        return;
    }
    if (verbose) {
        std::cout << "\n--- Perft Divide for FEN: " << pos.to_fen() << " at depth " << depth << " ---" << std::endl;
    }

    std::vector<hyperion::core::Move> moves;
    move_gen.generate_legal_moves(pos, moves);

    // Optional: Sort moves for consistent output, helps when comparing to other engines/results
    std::sort(moves.begin(), moves.end(), [](const hyperion::core::Move& a, const hyperion::core::Move& b) {
        if (a.from_sq != b.from_sq) return static_cast<int>(a.from_sq) < static_cast<int>(b.from_sq);
        return static_cast<int>(a.to_sq) < static_cast<int>(b.to_sq);
    });

    uint64_t total_nodes = 0;
    for (const hyperion::core::Move& move : moves) {
        pos.make_move(move);
        uint64_t nodes_for_this_move = perft(pos, depth - 1, move_gen); // Note: depth-1
        pos.unmake_move(move);
        if (verbose) {
            std::cout << move_to_simple_str(move) << ": " << nodes_for_this_move << std::endl;
        }
        total_nodes += nodes_for_this_move;
    }
    if (verbose) {
        std::cout << "Moves found: " << moves.size() << std::endl;
        std::cout << "Total nodes: " << total_nodes << std::endl;
    } else {
        // For non-verbose, just print the total to compare against known values
        std::cout << "FEN: " << pos.to_fen() << " | Depth: " << depth << " | Nodes: " << total_nodes << std::endl;
    }
}
*/
void perft_divide(hyperion::core::Position& pos, int depth, hyperion::core::MoveGenerator& move_gen, bool verbose = true) {
    if (depth == 0) {
        if (verbose) std::cout << "Perft Divide at depth 0: 1 node" << std::endl;
        // This case typically isn't hit if depth >= 1 for perft_divide
        return;
    }
    if (verbose) {
        std::cout << "\n--- Perft Divide for FEN: " << pos.to_fen() << " at depth " << depth << " ---" << std::endl;
    }

    std::vector<hyperion::core::Move> pseudo_moves;
    move_gen.generate_pseudo_legal_moves(pos, pseudo_moves); // Get pseudo-legal moves


    std::sort(pseudo_moves.begin(), pseudo_moves.end(), [](const hyperion::core::Move& a, const hyperion::core::Move& b) {
        if (a.from_sq != b.from_sq) return static_cast<int>(a.from_sq) < static_cast<int>(b.from_sq);
        return static_cast<int>(a.to_sq) < static_cast<int>(b.to_sq);
    });

    uint64_t total_nodes = 0;
    int original_side_to_move = pos.get_side_to_move();

    for (const hyperion::core::Move& pseudo_m : pseudo_moves) {
        pos.make_move(pseudo_m);
        uint64_t nodes_for_this_branch = 0;

        if (!pos.is_king_in_check(original_side_to_move)) {
            // If the move is legal, then count its nodes
            nodes_for_this_branch = perft(pos, depth - 1, move_gen); // Note: depth-1
            if (verbose) {
                 // Only print moves that were actually legal
                std::cout << move_to_simple_str(pseudo_m) << ": " << nodes_for_this_branch << std::endl;
            }
            total_nodes += nodes_for_this_branch;
        } else {
            // If verbose and you want to see illegal moves attempted and their "0" count:
            // if (verbose) {
            //     std::cout << move_to_simple_str(pseudo_m) << ": 0 (illegal - king in check)" << std::endl;
            // }
        }
        pos.unmake_move(pseudo_m);
    }

    if (verbose) {
        // Note: moves.size() here is pseudo_moves.size(). You might want to count legal moves separately if needed.
        std::cout << "Pseudo-moves found: " << pseudo_moves.size() << std::endl;
        std::cout << "Total (legal) nodes: " << total_nodes << std::endl;
    } else {
        std::cout << "FEN: " << pos.to_fen() << " | Depth: " << depth << " | Nodes: " << total_nodes << std::endl;
    }
}


// Basic check function
inline void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "Assertion failed: " << message << std::endl;
        
        exit(1);
    }
    // std::cout << "Check passed: " << message << std::endl;
}


void run_perft_test(const std::string& fen, int depth, uint64_t expected_nodes, 
                    hyperion::core::Position& pos, hyperion::core::MoveGenerator& move_gen) {
    pos.set_from_fen(fen);
    std::cout << "\nTesting FEN: " << fen << " at depth " << depth << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    uint64_t actual_nodes = 0;
    // For deeper depths, direct perft is faster if you only need the total.
    // ror shallower depths or debugging, perft_divide is more informative.
    if (depth <= 5) { // Adjust this threshold as needed
         perft_divide(pos, depth, move_gen, false); // Run verbose perft_divide
         pos.set_from_fen(fen); // Reset position
         actual_nodes = perft(pos, depth, move_gen); // Recalculate for the check
    } else {
        actual_nodes = perft(pos, depth, move_gen);
        std::cout << "Total nodes: " << actual_nodes << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
    std::cout << "Time taken: " << duration_ms.count() << " ms" << std::endl;
    std::cout << "Nodes per second: " << (actual_nodes * 1000.0 / (duration_ms.count() > 0 ? duration_ms.count() : 1)) << std::endl;

    check(actual_nodes == expected_nodes, "Perft node count mismatch. Expected: " + std::to_string(expected_nodes) + ", Got: " + std::to_string(actual_nodes));
    std::cout << "Perft Test Passed for FEN: " << fen << " at depth " << depth << std::endl;
}


int main() {
    hyperion::core::Zobrist::initialize_keys();
    hyperion::core::initialize_attack_tables(); 

    hyperion::core::Position pos;
    hyperion::core::MoveGenerator move_gen;

    std::cout << "========== Starting MoveGen Tests ==========" << std::endl;

    // --- test Case 0: debug station ---
    //std::string debug_fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1";
    //run_perft_test(debug_fen, 3, 11003, pos, move_gen);
    //std::string fen_after_d4 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1";
    //run_perft_test(fen_after_d4, 2, 539, pos, move_gen); 
    //std::string fen_after_b1b3 = "rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1";
    //run_perft_test(fen_after_b1b3, 2, 8807, pos, move_gen);
    //std::string fen_pos5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
    //run_perft_test(fen_pos5, 1, 44, pos, move_gen);

    //run_perft_test("k4b2/3p3p/8/5R2/8/8/2p5/K7 w - - 0 1",1,16,pos,move_gen);
    //run_perft_test("k4b2/3p3p/8/5Q2/8/8/2p5/K7 w - - 0 1",1,25,pos,move_gen);
    //run_perft_test("k7/3p3p/8/5Q2/8/8/2p5/K7 w - - 0 1",1,25,pos,move_gen);
    //run_perft_test("1k6/8/8/8/8/3n1p2/6P1/1K6 w - - 0 1",1,6,pos,move_gen);
    //run_perft_test("6k1/6n1/8/4n3/3Bp3/8/6K1/8 w - - 1 1",1,17,pos,move_gen);
    //run_perft_test("6k1/6n1/8/8/3Bp3/8/6K1/8 w - - 1 1",1,19,pos,move_gen);
    //run_perft_test("6k1/6n1/8/8/4p3/2B5/6K1/8 w - - 1 1",1,17,pos,move_gen);
    //run_perft_test("6k1/8/8/4n1p1/8/2B5/6K1/8 w - - 1 1",1,15,pos,move_gen);
    //run_perft_test("6k1/8/8/4n1p1/8/8/3B2K1/8 w - - 1 1",1,15,pos,move_gen);
    //run_perft_test("6k1/8/8/p3n3/8/8/3B2K1/8 w - - 1 1",1,16,pos,move_gen);
    //run_perft_test("6k1/8/8/4n3/1p6/8/3B2K1/8 w - - 1 10",1,15,pos,move_gen);
    //run_perft_test("6k1/8/8/p3n3/p7/8/1P4K1/R7 w - - 1 1",1,19,pos,move_gen);
    //run_perft_test("6k1/p7/8/4n3/p7/8/1P4K1/R7 w - - 1 1",1,19,pos,move_gen);
    //run_perft_test("6k1/p7/8/p3n3/8/8/1P4K1/R7 w - - 1 1",1,20,pos,move_gen);
    //run_perft_test("6k1/p7/8/p3n3/8/8/1P4K1/Q7 w - - 1 1", 1, 20,pos,move_gen);
    //run_perft_test("6k1/8/p7/p3n3/8/8/1P4K1/Q7 w - - 1 1",1,20,pos,move_gen);
    //run_perft_test("6k1/8/8/p3n3/8/8/1P4K1/Q7 w - - 1 1",1,21,pos,move_gen);
    //run_perft_test("6k1/8/8/8/8/8/6K1/Q7 w - - 1 1",1 , 29 , pos ,move_gen);
    //run_perft_test("rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2", 1, 22, pos, move_gen);

    // --- end debug cases ---

    // --- Test Case 1: Initial Position ---
    // Known Perft values for starting position:
    // Depth 1: 20
    // Depth 2: 400
    // Depth 3: 8,902
    // Depth 4: 197,281
    // Depth 5: 4,865,609
    // Depth 6: 119,060,324 (takes a while)
    std::string start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    run_perft_test(start_fen, 1, 20, pos, move_gen);
    run_perft_test(start_fen, 2, 400, pos, move_gen);
    run_perft_test(start_fen, 3, 8902, pos, move_gen);
    run_perft_test(start_fen, 4, 197281, pos, move_gen);
    run_perft_test(start_fen, 5, 4865609, pos, move_gen);
    run_perft_test(start_fen, 6, 119060324, pos, move_gen);
    //run_perft_test(start_fen,7 , 3195901860, pos, move_gen);

    // --- Test Case 2: Kiwipete (tests many castling/EP scenarios) ---
    // FEN: r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
    // Depth 1: 48
    // Depth 2: 2,039
    // Depth 3: 97,862
    // Depth 4: 4,085,603
    //std::string kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    //run_perft_test(kiwipete_fen, 1, 48, pos, move_gen);
    //run_perft_test(kiwipete_fen, 2, 2039, pos, move_gen);
    //run_perft_test(kiwipete_fen, 3, 97862, pos, move_gen);
    //run_perft_test(kiwipete_fen, 4, 4085603, pos, move_gen);


    // --- Test Case 3: Position with promotions, checks, etc. ---
    // FEN: 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1
    // Depth 1: 14
    // Depth 2: 191
    // Depth 3: 2,812
    //std::string fen_pos3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    //run_perft_test(fen_pos3, 1, 14, pos, move_gen);
    //run_perft_test(fen_pos3, 2, 191, pos, move_gen);
    // run_perft_test(fen_pos3, 3, 2812, pos, move_gen);

    // --- Test Case 4: From Chess Programming Wiki (Perft results page) ---
    // FEN: r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
    // Depth 1: 6
    // Depth 2: 264
    // Depth 3: 9,467
    //std::string fen_pos4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    //run_perft_test(fen_pos4, 1, 6, pos, move_gen);
    //run_perft_test(fen_pos4, 2, 264, pos, move_gen);
    //run_perft_test(fen_pos4, 3, 9467, pos, move_gen);


    // You can add more specific non-Perft tests here if you want to verify
    // the generation of a particular type of move from a custom setup.
    // For example, set up a board for en passant and check if only that EP move is generated
    // if it's the only legal move, or if it's present in the list of moves.

    std::cout << "\n========== All MoveGen Tests Completed Successfully! ==========" << std::endl;

    return 0;
}