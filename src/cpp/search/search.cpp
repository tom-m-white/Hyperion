#include "search.hpp"
#include "eval.hpp"
#include "../core/movegen.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>

namespace hyperion { 
namespace engine {

// Ive included both the tracitional MCTS and a hand crafted evalutation MCTS. THe handcrafted evaluation is a side project, but it is effective at seeing
// what a traditional nn-mcts would get. I did include the handcrafted evaluation below. You much just comment / uncomment everything shown below

// UCT exploration constant. Higher values favor exploring lessvisited nodes
constexpr double UCT_C = 1.414; // sqrt(2)
// constexpr double UCT_C = 3.14; // pi
// you kinda just need to try around these values and see what works
// to be honest ive changed this number so much and it is SUPPOSED to do something, but doesnt really do much :/
// possible bug maybe?
// constexpr double UCT_C = .6; // number I pulled out of my ass


//--
/* Search::Search */
//--
// Constructs a Search object, initializing the random number generator
// The random generator is used for the simulation (playout) phase of MCTS
Search::Search() : random_generator(std::random_device{}()) {
    std::string model_path = "/home/white/Hyperion/data/completed_models/hyperion_16b_196f.pt";
    nn_ = std::make_unique<NeuralNetwork>(model_path);
}


//--
/* Search::find_best_move */
//--
// The main entry point for the Monte Carlo Tree Search (MCTS)
// It iteratively builds a game tree for a specified duration, then selects the best move
    //  root_pos: The starting position of the search
    //  time_limit_ms: The maximum time in milliseconds to run the search
    // The best core::Move found for the root_pos
core::Move Search::find_best_move(core::Position& root_pos, int time_limit_ms) {
    root_node = std::make_unique<Node>();
    tt.clear();
    tt.store(root_pos.current_hash, root_node.get());

    // Evaluate the root once to get its value and policy for its children.
    expand_and_evaluate(root_node.get(), root_pos);

    auto start_time = std::chrono::steady_clock::now();
    int iterations = 0;

    // We start from 1 because we did one evaluation already.
    for (iterations = 1; ; ++iterations) { 
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= time_limit_ms) {
            break;
        }

        core::Position search_pos = root_pos;
        Node* node = select(root_node.get(), search_pos);

        // We selected a leaf. Now evaluate it to expand it.
        // Check if it's a real terminal node (game over)
        core::MoveGenerator mg;
        std::vector<core::Move> moves;
        mg.generate_legal_moves(search_pos, moves);

        if (moves.empty()) {
            double terminal_value = search_pos.is_in_check() ? -1.0 : 0.0;
            backpropagate(node, terminal_value);
        } else {
            // It's a leaf but not a terminal position, so expand and evaluate it.
            expand_and_evaluate(node, search_pos);
        }
    }
    
    std::cout << "info depth " << iterations << " nodes " << tt.size() << std::endl;
    return get_best_move_from_root();
}

// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
/*
core::Move Search::find_best_move(core::Position& root_pos, int time_limit_ms) {
    // --- Setup ---
    // Initialize the search tree with a root node
    root_node = std::make_unique<Node>();
    
    // Clear the transposition table from any previous search
    tt.clear();
    // Store the root node in the transposition table
    tt.store(root_pos.current_hash, root_node.get());

    auto start_time = std::chrono::steady_clock::now();
    int iterations = 0;

    // --- Main MCTS Loop ---
    // The loop continues until the time limit is exceeded
    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= time_limit_ms) {
            break;
        }

        // Create a copy of the position to modify during this iteration's traversal
        core::Position search_pos = root_pos; 
        
        // MCTS consists of four main phases per iteration:
        // 1. Selection: Traverse the tree to find a promising leaf node
        Node* node = select(root_node.get(), search_pos);
        // 2. Expansion: Add a new child to the selected node
        node = expand(node, search_pos);
        // 2. Expansion
        if (!is_terminal(search_pos)) { // Only expand if not a terminal node
            node = expand(node, search_pos);
        }
        // 3. Simulation: Run a random playout from the new node
        double result = simulate(search_pos);
        // 4. Backpropagation: Update node statistics back up the tree
        backpropagate(node, result);
        
        iterations++;
    }
    
    // Output search statistics
    std::cout << "info depth " << iterations << " nodes " << tt.size() << std::endl;

    // After the search, determine the best move from the root
    return get_best_move_from_root();
} */
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================

//--
/* Search::select */
//--
// Performs the selection phase of MCTS
// It traverses the tree from a given node, choosing the child with the highest UCT score at each step,
// until it reaches a leaf node (a node that is not fully expanded or is terminal)
    //  node: The starting node for the selection process (usually the root)
    //  pos: The board position, which is updated as the selection traverses the tree
    // A pointer to the selected leaf Node
Node* Search::select(Node* node, core::Position& pos) {
    while (true) {
        /*if (node == root_node.get()) {
            std::cout << "DEBUG: Select is at ROOT node." << std::endl;
        } else {
            std::cout << "DEBUG: Select is at node with move " << core::move_to_uci_string(node->move) << std::endl;
        }


        if (node->children.empty()) {
            std::cout << "DEBUG: Selected a leaf node. Returning." << std::endl;
            return node;
        }
        */
        Node* best_child = nullptr;
        double max_score = -std::numeric_limits<double>::infinity();

        for (const auto& child : node->children) {
            double score = uct_score(child.get(), node->visits);
            if (score > max_score) {
                max_score = score;
                best_child = child.get();
            }
        }
        
        if (!best_child) return node; // Should not happen

        pos.make_move(best_child->move);
        node = best_child;
    }
}

void Search::expand_and_evaluate(Node* node, const core::Position& pos) {

    InferenceResult nn_result = nn_->infer(pos);
    double value = nn_result.value;
     
    backpropagate(node, value);

    core::MoveGenerator move_gen;
    std::vector<core::Move> legal_moves;
    move_gen.generate_legal_moves(pos, legal_moves);

    for (const auto& move : legal_moves) {
        node->children.push_back(std::make_unique<Node>(node, move));
        Node* child = node->children.back().get();
        
        try {
            int policy_index = get_policy_index(move, pos);
           
            child->policy_prior = std::exp(nn_result.policy[policy_index]);
        } catch (const std::runtime_error& e) {
            child->policy_prior = 0.001; 
        }
    }
}

void Search::backpropagate(Node* node, double value) {
    while (node != nullptr) {
        node->visits++;
        node->value += (node->parent == nullptr) ? value : -value; 
        
        node = node->parent;
        value = -value;
    }
}

//--
/* Search::uct_score */
//--
// Calculates the UCT (Upper Confidence Bound for Trees) score for a given node
// This score balances exploitation (choosing known good moves) and exploration (trying new moves)
    //  node: The child node for which to calculate the score
    //  parent_visits: The number of times the parent of 'node' has been visited
    // The calculated UCT score as a double

double Search::uct_score(const Node* node, int parent_visits) const {
    double q_value = (node->visits == 0) ? 0.0 : -node->value / node->visits;

    double u_value = UCT_C * node->policy_prior * (std::sqrt(parent_visits) / (1.0 + node->visits));
    
    return q_value + u_value;
}
//--
/* Search::get_best_move_from_root */
//--
// Determines the best move from the root node after the MCTS search is complete
// The most robust move is the one that was explored the most times
    // The core::Move corresponding to the most visited child of the root node
core::Move Search::get_best_move_from_root() {
    int max_visits = -1;
    core::Move best_move; 
    
    if (!root_node) {
        return best_move;
    }

    // The best move is the one corresponding to the most visited child
    for (const auto& child : root_node->children) {
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best_move = child->move;
        }
    }
    return best_move;
}
} // namespace engine
} // namespace hyperion