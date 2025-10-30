#include "tt.hpp"

namespace hyperion {
namespace engine {

//--
/* TranspositionTable::find */
//--
// Finds a node in the transposition table using its Zobrist hash
// It searches the internal unordered_map for an entry matching the provided hash
    //  hash The Zobrist hash of the position to find
    // A pointer to the Node if the hash is found in the table; otherwise, returns nullptr
Node* TranspositionTable::find(uint64_t hash) {
    
    auto it = table.find(hash);
    if (it == table.end()) {
        return nullptr;
    }

    return it->second;
}

//--
/* TranspositionTable::store */
//--
// Stores a node pointer in the transposition table, associating it with a Zobrist hash
// This function uses the hash as a key and the node pointer as the value
// If a node with the same hash already exists in the table, its pointer will be overwritten
    //  hash The Zobrist hash of the position to store
    //  node A pointer to the Node object to be store
void TranspositionTable::store(uint64_t hash, Node* node) {
    // The [] operator is convenient for both inserting a new element and updating an existing one
    table[hash] = node;
}

//--
/* TranspositionTable::clear */
//--
// Clears all entries from the transposition table
// This is typically called at the beginning of a new search (when calculating a new move)
// to ensure the search starts with a fresh tree
void TranspositionTable::clear() {
    table.clear();
}

//--
/* TranspositionTable::size */
//--
// Returns the current number of nodes stored in the transposition table
// This is am utility function for debugging
// returns The total number of key-value pairs (nodes) currently in the table as an integer
int TranspositionTable::size() {
    // The size() method of unordered_map returns a size_t, so we cast it to int
    // as specified in the header file
    return static_cast<int>(table.size());
}

} // namespace engine
} // namespace hyperion