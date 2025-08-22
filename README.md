# Hyperion Chess Engine
Hyperion is a MCTS-NN based chess engine built primarily in C++. The Engine is still in development, but should be done by the end of Summer 2025.

## Perft test results:
- Nodes per second: 4.39454e+07 (43.9 Million NPS)

## How to build?
1. Open a terminal in the `hyperion` directory

2. Next, cd into the build directory (`hyperion/build/`)

3. Now run `cmake -G "MinGW Makefiles" ..`

4. Still in the build directory, run `mingw32-make` (if this doesn't work, try just `make` instead)

5. Run the executables by running the command `.\bin\HyperionEngine.exe`
 
## How to build with performance enhancements:
1. Open a terminal in the `hyperion` directory

2. Next, cd into the build directory (`hyperion/build/`)

3. Now run `cmake -G "MinGW Makefiles" -DHYPERION_ENABLE_BMI2=ON -DCMAKE_BUILD_TYPE=Release ..'

4. Still in the build directory, run `mingw32-make -jN` where N is the number of cores in your cpu (optional, but slower `mingw32-make`)

5. Run the executables by running the command `.\bin\HyperionEngine.exe`

## Tasks Completed
- [x] Chess Logic Library
- [x] Basic Monte-Carlo Tree Search Implementation
- [X] Initial Neural Network Creation (Supervised Learning)
- [X] Initial Neural Network Implementation 
- [ ] Neural Network Self-Play (Reinforcement Learning)
- [ ] Refinements/Advanced Features
