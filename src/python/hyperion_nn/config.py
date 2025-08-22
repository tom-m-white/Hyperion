# will be used for the following:
# - storing hyperparameters for the neural network
#   - ex: learning rate, batch size, number of epochs, number of layers, etc.
# - storing some path directories
# - NN input/output dimensions

import torch
import os
import math

class PathsConfig:

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    

    RAW_TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'raw-games')
    PROCESSED_TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'processed-games')
    RAW_VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'raw-validation')
    PROCESSED_VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'processed-validation')

    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    CHECKPOINT_DIR = os.path.join(MODELS_DIR, 'checkpoints')

    LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
    STEPS_LOG_DIR = os.path.join(DATA_DIR, 'steps')
    POST_VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'post-validation-data')


class HardwareBasedConfig:

    # ! IMPORTANT: these is ARBITRARY and should be changed to match the actual hardware capabilities (vram, gpu, etc.)
    BATCH_SIZE = 64 #256 

    NUM_WORKERS = 8
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

class TrainingConfig:


    # ~~ Training hyperparameters ~~
    LEARNING_RATE = 0.001  # how far the "steps" are in the gradient descent algorithm, just trust me that this is the right value

    TOTAL_SAMPLES_TO_TRAIN = 1_000_000_000  # total number of samples to train on, this is ARBITRARY and should be changed later
    
    TOTAL_TARGET_TRAINING_STEPS = TOTAL_SAMPLES_TO_TRAIN // HardwareBasedConfig.BATCH_SIZE + 1  # total number of training steps, this is ARBITRARY and should be changed later

    WEIGHT_DECAY = 0.0001  # prevents overfitting by adding a penalty for large weights (L2 regularization)

    OPTIMIZER = 'adam'  # optimizer to use for training (e.g., 'adam', 'sgd', etc.)

    VALIDATION_SPLIT = 0.02  # 2% of the data will be used for validation

    # ~~ Logging/Checkpointing ~~
    SAVE_CHECKPOINTS_EVERY_N_STEPS = 25_000  # save a checkpoint every N training steps
    #VALIDATE_EVERY_N_STEPS = 3 * SAVE_CHECKPOINTS_EVERY_N_STEPS
    VALIDATE_EVERY_N_STEPS = 5_000
    LOG_EVERY_N_STEPS = 1  # log training progress every N training steps

    # validate the model every N training steps
    # IMPORTANT: this is ARBITRARY and should be researched more, though it seems to be not that complicated
    # POLICY_LOSS_WEIGHT = 1.0  # weight for the policy loss in the total loss calculation
    # VALUE_LOSS_WEIGHT = 1.0  # weight for the value loss in the total loss calculation

    # This will prob not be used in the future, but it is here in case
    # MOMENTUM = 0.9  # momentum for the optimizer (if applicable, e.g., for SGD)

class ModelConfig:

    NUM_INPUT_PLANES = 20  # number of input planes (see fen_parser.py for details)
    
    INPUT_SHAPE = (NUM_INPUT_PLANES, 8, 8)  # input shape for the model

    TOTAL_OUTPUT_PLANES = 72  # total number of output planes (see move_encoder.py for details)
    TOTAL_OUTPUT_SIZE = 64 * TOTAL_OUTPUT_PLANES  # total output size for the policy head

    # ! IMPORTANT: these are ARBITRARY and should be changed later with finalized NN Arch
    # size table: b = residual block (depth), f = filters (width), * = tested
    #
    # |-----------|-----------|------------|------------|------------|
    # | 20b x 64f | 20b x 96f | 20b x 128f | 20b x 196f |*20b x 256f*|
    # |-----------|-----------|------------|------------|------------|
    # | 16b x 64f | 16b x 96f |*16b x 128f*|*16b x 196f*| 16b x 256f | 16b x 512f
    # |-----------|-----------|------------|------------|------------|
    # | 12b x 64f |*12b x 96f*|*12b x 128f*| 12b x 196f | 12b x 256f |
    # |-----------|-----------|------------|------------|------------|
    # |  8b x 64f | *8b x 96f*|  8b x 128f | *8b x 196f*| 8b x 256f  |
    # |-----------|-----------|------------|------------|------------|
    # | *4b x 64f*|  4b x 96f |  4b x 128f |  4b x 196f | 4b x 256f  |
    # |-----------|-----------|------------|------------|------------|

    NUM_RESIDUAL_BLOCKS = 16
    NUM_FILTERS =  512

    POLICY_HEAD_SIZE = 64 * 73  # 64 squares * 73 possible moves (including underpromotions)


# ! IMPORTANT: this is ARBITRARY and again i have NO CLUE what this means, or if we are even going to use it
class SelfPlayConfig:
    """
    Configuration for the self-play data generation process.
    """
    # TODO: Finalize what this will be
    # Path to the compiled C++ engine executable
    ENGINE_EXECUTABLE_PATH = os.path.join(PathsConfig.ROOT_DIR, "build", "HyperionEngine") # Example name

    # ^ IDEK if we are gonna use this, it initially seem likely not
    # Number of MCTS simulations to run for each move during self-play
    MCTS_SIMULATIONS_PER_MOVE = 800

    # Number of games to generate in each self-play iteration
    GAMES_PER_ITERATION = 5000

    # ! IMPORTANT: I have NO CLUE what this means
    # Dirichlet noise alpha value for root node exploration
    DIRICHLET_ALPHA = 0.3

    # ! IMPORTANT: this is ARBITRARY and should be researched more, as it seems it is very important and useful
    # Temperature for move selection during the opening phase of self-play
    # Higher temperature = more exploration.
    OPENING_TEMPERATURE = 1.0
    TEMPERATURE_CUTOFF_MOVE = 30 # After this move, temperature becomes ~0 (play greedily)
    
