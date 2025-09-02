#--
# File: src/python/hyperion_nn/export.py
#--
# This script provides the functionality to convert a PyTorch training checkpoint 
# into a TorchScript model. The resulting TorchScript file (.pt) is a serialized, 
# self-contained model that can be loaded and executed in a non-Python environment,
# most notably in C++ using the LibTorch library. This is a crucial step for 
# deploying the model in a high-performance production environment for hyperion.
#
# ---
#
# How to use this:
#
#   python3 src/python/hyperion_nn/export.py [path_to_checkpoint] [path_for_output_model]
#
# For example:
#       
#   python3 src/python/hyperion_nn/export.py data/models/16B-196F/checkpoint_step_200000_16B-196F.pt data/completed_models/hyperion_16b_196f.pt
#

# Location: src/python/hyperion_nn/export.py

import torch
import argparse
import os
from hyperion_nn.models.resnet_cnn import HyperionNN
from hyperion_nn import config

#--
# export_model
#--
# Loads a model from a training checkpoint, cleans its state dictionary,
# and exports it to a production-ready TorchScript format.
#
# TorchScript tracing works by executing the model with a sample input and 
# recording the operations performed. This creates a graph that can be optimized
# and run independently of the Python source code.
#
# Args:
#   checkpoint_path (str): The file path to the input training checkpoint (.pt file).
#   output_filename (str): The file path where the final TorchScript model will be saved.
#
def export_model(checkpoint_path: str, output_filename: str):
    """
    Loads a model from a training checkpoint and exports it to a 
    TorchScript format for use in C++ (LibTorch).
    """
    print(f"Loading model architecture using settings from config.py...")
    
    #--
    # Step 1: Instantiate the model architecture.
    #--
    # This creates an "empty shell" of the neural network, with the same layers and
    # structure as the one used during training, but with randomly initialized weights.
    # The architecture is defined in the HyperionNN class.
    model = HyperionNN()
    
    print(f"  > Model created with {config.ModelConfig.NUM_RESIDUAL_BLOCKS} blocks and {config.ModelConfig.NUM_FILTERS} filters.")

    #--
    # Step 2: Load the trained weights from the checkpoint file.
    #--
    # Checkpoints often contain more than just weights (e.g., optimizer state, epoch).
    # - `map_location='cpu'`: Ensures the model loads on any machine, even without a GPU.
    # - `weights_only=True`: A security measure to prevent loading arbitrary pickled code.
    print(f"Loading weights from checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
    
    #--
    # Step 2a: Extract the model's state dictionary if the checkpoint is a dictionary.
    #--
    # Training frameworks (like PyTorch Lightning) often save checkpoints as a dict
    # with keys like 'model_state_dict', 'optimizer_state_dict', etc. We only need
    # the model's weights and biases for inference.
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    #--
    # Step 2b: Clean the keys of the state dictionary.
    #--
    # During training, tools like `torch.compile` can add a prefix (e.g., '_orig_mod.')
    # to the names of the model's layers in the state dictionary. The base `HyperionNN`
    # model object does not have this prefix. This line removes it to ensure the keys
    # match perfectly when we load the weights into our model instance.
    new_state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
    print("Cleaned state_dict keys by removing '_orig_mod.' prefix.")

    #--
    # Step 3: Load the cleaned state dictionary into the model.
    #--
    # This populates the "empty shell" created in Step 1 with the trained weights
    # from the checkpoint file.
    model.load_state_dict(new_state_dict)

    #--
    # Step 4: Set the model to evaluation mode.
    #--
    # This is a critical step. It disables layers that behave differently during
    # training vs. inference, such as Dropout (which would be turned off) and
    # BatchNorm (which would use running averages instead of batch statistics).
    model.eval()
    print("Model set to evaluation mode.")

    #--
    # Step 5: Define a dummy input with the correct shape and type.
    #--
    # TorchScript's `trace` method requires a sample input to run through the model.
    # The shape must match what the model expects:
    # (batch_size, num_channels, height, width).
    # Here, we use a batch size of 1, 20, 8, 8.
    dummy_input = torch.randn(1, config.ModelConfig.NUM_INPUT_PLANES, 8, 8, dtype=torch.float32)
    print(f"Creating a dummy input with shape: {dummy_input.shape}")

    #--
    # Step 6: Trace the model to create the TorchScript object.
    #--
    # This is the core of the export process. PyTorch executes the model with the
    # dummy input, records all the operations, and builds a static graph. This
    # graph is the TorchScript module.
    print("Tracing the model...")
    traced_script_module = torch.jit.trace(model, dummy_input)

    #--
    # Step 7: Save the traced model to the specified output file.
    #--
    # First, ensure the destination directory exists. Then, serialize the traced
    # module to disk. This .pt file is now ready to be loaded by LibTorch in C++.
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    traced_script_module.save(output_filename)
    print(f"Successfully exported TorchScript model to: {output_filename}")


#--
# Main Execution Block
#--
# This code runs only when the script is executed directly from the command line.
# It sets up an argument parser to handle command-line inputs for the checkpoint
# path and the desired output file path.
if __name__ == '__main__':
    #--
    # Initialize the argument parser with a description.
    #--
    parser = argparse.ArgumentParser(description="Export a PyTorch model checkpoint to TorchScript for C++.")
    
    #--
    # Define the required command-line arguments.
    #--
    # 'checkpoint': The path to the source .pt file from a training run.
    parser.add_argument("checkpoint", type=str, help="Path to the input model checkpoint .pt file.")
    # 'output': The path where the final, deployable TorchScript model will be saved.
    parser.add_argument("output", type=str, help="Path to save the FINAL output TorchScript .pt file.")
    
    #--
    # Parse the arguments and run the export function.
    #--
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)