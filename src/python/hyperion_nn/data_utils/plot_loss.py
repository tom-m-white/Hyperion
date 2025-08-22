import re
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

# Use a professional and clean plot style
plt.style.use('seaborn-v0_8-whitegrid')

def parse_log_file(log_path, skip_initial=0):
    """
    Parses a training log file to extract step and loss values, with an
    option to skip the first N data points.

    Args:
        log_path (str): The path to the log file.
        skip_initial (int): The number of initial data points to skip.
    """
    log_pattern = re.compile(r"Step \[(\d+)/\d+\], Loss: ([\d\.]+)")
    steps = []
    losses = []

    print(f"--> Reading log file from: {log_path}")
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    steps.append(step)
                    losses.append(loss)
    except FileNotFoundError:
        print(f"!! WARNING: Log file not found at '{log_path}'. Skipping this model.")
        return None, None
    
    total_found = len(steps)
    if not total_found:
        print(f"!! WARNING: No data points found in '{log_path}'.")
        return None, None

    print(f"--> Found {total_found} total data points.")

    # --- NEW LOGIC TO SKIP INITIAL POINTS ---
    if skip_initial > 0:
        if total_found <= skip_initial:
            print(f"!! WARNING: Requested to skip {skip_initial} points, but only found {total_found}. "
                  "Returning no data for this model.")
            return [], [] # Return empty lists
        
        print(f"--> Skipping the first {skip_initial} data points as requested.")
        steps = steps[skip_initial:]
        losses = losses[skip_initial:]

    print(f"--> Using {len(steps)} data points for plotting.")
    return steps, losses

def plot_multiple_loss_graphs(all_data, y_scale='linear', x_scale='linear', 
                              output_file='combined_training_loss.png', smooth_window=50):
    """
    Creates and saves a single plot of loss vs. steps for multiple models.
    """
    if not all_data:
        print("!! ERROR: No data was collected from any log files. Nothing to plot.")
        return

    plt.figure(figsize=(15, 8))
    apply_smoothing = smooth_window > 1

    for model_name, (steps, losses) in all_data.items():
        if not steps or not losses:
            continue

        if apply_smoothing:
            raw_plot = plt.plot(steps, losses, alpha=0.25, linestyle='-')
            line_color = raw_plot[0].get_color()
            loss_series = pd.Series(losses)
            smoothed_loss = loss_series.rolling(window=smooth_window, min_periods=1).mean()
            plt.plot(steps, smoothed_loss, color=line_color, linewidth=2.5, label=model_name)
        else:
            plt.plot(steps, losses, label=model_name, alpha=0.8, linestyle='-')

    title = 'Combined Training Loss vs. Steps'
    if apply_smoothing:
        title += f' ({smooth_window}-Step Rolling Average)'
    
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel(f'Loss ({y_scale.capitalize()} Scale)', fontsize=14)
    
    plt.xscale(x_scale)
    plt.yscale(y_scale)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Models', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Graph saved successfully to: {output_file}")
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Plot training loss from multiple log files located in sub-directories.""",
        formatter_class=argparse.RawTextHelpFormatter
    )

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    default_log_dir = os.path.join(project_root, 'data', 'steps')
    
    parser.add_argument('--log_dir', type=str, default=default_log_dir,
                        help='Path to the parent directory containing model sub-folders.')
    parser.add_argument('--log_filename', type=str, default='step_log.txt',
                        help='The name of the log file inside each model folder.')
    parser.add_argument('--output', type=str, default='combined_training_loss.png',
                        help='Path to save the output plot image.')
    parser.add_argument('--yscale', type=str, default='linear', choices=['linear', 'log'],
                        help="Set the y-axis scale ('linear' or 'log'). Default is linear.")
    parser.add_argument('--xscale', type=str, default='linear', choices=['linear', 'log'],
                        help="Set the x-axis scale ('linear' or 'log'). Default is linear.")
    parser.add_argument('--smooth', type=int, default=50,
                        help='Window size for rolling average smoothing. Set to 1 to disable.')
    # ===> NEW ARGUMENT FOR SKIPPING INITIAL POINTS <===
    parser.add_argument('--skip_initial', type=int, default=0,
                        help='Number of initial data points to skip from each log to avoid skewing the y-axis.')
    
    args = parser.parse_args()

    all_model_data = {}
    
    if not os.path.isdir(args.log_dir):
        print(f"!! ERROR: The specified log directory does not exist: {args.log_dir}")
        exit(1)

    print(f"Scanning for model folders in: {args.log_dir}\n")
    
    for model_folder_name in sorted(os.listdir(args.log_dir)):
        model_path = os.path.join(args.log_dir, model_folder_name)
        
        if os.path.isdir(model_path):
            print(f"Processing model: {model_folder_name}")
            log_file_path = os.path.join(model_path, args.log_filename)
            
            # Pass the new argument to the parsing function
            steps, losses = parse_log_file(log_file_path, skip_initial=args.skip_initial)
            
            if steps and losses:
                all_model_data[model_folder_name] = (steps, losses)
            print("-" * 20)

    plot_multiple_loss_graphs(all_model_data, args.yscale, args.xscale, args.output, args.smooth)