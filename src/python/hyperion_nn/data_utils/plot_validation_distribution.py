# python -m hyperion_nn.data_utils.plot_validation_distribution
# python -m hyperion_nn.data_utils.plot_validation_distribution --kde
# python -m hyperion_nn.data_utils.plot_validation_distribution --file data/validation/validation_results_step_30000.pt
# python -m hyperion_nn.data_utils.plot_validation_distribution --file data/validation/validation_results_step_30000.pt --kde

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import argparse
import glob
import hyperion_nn.config as config

def plot_bin_distribution(file_path):
    """
    Loads validation results and plots the distribution of outcomes
    for predicted values using discrete bins.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # --- Load Data ---
    data = torch.load(file_path)
    predictions = data['predictions'].squeeze().numpy()
    targets = data['targets'].squeeze().numpy()

    # --- Bin the Data ---
    num_bins = 20
    bins = np.linspace(-1, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    win_counts = {i: 0 for i in range(num_bins)}
    loss_counts = {i: 0 for i in range(num_bins)}
    draw_counts = {i: 0 for i in range(num_bins)}

    binned_predictions = np.digitize(predictions, bins) - 1

    # --- Count Outcomes per Bin ---
    for i in range(len(binned_predictions)):
        bin_index = binned_predictions[i]
        if 0 <= bin_index < num_bins:
            target = targets[i]
            if target == 1:
                win_counts[bin_index] += 1
            elif target == -1:
                loss_counts[bin_index] += 1
            elif target == 0:
                draw_counts[bin_index] += 1

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    win_line = [win_counts[i] for i in range(num_bins)]
    loss_line = [loss_counts[i] for i in range(num_bins)]
    draw_line = [draw_counts[i] for i in range(num_bins)]

    ax.plot(bin_centers, win_line, marker='o', linestyle='-', color='green', label='Actual Wins')
    ax.plot(bin_centers, loss_line, marker='o', linestyle='-', color='red', label='Actual Losses')
    ax.plot(bin_centers, draw_line, marker='o', linestyle='-', color='gray', label='Actual Draws')

    # --- Formatting ---
    step_number = os.path.basename(file_path).split('_')[-1].split('.')[0]
    ax.set_title(f'Value Head Calibration at Training Step {step_number}', fontsize=16)
    ax.set_xlabel('Model Predicted Value', fontsize=12)
    ax.set_ylabel('Number of Games', fontsize=12)
    ax.legend()
    ax.set_xticks(np.linspace(-1, 1, 11))

    # --- Save the Plot ---
    os.makedirs(config.PathsConfig.POST_VALIDATION_DATA_DIR, exist_ok=True)
    plot_filename = os.path.join(config.PathsConfig.POST_VALIDATION_DATA_DIR, f'value_distribution_step_{step_number}.png')
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Plot saved to {plot_filename}")

def plot_value_distribution_kde(file_path):
    """
    Loads validation results and plots a stacked KDE of outcomes
    based on the model's predicted value.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # --- Load Data ---
    data = torch.load(file_path)
    predictions = data["predictions"].squeeze().numpy()
    targets = data["targets"].squeeze().numpy()

    # --- Prepare Data for Seaborn ---
    df = pd.DataFrame({"Predicted Value": predictions, "Target": targets})
    outcome_map = {1: "Win", -1: "Loss", 0: "Draw"}
    df["Outcome"] = df["Target"].map(outcome_map)

    # --- Plotting ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.kdeplot(
        data=df,
        x="Predicted Value",
        hue="Outcome",
        multiple="fill",
        common_norm=False,
        hue_order=["Win", "Draw", "Loss"],
        palette={"Win": "green", "Loss": "red", "Draw": "gray"},
        bw_adjust=0.5,
        ax=ax,
    )

    # --- Formatting ---
    step_number = os.path.basename(file_path).split("_")[-1].split(".")[0]
    ax.set_title(f"Value Head Calibration (KDE) at Training Step {step_number}", fontsize=16)
    ax.set_xlabel("Model Predicted Value", fontsize=12)
    ax.set_ylabel("Proportion of Outcomes", fontsize=12)
    ax.set_xlim(-1, 1)
    ax.set_xticks(np.linspace(-1, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 6))
    legend = ax.get_legend()
    if legend:
        legend.set_title("Actual Outcome")
    plt.tight_layout()

    # --- Save the Plot ---
    os.makedirs(config.PathsConfig.POST_VALIDATION_DATA_DIR, exist_ok=True)
    plot_filename = os.path.join(
        config.PathsConfig.POST_VALIDATION_DATA_DIR,
        f"value_kde_step_{step_number}.png",
    )
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Plot saved to {plot_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot value head calibration from a validation results file. "
                    "By default, it plots the latest results file using a binned distribution plot."
    )
    parser.add_argument(
        '--file',
        type=str,
        help="Path to a specific validation_results_step_*.pt file. If not provided, the latest file is used."
    )
    parser.add_argument(
        '--kde',
        action='store_true',
        help="Use a Kernel Density Estimate (KDE) plot instead of the default binned plot."
    )
    args = parser.parse_args()

    file_to_plot = None
    if args.file:
        file_to_plot = args.file
    else:
        # If no file is specified, find the latest one based on the step number.
        validation_dir = config.PathsConfig.POST_VALIDATION_DATA_DIR
        list_of_files = glob.glob(os.path.join(validation_dir, 'validation_results_step_*.pt'))

        if not list_of_files:
            print(f"No validation result files found in '{validation_dir}'. Run training with validation first.")
        else:
            # Find the file with the highest step number in its name for reliability
            def get_step_from_filename(f):
                try:
                    return int(os.path.basename(f).split('_')[-1].split('.')[0])
                except (ValueError, IndexError):
                    return -1

            latest_file = max(list_of_files, key=get_step_from_filename)
            print(f"No file specified. Using latest results from: {latest_file}")
            file_to_plot = latest_file

    # If a file was found (either specified or latest), create the plot
    if file_to_plot:
        if args.kde:
            print("Generating KDE plot...")
            plot_value_distribution_kde(file_to_plot)
        else:
            print("Generating binned distribution plot...")
            plot_bin_distribution(file_to_plot)