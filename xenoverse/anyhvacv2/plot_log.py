import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np # Import numpy for NaN handling if needed

def parse_log_file(log_file_path):
    """
    Parses the log file to extract Step, Reward, energy_cost, target_cost,
    switch_cost, cool_power, and heat_power.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        pd.DataFrame: DataFrame containing the parsed data, sorted by Step.
                      Returns None if the file cannot be read or no data is found.
    """
    data = []
    # Updated Regex to capture Reward, cool_power, and heat_power
    # Made power fields optional and non-greedy to handle potential missing/truncated data
    log_pattern = re.compile(
        r"Step\s+(\d+)\s+"                   # Group 1: Step
        r".*?Reward:\s*(-?[\d.]+)"           # Group 2: Reward
        r".*?energy_cost:\s*(-?[\d.]+)"      # Group 3: energy_cost
        r".*?target_cost:\s*(-?[\d.]+)"      # Group 4: target_cost
        r".*?switch_cost:\s*(-?[\d.]+)"      # Group 5: switch_cost
        r"(?:.*?cool_power:\s*(-?[\d.]+))?"   # Group 6: cool_power (optional)
        r"(?:.*?heat_power:\s*(-?[\d.]+))?"   # Group 7: heat_power (optional)
    )

    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return None

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    try:
                        step = int(match.group(1))
                        reward = float(match.group(2))
                        energy_cost = float(match.group(3))
                        target_cost = float(match.group(4))
                        switch_cost = float(match.group(5))

                        # Handle potentially missing power values
                        cool_power_str = match.group(6)
                        heat_power_str = match.group(7)

                        cool_power = float(cool_power_str) if cool_power_str else np.nan
                        heat_power = float(heat_power_str) if heat_power_str else np.nan

                        data.append([step, reward, energy_cost, target_cost, switch_cost, cool_power, heat_power])
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping line due to parsing error: {line.strip()} - {e}")
                        continue # Skip lines with parsing errors

        if not data:
            print("Error: No matching log entries found in the file.")
            return None

        # Update columns list
        df = pd.DataFrame(data, columns=['Step', 'Reward', 'energy_cost', 'target_cost', 'switch_cost', 'cool_power', 'heat_power'])
        df = df.sort_values(by='Step').reset_index(drop=True) # Ensure data is sorted by step
        # Convert potentially object columns (due to NaN) to numeric if needed
        for col in ['cool_power', 'heat_power']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    except IOError as e:
        print(f"Error reading log file {log_file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return None


def plot_costs(df, output_filename="cost_plot.png"):
    """
    Generates and saves a line plot of the costs over steps.

    Args:
        df (pd.DataFrame): DataFrame with Step and cost columns.
        output_filename (str): Name of the file to save the plot.
    """
    if df is None or df.empty or not all(col in df.columns for col in ['Step', 'energy_cost', 'target_cost', 'switch_cost']):
        print("Cost data missing or unavailable for plotting.")
        return

    plt.figure(figsize=(15, 7))

    plt.plot(df['Step'], df['energy_cost'], label='Energy Cost', alpha=0.8)
    plt.plot(df['Step'], df['target_cost'], label='Target Cost', alpha=0.8)
    plt.plot(df['Step'], df['switch_cost'], label='Switch Cost', alpha=0.8)

    plt.xlabel("Step")
    plt.ylabel("Cost Value")
    plt.title("Cost Components Over Training Steps")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    try:
        plt.savefig(output_filename)
        print(f"Cost plot saved to {output_filename}")
        plt.close()
    except Exception as e:
        print(f"Error saving cost plot: {e}")

def plot_reward(df, output_filename="reward_plot.png"):
    """
    Generates and saves a line plot of the reward over steps.

    Args:
        df (pd.DataFrame): DataFrame with Step and Reward columns.
        output_filename (str): Name of the file to save the plot.
    """
    if df is None or df.empty or 'Reward' not in df.columns:
        print("Reward data missing or unavailable for plotting.")
        return

    plt.figure(figsize=(15, 7))

    plt.plot(df['Step'], df['Reward'], label='Reward', color='green')

    plt.xlabel("Step")
    plt.ylabel("Reward Value")
    plt.title("Reward Over Training Steps")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    try:
        plt.savefig(output_filename)
        print(f"Reward plot saved to {output_filename}")
        plt.close()
    except Exception as e:
        print(f"Error saving reward plot: {e}")

# --- New Function for Power Plot ---
def plot_powers(df, output_filename="power_plot.png"):
    """
    Generates and saves a line plot of cool_power and heat_power over steps.

    Args:
        df (pd.DataFrame): DataFrame with Step, cool_power, and heat_power columns.
        output_filename (str): Name of the file to save the plot.
    """
    required_cols = ['Step', 'cool_power', 'heat_power']
    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        print(f"Power data ({', '.join(required_cols)}) missing or unavailable for plotting.")
        return
    # Check if columns actually contain data after potential parsing issues
    if df['cool_power'].isnull().all() and df['heat_power'].isnull().all():
         print("Both cool_power and heat_power columns are empty or contain only NaN values. Skipping power plot.")
         return

    plt.figure(figsize=(15, 7))

    # Plot only if data exists for the column
    if not df['cool_power'].isnull().all():
        plt.plot(df['Step'], df['cool_power'], label='Cool Power', color='blue', alpha=0.8)
    else:
         print("Warning: No valid data found for cool_power.")

    if not df['heat_power'].isnull().all():
        plt.plot(df['Step'], df['heat_power'], label='Heat Power', color='red', alpha=0.8)
    else:
         print("Warning: No valid data found for heat_power.")


    plt.xlabel("Step")
    plt.ylabel("Power Value")
    plt.title("Power Consumption Over Training Steps")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    try:
        plt.savefig(output_filename)
        print(f"Power plot saved to {output_filename}")
        plt.close()
    except Exception as e:
        print(f"Error saving power plot: {e}")

# --- New Function for Summary Statistics ---
def calculate_and_print_summary(df, interval=10000):
    """
    Calculates and prints average statistics for specified step intervals.

    Args:
        df (pd.DataFrame): DataFrame containing the parsed data.
        interval (int): The step interval size for calculating averages.
    """
    required_cols = ['Step', 'Reward', 'cool_power', 'heat_power', 'energy_cost']
    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        print(f"Summary data ({', '.join(required_cols)}) missing or unavailable.")
        return

    max_step = df['Step'].max()
    if pd.isna(max_step):
        print("No valid steps found to calculate summary.")
        return

    print("\n--- Summary Statistics (Average per Interval) ---")
    header = f"{'Step Interval':<20} | {'Avg Reward':<15} | {'Avg Cool Power':<18} | {'Avg Heat Power':<18} | {'Avg Energy Cost':<18}"
    print(header)
    print("-" * len(header))

    summary_data = [] # Optional: store data if needed later

    for start_step in range(0, max_step, interval):
        end_step = start_step + interval
        # Filter data for the current interval (exclusive of start, inclusive of end)
        interval_df = df[(df['Step'] > start_step) & (df['Step'] <= end_step)]

        if interval_df.empty:
            # Optional: Print a line for empty intervals or just skip
            # print(f"{f'{start_step+1}-{end_step}':<20} | {'No Data':<15} | {'No Data':<18} | {'No Data':<18} | {'No Data':<18}")
            continue

        # Calculate averages - .mean() handles NaNs by default (ignores them)
        avg_reward = interval_df['Reward'].mean()
        avg_cool_power = interval_df['cool_power'].mean()
        avg_heat_power = interval_df['heat_power'].mean()
        avg_energy_cost = interval_df['energy_cost'].mean()

        # Format for printing - handle potential NaN results from mean() if all values were NaN
        row_data = {
            "interval": f"{start_step+1}-{end_step}",
            "reward": f"{avg_reward:.4f}" if not pd.isna(avg_reward) else "NaN",
            "cool": f"{avg_cool_power:.4f}" if not pd.isna(avg_cool_power) else "NaN",
            "heat": f"{avg_heat_power:.4f}" if not pd.isna(avg_heat_power) else "NaN",
            "energy": f"{avg_energy_cost:.6f}" if not pd.isna(avg_energy_cost) else "NaN"
        }
        summary_data.append(row_data)

        print(f"{row_data['interval']:<20} | {row_data['reward']:<15} | {row_data['cool']:<18} | {row_data['heat']:<18} | {row_data['energy']:<18}")

    print("-" * len(header))
    # return summary_data # Optionally return the calculated data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cost, reward, power and calculate summary from a training log file.")
    parser.add_argument("log_file", help="Path to the training log file.")
    # Updated help text for -o
    parser.add_argument("-o", "--output_base", default="plot",
                        help="Base name for the output plot files (default: 'plot'). Generates 'basename_costs.png', 'basename_reward.png', and 'basename_powers.png'.")
    parser.add_argument("-i", "--interval", type=int, default=10000,
                        help="Step interval for summary statistics calculation (default: 10000).")


    args = parser.parse_args()

    # Construct output filenames
    cost_plot_filename = f"{args.output_base}_costs.png"
    reward_plot_filename = f"{args.output_base}_reward.png"
    power_plot_filename = f"{args.output_base}_powers.png" # New plot filename

    log_df = parse_log_file(args.log_file)

    if log_df is not None:
        # --- Calculate and Print Summary First ---
        calculate_and_print_summary(log_df, interval=args.interval)

        # --- Generate Plots ---
        plot_costs(log_df, cost_plot_filename)
        plot_reward(log_df, reward_plot_filename)
        plot_powers(log_df, power_plot_filename) # Call the new plot function
    else:
        print("Failed to parse log file. No summary or plots generated.")