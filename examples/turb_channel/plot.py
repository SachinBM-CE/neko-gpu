import re
import matplotlib.pyplot as plt

def parse_and_plot_log(log_file_path):
    """
    Parses a log file to extract 'COMPUTE' lines and plots tau_new vs. tau_old
    on a twin-axis chart.
    
    Args:
        log_file_path (str): The path to the log file.
    """
    
    # Regex to find lines starting with 'COMPUTE:' and extract the data
    # It captures three groups:
    # 1. (\d+): The timestep (one or more digits)
    # 2. ([0-9.Ee+-]+): The tau_new value (a float, possibly in scientific notation)
    # 3. ([0-9.Ee+-]+): The tau_old value (a float, possibly in scientific notation)
    compute_regex = re.compile(
        r"^\s*COMPUTE:\s*(\d+),"
        r"\s*tau_new_l_d\[1\]=([0-9.Ee+-]+),"
        r"\s*tau_old_l_d\[1\]=([0-9.Ee+-]+)"
    )
    
    # Lists to store the extracted data
    timesteps = []
    tau_new_values = []
    tau_old_values = []
    
    print(f"Opening and parsing log file: {log_file_path}...")
    
    try:
        with open(log_file_path, 'r') as f:
            for line_number, line in enumerate(f):
                match = compute_regex.search(line)
                if match:
                    try:
                        # Extract matched groups and convert to the correct type
                        timestep = int(match.group(1))
                        tau_new = float(match.group(2))
                        tau_old = float(match.group(3))
                        
                        # Add the data to our lists
                        timesteps.append(timestep)
                        tau_new_values.append(tau_new)
                        tau_old_values.append(tau_old)
                    except ValueError as e:
                        print(f"Warning: Could not parse values on line {line_number + 1}: {e}")
                        
    except FileNotFoundError:
        print(f"Error: Log file not found at path: {log_file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Check if we found any data before trying to plot
    if not timesteps:
        print("No 'COMPUTE' data matching the pattern was found in the file.")
        return

    print(f"Successfully parsed {len(timesteps)} data points.")
    print("Generating plot...")

    # --- Plotting ---
    
    # Create a figure and the first (left) y-axis
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot tau_new on the left axis (ax1)
    color1 = 'tab:blue'
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('tau_new', color=color1, fontsize=12)
    line1 = ax1.plot(timesteps, tau_new_values, color=color1, label='tau_new (left axis)')
    ax1.tick_params(axis='y', labelcolor=color1)
    # Use scientific notation for the y-axis if values are very small
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Create the second (right) y-axis, sharing the x-axis
    ax2 = ax1.twinx()  
    
    # Plot tau_old on the right axis (ax2)
    color2 = 'tab:red'
    ax2.set_ylabel('tau_old', color=color2, fontsize=12)
    line2 = ax2.plot(timesteps, tau_old_values, color=color2, linestyle='--', label='tau_old (right axis)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # --- Final Touches ---
    
    # Add a title
    plt.title('Tau New vs. Tau Old over Timesteps', fontsize=16)
    
    # Add a unified legend
    # We get the 'handles' (the lines) and 'labels' from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    # Add a grid for readability
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Ensure the plot layout is clean
    fig.tight_layout()  
    
    # Display the plot
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    # --- !!! IMPORTANT !!! ---
    # --- Change this path to point to your actual log file ---
    log_file_to_parse = 'log_rlwm_spald.txt'
    # -------------------------
    
    parse_and_plot_log(log_file_to_parse)
