import os
import matplotlib.pyplot as plt
import numpy as np

# === Ensure plot directory exists ===
def ensure_plot_dir(timestep):
    folder = f"Data_plots/Plots_{timestep}"
    os.makedirs(folder, exist_ok=True)
    return folder

# === Read Collision and Possible Collision Counts ===
def read_collision_data(timestep):
    col_counts = {}
    pos_col_counts = {}

    for x in range(1, 9):
        filename = f"controllers/mavic_{x}_Supervisor/Mavic_2_PRO_{x}_collision_count_with_timestep_{timestep}.txt"
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                col_count = int(file.readline().strip().split(":")[-1])
                pos_col_count = int(file.readline().strip().split(":")[-1])
                col_counts[x] = col_count
                pos_col_counts[x] = pos_col_count
    return col_counts, pos_col_counts

# === Read Timing Data ===
def read_timing_data(timestep):
    timing_values = {}

    for x in range(1, 9):
        filename = f"controllers/mavic_{x}_Supervisor/Mavic_2_PRO_{x}_timing_with_timestep_{timestep}.txt"
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                lines = file.readlines()
                timing_values[x] = [float(value.strip()) for value in lines[2:]]
    return timing_values

# === Plot Collisions ===
def plot_collisions(col_counts, pos_col_counts, timestep, save=False):
    folder = ensure_plot_dir(timestep)
    x_labels = list(col_counts.keys())
    col_count_values = list(col_counts.values())
    pos_col_count_values = list(pos_col_counts.values())

    x = np.arange(len(x_labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, col_count_values, width, label="Συγκρούσεις που έγιναν", color='green', alpha=0.8)
    bars2 = ax.bar(x + width/2, pos_col_count_values, width, label="Πιθανές συγκρούσεις που υπολογίστηκαν", color='orange', alpha=0.8)

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, 0.1, height, ha='center', color='black', fontsize=6, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, 0.1, height, ha='center', color='black', fontsize=6, fontweight='bold')

    ax.set_xlabel("Drone ID")
    ax.set_ylabel("Συγκρούσεις")
    ax.set_title("Συγκρούσεις που έγιναν και συγκρούσεις που θα μπορούσαν να συμβούν")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save:
        plt.savefig(f"{folder}/collision_summary_timestep_{timestep}.png", dpi=300)
    else:
        plt.show()

# === Plot Timing per Step ===
def plot_timing(timing_values, timestep, save=False):
    folder = ensure_plot_dir(timestep)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    max_x = max(len(values) for values in timing_values.values()) if timing_values else 10
    y_min, y_max = 9, 12

    for i, (x, values) in enumerate(timing_values.items()):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        ax.plot(values, marker='o', markersize=3, linewidth=1, label=f"Drone {x}")
        mean_value = np.mean(values)
        ax.axhline(mean_value, color='red', linestyle='--', linewidth=1, label=f"Mean: {mean_value:.2f}")
        ax.text(len(values) * 0.98, mean_value + 0.05, f"{mean_value:.2f}", color='red', fontsize=9, ha='right')

        ax.set_title(f"Drone {x} Timing", fontsize=10)
        ax.set_xlabel("Measurement Index", fontsize=9)
        ax.set_ylabel("Timing Value (s)", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, max_x)
        ax.set_ylim(y_min, y_max)

    for i in range(len(timing_values), 8):
        row, col = divmod(i, 4)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    if save:
        plt.savefig(f"{folder}/timing_line_plot_timestep_{timestep}.png", dpi=300)
    else:
        plt.show()

# === Plot Histogram of Timing ===
def plot_timing_histograms(timing_values, timestep, save=False):
    folder = ensure_plot_dir(timestep)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)

    for i, (x, values) in enumerate(timing_values.items()):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        ax.hist(values, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        mean_val = np.mean(values)
        ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.2f}")
        ax.set_title(f"Drone {x} Timing Dist.", fontsize=10)
        ax.set_xlabel("Timing Value (s)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(9, 12)

    for i in range(len(timing_values), 8):
        row, col = divmod(i, 4)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    if save:
        plt.savefig(f"{folder}/timing_histogram_timestep_{timestep}.png", dpi=300)
    else:
        plt.show()

# === Main Function ===
def main(timestep, save_plots=True):
    col_counts, pos_col_counts = read_collision_data(timestep)
    timing_values = read_timing_data(timestep)

    plot_collisions(col_counts, pos_col_counts, timestep, save=save_plots)
    plot_timing(timing_values, timestep, save=save_plots)
    plot_timing_histograms(timing_values, timestep, save=save_plots)

# === Run Program ===
if __name__ == "__main__":
    main(timestep=500, save_plots=True)
