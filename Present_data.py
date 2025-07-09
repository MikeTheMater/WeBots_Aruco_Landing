import os
import matplotlib.pyplot as plt
import numpy as np

# === Ensure plot directory exists ===
def ensure_plot_dir(timestep, normal):
    folder = f"Data_plots/Plots_timestep_{timestep}_normal_{normal}"
    os.makedirs(folder, exist_ok=True)
    return folder

# === Read Collision and Possible Collision Counts ===
def read_collision_data(timestep, normal, num_drones=8):
    col_counts = {}
    pos_col_counts = {}
    mean_col_count = 0
    mean_pos_col_count = 0

    for x in range(1, num_drones + 1):
        filename = f"controllers/mavic_{x}_Supervisor/Mavic_2_PRO_{x}_collision_count_with_timestep_{timestep}_and_normal_{normal}_No_of_drones_{num_drones}.txt"
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                col_count = int(file.readline().strip().split(":")[-1])
                pos_col_count = int(file.readline().strip().split(":")[-1])
                col_counts[x] = col_count
                pos_col_counts[x] = pos_col_count
                mean_col_count += col_count
                mean_pos_col_count += pos_col_count

    count = len(col_counts) if col_counts else 1
    mean_col_count /= count
    mean_pos_col_count /= count
    return col_counts, pos_col_counts, mean_col_count, mean_pos_col_count

def read_timing_data(timestep, normal, num_drones=8):
    timing_values = {}
    for x in range(1, num_drones + 1):
        filename = f"controllers/mavic_{x}_Supervisor/Mavic_2_PRO_{x}_timing_with_timestep_{timestep}_and_normal_{normal}_No_of_drones_{num_drones}.txt"
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                lines = file.readlines()[2:]  # Skip the first 2 lines: header + bad value
                values = []
                for line in lines:
                    try:
                        values.append(float(line.strip()))
                    except ValueError:
                        continue
                if values:
                    timing_values[x] = values
    return timing_values


# === Plot Collisions ===
def plot_collisions(col_counts, pos_col_counts, mean_col_count, mean_pos_col_count, timestep, normal, save=False):
    folder = ensure_plot_dir(timestep, normal)
    x_labels = [f"Drone {k}" for k in col_counts.keys()]
    col_count_values = list(col_counts.values())
    pos_col_count_values = list(pos_col_counts.values())

    x = np.arange(len(x_labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, col_count_values, width, label="Συγκρούσεις", color='green', alpha=0.8)
    bars2 = ax.bar(x + width/2, pos_col_count_values, width, label="Πιθανές Συγκρούσεις", color='orange', alpha=0.8)

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, f"{height}", ha='center', fontsize=7, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, f"{height}", ha='center', fontsize=7, fontweight='bold')

    ax.axhline(mean_col_count, color='red', linestyle='--', linewidth=1.5, label=f"Μ.Ο. Συγκρούσεων: {mean_col_count:.2f}")
    ax.axhline(mean_pos_col_count, color='blue', linestyle='--', linewidth=1.5, label=f"Μ.Ο. Πιθανών: {mean_pos_col_count:.2f}")

    ax.set_xlabel("Drone ID")
    ax.set_ylabel("Συγκρούσεις")
    ax.set_title(f"Συγκρούσεις & Πιθανές Συγκρούσεις (timestep={timestep}, normal={normal})")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save:
        plt.savefig(f"{folder}/collision_summary.png", dpi=300)
    else:
        plt.show()

def plot_timing(timing_values, timestep, normal, save=False):

    folder = ensure_plot_dir(timestep, normal)
    num_drones = len(timing_values)
    rows = (num_drones + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(14, rows * 3), sharex=True)
    axes = axes.flatten()

    # Gather global deviation range
    all_devs = []
    for values in timing_values.values():
        mean_val = np.mean(values)
        all_devs.extend([v - mean_val for v in values])
    y_min = min(values) - 0.05
    y_max = max(values) + 0.05

    for i, (x, values) in enumerate(timing_values.items()):
        ax = axes[i]
        mean_value = np.mean(values)
        
        ax.plot(values, marker='o', markersize=1.5, linewidth=0.5, label=f"Drone {x}")
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1, label="Μ.Ο.: {:.2f}".format(mean_value))
        ax.set_title(f"Drone {x} Timing", fontsize=10)
        ax.set_xlabel("Measurement Index", fontsize=9)
        ax.set_ylabel("Απόκλιση Χρόνου (s)", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, len(values))
        ax.set_ylim(0, y_max)

    for j in range(len(timing_values), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save:
        plt.savefig(f"{folder}/timing_line_plot.png", dpi=300)
    else:
        plt.show()

def plot_timing_histograms(timing_values, timestep, normal, save=False):
    

    folder = ensure_plot_dir(timestep, normal)
    num_drones = len(timing_values)
    rows = (num_drones + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(14, rows * 3), sharex=True)
    axes = axes.flatten()

    for i, (x, values) in enumerate(timing_values.items()):
        ax = axes[i]
        mean_val = np.mean(values)
        deviations = [v - mean_val for v in values]
        ax.hist(values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', label=f"Μ.Ο.: 0.00")
        ax.set_title(f"Drone {x} Histogram", fontsize=10)
        ax.set_xlabel("Απόκλιση Χρόνου (s)", fontsize=9)
        ax.set_ylabel("Συχνότητα", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(len(timing_values), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save:
        plt.savefig(f"{folder}/timing_histogram.png", dpi=300)
    else:
        plt.show()


# === Main Function ===
def main(timesteps, normals, save_plots=True, num_drones=8):
    for timestep in timesteps:
        for normal in normals:
            print(f"\n--- Processing Timestep={timestep}, Normal={normal} ---")
            col_counts, pos_col_counts, mean_col_count, mean_pos_col_count = read_collision_data(timestep, normal, num_drones)
            timing_values = read_timing_data(timestep, normal, num_drones)

            if not col_counts and not timing_values:
                print(f"No data found for timestep={timestep}, normal={normal}, skipping.")
                continue

            plot_collisions(col_counts, pos_col_counts, mean_col_count, mean_pos_col_count, timestep, normal, save=save_plots)
            plot_timing(timing_values, timestep, normal, save=save_plots)
            plot_timing_histograms(timing_values, timestep, normal, save=save_plots)

# === Run Program ===
if __name__ == "__main__":
    timesteps = [50, 100, 250, 500, 1000]               # Customize as needed
    normals = [0.125, 0.25, 0.5, 1]           # Customize as needed
    main(timesteps=timesteps, normals=normals, save_plots=True, num_drones=8)