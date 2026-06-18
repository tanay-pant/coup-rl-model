import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_path="training_log.csv", save_path="training_metrics.png"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Coup RL Training Metrics', fontsize=16)

    # Plot Mean Reward
    axs[0].plot(df['Iteration'], df['Mean Reward'], color='green')
    axs[0].set_ylabel('Mean Reward')
    axs[0].set_title('Mean Reward per Iteration')
    axs[0].grid(True)

    # Plot Policy Loss
    axs[1].plot(df['Iteration'], df['Policy Loss'], color='blue')
    axs[1].set_ylabel('Policy Loss')
    axs[1].set_title('Policy Loss (Action Probability Shift)')
    axs[1].grid(True)

    # Plot Value Loss
    axs[2].plot(df['Iteration'], df['Value Loss'], color='orange')
    axs[2].set_ylabel('Value Loss')
    axs[2].set_title('Value Function Loss (Reward Prediction Error)')
    axs[2].grid(True)

    # Plot Entropy
    axs[3].plot(df['Iteration'], df['Entropy'], color='purple')
    axs[3].set_ylabel('Entropy')
    axs[3].set_xlabel('Iteration')
    axs[3].set_title('Entropy (Randomness/Exploration)')
    axs[3].grid(True)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    plot_metrics()
