import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('training_lstm_advanced_log.csv')

# Print summary stats
print(f"Total iterations: {len(df)}")
print(f"Starting Mean Reward: {df['Mean Reward'].iloc[:10].mean():.4f}")
print(f"Recent Mean Reward: {df['Mean Reward'].iloc[-100:].mean():.4f}")
print(f"Max Mean Reward: {df['Mean Reward'].max():.4f} at Iteration {df.loc[df['Mean Reward'].idxmax(), 'Iteration']}")

# Smooth the curves
df['Smoothed_Reward'] = df['Mean Reward'].rolling(window=100, min_periods=1).mean()
df['Smoothed_Policy_Loss'] = df['Policy Loss'].rolling(window=100, min_periods=1).mean()
df['Smoothed_VF_Loss'] = df['Value Loss'].rolling(window=100, min_periods=1).mean()

# Plot
plt.figure(figsize=(15, 10))

# Reward
plt.subplot(3, 1, 1)
plt.plot(df['Iteration'], df['Mean Reward'], alpha=0.3, color='blue', label='Raw')
plt.plot(df['Iteration'], df['Smoothed_Reward'], color='darkblue', linewidth=2, label='100-Iter MA')
plt.title('Mean Reward over Iterations')
plt.ylabel('Reward')
plt.grid(True, alpha=0.3)
plt.legend()

# Losses
plt.subplot(3, 1, 2)
plt.plot(df['Iteration'], df['Smoothed_Policy_Loss'], color='red', label='Policy Loss')
plt.plot(df['Iteration'], df['Smoothed_VF_Loss'], color='green', label='Value Loss')
plt.title('Losses (Smoothed)')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

# Entropy
plt.subplot(3, 1, 3)
plt.plot(df['Iteration'], df['Entropy'], color='purple')
plt.title('Entropy over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/training_progress_36k.png', dpi=150)
print("Plot saved to plots/training_progress_36k.png")
