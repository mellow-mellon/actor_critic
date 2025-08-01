import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend for interactive plotting
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# --- (Data loading and model definitions remain the same) ---
# Load preprocessed dataset
df = pd.read_csv("preprocessed_actor_critic_input.csv")

# Split into state, action, reward
state_cols = df.columns.difference(["action", "reward"])
X = df[state_cols].values.astype(np.float32)
y_action = df["action"].values.astype(np.float32).reshape(-1, 1)
y_reward = df["reward"].values.astype(np.float32).reshape(-1, 1)

# Train/val split
X_train, X_val, y_action_train, y_action_val, y_reward_train, y_reward_val = train_test_split(
    X, y_action, y_reward, test_size=0.2, random_state=42
)

# Convert to PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train).to(device)
y_action_train = torch.tensor(y_action_train).to(device)
y_reward_train = torch.tensor(y_reward_train).to(device)
X_val = torch.tensor(X_val).to(device)
y_action_val = torch.tensor(y_action_val).to(device)
y_reward_val = torch.tensor(y_reward_val).to(device)


# Define Actor and Critic
class Actor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# Instantiate models, optimizers, and loss
input_dim = X_train.shape[1]
actor = Actor(input_dim).to(device)
critic = Critic(input_dim).to(device)
actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
critic_opt = optim.Adam(critic.parameters(), lr=1e-4)
mse = nn.MSELoss()


# --- 💡 NEW: Setup for Dual-Axis Live Plotting ---
plt.ion() # Turn on interactive mode
fig, ax1 = plt.subplots(figsize=(12, 7))

# Setup for first y-axis (Critic Loss - MSE)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Critic Loss (MSE)", color='r')
ax1.tick_params(axis='y', labelcolor='r')
train_loss_line, = ax1.plot([], [], 'r-', label="Training Critic Loss")
val_loss_line, = ax1.plot([], [], 'r--', label="Validation Critic Loss")

# Setup for second y-axis (Actor Loss)
ax2 = ax1.twinx() # Create a second y-axis that shares the same x-axis
ax2.set_ylabel("Actor Loss", color='g')
ax2.tick_params(axis='y', labelcolor='g')
actor_loss_line, = ax2.plot([], [], 'g-.', label="Training Actor Loss")

fig.suptitle("Live Training Progress")
fig.tight_layout()

# Combine legends from both axes
lines = [train_loss_line, val_loss_line, actor_loss_line]
ax1.legend(lines, [l.get_label() for l in lines])

# Lists to store loss values
epochs_list, actor_losses, train_critic_losses, val_critic_losses = [], [], [], []
# ---

# Training parameters
log_frequency = 100
epochs = 15000

# Training loop
for epoch in range(epochs):
    actor.train()
    critic.train()

    # --- (Training steps for actor and critic) ---
    price_pred = actor(X_train)
    critic_input_true = torch.cat([X_train, y_action_train], dim=1)
    value_true = critic(critic_input_true)
    critic_loss = mse(value_true, y_reward_train)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    
    critic_input_pred = torch.cat([X_train, price_pred], dim=1)
    value_pred = critic(critic_input_pred)
    advantage = y_reward_train - value_pred.detach()
    actor_loss = -torch.mean(advantage * price_pred)
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    # ---

    # --- 💡 NEW: Update plotting logic ---
    if (epoch + 1) % log_frequency == 0:
        actor.eval()
        critic.eval()
        with torch.no_grad():
            critic_val_input = torch.cat([X_val, y_action_val], dim=1)
            val_reward_pred = critic(critic_val_input)
            val_critic_loss = mse(val_reward_pred, y_reward_val)

        print(
            f"Epoch {epoch+1}: Actor Loss = {actor_loss.item():.4f}, "
            f"Train Critic Loss = {critic_loss.item():.4f}, "
            f"Val Critic Loss = {val_critic_loss.item():.4f}"
        )

        # 1. Append new data
        epochs_list.append(epoch + 1)
        actor_losses.append(actor_loss.item())
        train_critic_losses.append(critic_loss.item())
        val_critic_losses.append(val_critic_loss.item())

        # 2. Update plot data
        actor_loss_line.set_data(epochs_list, actor_losses)
        train_loss_line.set_data(epochs_list, train_critic_losses)
        val_loss_line.set_data(epochs_list, val_critic_losses)

        # 3. Rescale and redraw the canvas
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

# ---
plt.ioff()
print("✅ Model training complete!")