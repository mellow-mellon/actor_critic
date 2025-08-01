import torch
import torch.nn as nn
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load preprocessor
preprocessor = joblib.load("preprocessor.pkl")

# Define Actor and Critic classes
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
            nn.Linear(input_dim + 1, 64),  # +1 to include action input
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Function to load a model
def load_model(model_path, model_class, input_dim):
    model = model_class(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load dataset
df = pd.read_csv("preprocessed_actor_critic_input.csv")

# Columns
state_cols = df.columns.difference(["action", "reward"])
X = df[state_cols].values.astype(np.float32)
y_action = df["action"].values.astype(np.float32)

# Train/test split
_, X_val, _, y_action_val = train_test_split(
    X, y_action, test_size=0.2, random_state=42
)

# Load models
input_dim = X_val.shape[1]
actor = load_model("actor_model.pt", Actor, input_dim)
critic = load_model("critic_model.pt", Critic, input_dim)

# Convert to tensors
X_val_tensor = torch.FloatTensor(X_val)
y_action_tensor = torch.FloatTensor(y_action_val).unsqueeze(1)

# Predict actor actions
with torch.no_grad():
    predicted_actions = actor(X_val_tensor).squeeze().unsqueeze(1)  # (N, 1)

# Concatenate state and action for critic input
critic_input_pred = torch.cat([X_val_tensor, predicted_actions], dim=1)
critic_input_actual = torch.cat([X_val_tensor, y_action_tensor], dim=1)

# Get critic rewards
with torch.no_grad():
    critic_rewards_predicted = critic(critic_input_pred).squeeze().numpy()
    critic_rewards_actual = critic(critic_input_actual).squeeze().numpy()

# Sort by critic predicted rewards from actor predictions
sorted_idx = np.argsort(critic_rewards_predicted)

# Limit visualization to first 200 samples
plot_limit = 200

# Plotting
plt.figure(figsize=(14, 8))

# Predicted reward (Critic) for actor actions
plt.plot(critic_rewards_predicted[sorted_idx][:plot_limit], label="Critic Reward (Actor Prices)", linewidth=2, color='green')

# Predicted reward (Critic) for actual test actions (baseline)
plt.plot(critic_rewards_actual[sorted_idx][:plot_limit], label="Critic Reward (Test Prices)", linestyle='--', linewidth=2, color='red')

plt.xlabel('Sorted Sample Index')
plt.ylabel('Critic Predicted Reward')
plt.title('Critic Rewards: Actor Predicted Prices vs Test Prices (First 200 Samples)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
