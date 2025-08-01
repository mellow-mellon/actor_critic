import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# --- Define Actor and Critic Classes ---
class Actor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class Critic(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- Load Models, Data, and Preprocessor ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 12

actor = Actor(input_dim).to(device)
actor.load_state_dict(torch.load("actor_model.pt"))
actor.eval()

critic = Critic(input_dim).to(device)
critic.load_state_dict(torch.load("critic_model.pt"))
critic.eval()

preprocessor = joblib.load("preprocessor.pkl")
df = pd.read_csv("preprocessed_actor_critic_input.csv")

# Get validation set
state_cols = df.columns.difference(["action", "reward"])
X = df[state_cols].values.astype(np.float32)
y_action = df["action"].values.astype(np.float32).reshape(-1, 1)
_, X_val, _, y_action_val = train_test_split(X, y_action, test_size=0.2, random_state=42)
X_val_tensor = torch.tensor(X_val).to(device)
y_action_val_tensor = torch.tensor(y_action_val).to(device)


# --- 1. Feature Importance Analysis ---
print("--- Running Feature Importance Analysis ---")
with torch.no_grad():
    baseline_actions = actor(X_val_tensor)
    baseline_rewards = critic(torch.cat([X_val_tensor, baseline_actions], dim=1)).sum()

    importances = []

    # 💡 FIX: Reconstruct feature names manually for cross-version compatibility
    try:
        # For scikit-learn >= 1.0
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
    except AttributeError:
        # For scikit-learn < 1.0
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names()

    # The numerical feature names are known from preprocessing.py
    numerical_features = ["driver_availability"]
    feature_names = list(cat_feature_names) + numerical_features

    for i in range(len(feature_names)):
        X_permuted = X_val_tensor.clone()
        X_permuted[:, i] = X_permuted[torch.randperm(X_permuted.size(0)), i]

        permuted_actions = actor(X_permuted)
        permuted_rewards = critic(torch.cat([X_permuted, permuted_actions], dim=1)).sum()

        importance = baseline_rewards - permuted_rewards
        importances.append(importance.item())

importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
plt.title('Feature Importance for Actor Model')
plt.xlabel('Drop in Total Expected Reward (Higher is More Important)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")


# --- 2. Actor vs. Dataset Pricing Analysis ---
print("\n--- Comparing Actor vs. Dataset Pricing ---")
with torch.no_grad():
    actor_prices = actor(X_val_tensor)
    actor_rewards = critic(torch.cat([X_val_tensor, actor_prices], dim=1))
    baseline_rewards = critic(torch.cat([X_val_tensor, y_action_val_tensor], dim=1))

avg_actor_reward = actor_rewards.mean().item()
avg_baseline_reward = baseline_rewards.mean().item()
improvement_pct = ((avg_actor_reward - avg_baseline_reward) / avg_baseline_reward) * 100
print(f"Actor's Average Expected Reward: ${avg_actor_reward:.2f}")
print(f"Dataset's Average Expected Reward: ${avg_baseline_reward:.2f}")
print(f"Overall Improvement: {improvement_pct:.1f}%")

# Use the 'cat_feature_names' list we already created
cat_feature_count = len(cat_feature_names)
X_val_readable = pd.DataFrame(preprocessor.named_transformers_['cat'].inverse_transform(X_val[:, :cat_feature_count]), columns=['location', 'loyalty', 'time_of_day'])
X_val_readable['driver_availability'] = preprocessor.named_transformers_['num'].inverse_transform(X_val[:, cat_feature_count:].reshape(-1, 1)).round().astype(int)

results_df = X_val_readable
results_df['dataset_price'] = y_action_val
results_df['actor_price'] = actor_prices.cpu().numpy()
results_df['reward_from_dataset_price'] = baseline_rewards.cpu().numpy()
results_df['reward_from_actor_price'] = actor_rewards.cpu().numpy()
results_df['reward_improvement'] = results_df['reward_from_actor_price'] - results_df['reward_from_dataset_price']

top_improvements = results_df.sort_values(by='reward_improvement', ascending=False).head(5)

print("\n--- Top 5 Scenarios Where Actor Performs Better ---")
pd.set_option('display.float_format', '${:,.2f}'.format)
print(top_improvements[['location', 'loyalty', 'time_of_day', 'driver_availability', 'dataset_price', 'actor_price', 'reward_improvement']])