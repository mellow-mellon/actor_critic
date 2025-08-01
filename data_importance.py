import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# --- Load Data and Preprocessor ---
preprocessor = joblib.load("preprocessor.pkl")
df = pd.read_csv("preprocessed_actor_critic_input.csv")

# --- Prepare Data for Random Forest ---
# X includes all state features, y is the reward
X = df.drop(columns=["action", "reward"])
y = df["reward"]

# --- Train Model and Get Feature Importance ---
# We use a simple, interpretable model to find inherent importance
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Get feature names from the preprocessor
feature_names = X.columns.tolist()
importances = model.feature_importances_

# --- Create and Save Plot ---
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x='importance', y='feature', data=importance_df, palette='plasma')
plt.title('Feature Importance for Predicting Reward (from Original Data)')
plt.xlabel('Predictive Power (Higher is More Important)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('data_importance.png')

print("✅ Data feature importance plot saved as 'data_importance.png'")