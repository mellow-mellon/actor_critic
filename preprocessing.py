import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
df = pd.read_csv("synthetic_uber_pricing_with_airport.csv")

# Define input and target features
categorical_features = ["location", "loyalty", "time_of_day"]
numerical_features = ["driver_availability"]
df["action"] = df["offered_price"]
df["reward"] = df["offered_price"] * df["accepted"]  # or your own reward logic

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(sparse=False), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

# Fit and transform
X = preprocessor.fit_transform(df[categorical_features + numerical_features])

# Handle sklearn version differences for feature names
try:
    # sklearn >= 1.0
    cat_columns = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features).tolist()
except AttributeError:
    # sklearn < 1.0
    cat_columns = preprocessor.named_transformers_["cat"].get_feature_names(categorical_features).tolist()

feature_names = cat_columns + numerical_features

# Build final DataFrame
processed_df = pd.DataFrame(X, columns=feature_names)
processed_df["action"] = df["action"]
processed_df["reward"] = df["reward"]

# Save or return for modeling
processed_df.to_csv("preprocessed_actor_critic_input.csv", index=False)
print("✅ Preprocessing complete. Saved to 'preprocessed_actor_critic_input.csv'")
joblib.dump(preprocessor, "preprocessor.pkl")
print("✅ Preprocessor saved as 'preprocessor.pkl'")