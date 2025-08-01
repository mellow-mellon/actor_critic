import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("synthetic_uber_pricing_with_airport.csv")

print("\n🔍 Shape:", df.shape)
print("\n🧾 Columns:", df.columns.tolist())
print("\n📊 Preview:")
print(df.head())

# Check for missing values
print("\n🧼 Missing values:")
print(df.isnull().sum())

# Check data types
print("\n🔢 Data types:")
print(df.dtypes)

# Check value ranges and distributions
print("\n📈 Summary statistics:")
print(df.describe(include='all'))

# Check unique values for categorical columns
categorical = ["location", "loyalty", "time_of_day"]
for col in categorical:
    print(f"\n🔎 Unique values in '{col}':", df[col].unique())

# Check reward logic (quick sanity check)
accepted_mean = df[df["accepted"] == 1]["reward"].mean()
rejected_mean = df[df["accepted"] == 0]["reward"].mean()
print(f"\n💰 Avg reward (accepted): {accepted_mean:.2f}")
print(f"💸 Avg reward (rejected): {rejected_mean:.2f}")

# Reward distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["reward"], bins=50, kde=True)
plt.title("Reward Distribution")
plt.xlabel("Reward")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Offered price vs WTP
plt.figure(figsize=(8, 5))
sns.scatterplot(x="offered_price", y="wtp", hue="accepted", data=df, alpha=0.5)
plt.title("Offered Price vs WTP with Acceptance")
plt.xlabel("Offered Price")
plt.ylabel("Willingness to Pay")
plt.tight_layout()
plt.show()

# Boxplots of reward
plt.figure(figsize=(8, 5))
sns.boxplot(x="location", y="reward", data=df)
plt.title("Reward by Location")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="loyalty", y="reward", data=df)
plt.title("Reward by Loyalty")
plt.tight_layout()
plt.show()

# ✅ Acceptance rate by location
plt.figure(figsize=(8, 5))
location_acceptance = df.groupby("location")["accepted"].mean().reset_index()
sns.barplot(data=location_acceptance, x="location", y="accepted", palette="Blues_d")
plt.title("Acceptance Rate by Location")
plt.ylabel("Acceptance Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ✅ Acceptance rate by loyalty
plt.figure(figsize=(8, 5))
loyalty_acceptance = df.groupby("loyalty")["accepted"].mean().reset_index()
sns.barplot(data=loyalty_acceptance, x="loyalty", y="accepted", palette="Oranges_d")
plt.title("Acceptance Rate by Loyalty Level")
plt.ylabel("Acceptance Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 1. Distribution of driver availability
plt.figure(figsize=(8, 5))
sns.histplot(df["driver_availability"], bins=30, kde=False)
plt.title("Distribution of Driver Availability")
plt.xlabel("Driver Availability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Acceptance rate vs driver availability
plt.figure(figsize=(8, 5))
acceptance_by_driver = df.groupby("driver_availability")["accepted"].mean().reset_index()
sns.lineplot(data=acceptance_by_driver, x="driver_availability", y="accepted")
plt.title("Acceptance Rate vs Driver Availability")
plt.xlabel("Driver Availability")
plt.ylabel("Acceptance Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 3. Reward by driver availability buckets
df["driver_bin"] = pd.cut(df["driver_availability"], bins=[-1, 2, 5, 10, 20, 50, 100], labels=["0-2", "3-5", "6-10", "11-20", "21-50", "51+"])
plt.figure(figsize=(8, 5))
sns.boxplot(x="driver_bin", y="reward", data=df)
plt.title("Reward by Driver Availability Buckets")
plt.xlabel("Driver Availability Range")
plt.tight_layout()
plt.show()

# 4. Offered price vs driver availability (colored by accepted)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="driver_availability", y="offered_price", hue="accepted", alpha=0.5)
plt.title("Offered Price vs Driver Availability (by Acceptance)")
plt.xlabel("Driver Availability")
plt.ylabel("Offered Price")
plt.tight_layout()
plt.show()
