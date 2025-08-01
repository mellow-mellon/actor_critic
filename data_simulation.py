
"""
Synthetic dataset for Uber pricing

"location": location,                      # 🗺️ Pickup location
"loyalty": loyalty,                        # 💳 Loyalty level
"time_of_day": time,                       # 🕒 Time of booking
"driver_availability": drivers,            # 🚗 Active drivers nearby
"offered_price": price,                    # 💵 Price set by agent
"wtp": wtp,                                # 🧠 Customer's private WTP
"prob_accept": prob_accept,                # 📈 Modeled purchase probability
"accepted": accepted,                      # ✅ Whether customer accepted
"reward": reward                           # 💰 Revenue if accepted
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
NUM_EPISODES = 10000
MAX_STEPS = 20
PRICE_RANGE = (5, 50)

# Updated location list including "airport"
locations = ["urban", "suburban", "rural", "airport"]
loyalty_levels = ["regular", "silver", "gold"]
times_of_day = ["morning", "afternoon", "evening", "night"]

def sample_location():
    return np.random.choice(locations)

def sample_loyalty():
    return np.random.choice(loyalty_levels, p=[0.5, 0.3, 0.2])

def sample_time_of_day():
    return np.random.choice(times_of_day)

def sample_driver_availability(location, time):
    base_map = {
        "urban": 20,
        "suburban": 10,
        "rural": 5,
        "airport": 15  # airport has moderate driver availability
    }
    time_factor = {
        "morning": 0.8,
        "afternoon": 1.0,
        "evening": 0.6,
        "night": 0.4
    }
    return int(np.random.poisson(base_map[location] * time_factor[time]))

def sample_wtp(location, loyalty):
    base = 20
    loc_factor = {
        "urban": 1.2,
        "suburban": 1.0,
        "rural": 0.9,
        "airport": 1.5  # customers at airports are willing to pay more
    }
    loyalty_factor = {
        "regular": 1.0,
        "silver": 1.1,
        "gold": 1.3
    }
    noise = np.random.normal(0, 3)
    return base * loc_factor[location] * loyalty_factor[loyalty] + noise

def p_accept(price, wtp):
    return 1 / (1 + np.exp((price - wtp) / 2))  # sigmoid function

# Generate synthetic dataset
data = []

for _ in range(NUM_EPISODES):
    for _ in range(MAX_STEPS):
        location = sample_location()
        loyalty = sample_loyalty()
        time = sample_time_of_day()
        drivers = sample_driver_availability(location, time)
        wtp = sample_wtp(location, loyalty)

        price = np.round(np.random.uniform(*PRICE_RANGE), 2)
        prob_accept = p_accept(price, wtp)
        accepted = int(np.random.rand() < prob_accept)
        driver_cost = 0.7 * price
        if accepted:
            reward = price - driver_cost
        else:
            alpha = 0.01   # penalty strength
            beta = 0.001   # penalize more if more drivers are available
            reward = -alpha * (price + beta * drivers)
        data.append({
            "location": location,
            "loyalty": loyalty,
            "time_of_day": time,
            "driver_availability": drivers,
            "offered_price": price,
            "wtp": wtp,
            "prob_accept": prob_accept,
            "accepted": accepted,
            "reward": reward
        })

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv("synthetic_uber_pricing_with_airport.csv", index=False)
print("✅ Dataset saved as 'synthetic_uber_pricing_with_airport.csv'")
