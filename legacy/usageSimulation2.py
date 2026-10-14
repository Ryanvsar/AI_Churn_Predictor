import random
from datetime import datetime, timedelta
import math

"""
------------------------------------------------------
Author: Ryan Vandersar
Date (DD-MM-YYYY): 26-12-2025
------------------------------------------------------
Program Title: Churn Prediction AI
------------------------------------------------------
Description:
Advanced Churn Simulation Engine
------------------------------------------------------
"""

# ---------------- CONFIG ----------------
NUM_USERS = 5000
TOTAL_DAYS = 120

OBSERVATION_DAYS = 80      
PREDICTION_DAYS = 40       

TARGET_CHURNED = 400
TARGET_AT_RISK = 1000

START_DATE = datetime(2025, 1, 1)

# ---------------- HELPERS ----------------
def sigmoid(x, k=0.05):
    return 1 / (1 + math.exp(k * x))


# ---------------- MAIN ----------------
def generate_users():
    user_ids = list(range(1, NUM_USERS + 1))

    churned_users = set(random.sample(user_ids, TARGET_CHURNED))
    remaining = list(set(user_ids) - churned_users)
    at_risk_users = set(random.sample(remaining, TARGET_AT_RISK))

    Users = {}

    for user_id in user_ids:
        sessions = []
        session_counter = 1

        # ---------- Baseline profile ----------
        base_length = random.randint(20, 70)
        base_events = random.randint(10, 45)
        base_frequency = random.uniform(0.7, 0.95)
        onboarding_boost = random.uniform(1.1, 1.4)

        # ---------- Group behavior ----------
        if user_id in churned_users:
            decay_strength = random.uniform(0.55, 0.75)
            freq_drop = random.uniform(0.30, 0.45)
            churn_label = 1

        elif user_id in at_risk_users:
            decay_strength = random.uniform(0.40, 0.60)
            freq_drop = random.uniform(0.20, 0.35)
            churn_label = 0

        else:
            decay_strength = random.uniform(0.10, 0.25)
            freq_drop = random.uniform(0.05, 0.15)
            churn_label = 0

        # ---------- Generate sessions ----------
        for day in range(TOTAL_DAYS):
            date = START_DATE + timedelta(days=day)
            date_int = int(date.strftime("%Y%m%d"))

            # probability of session
            usage_prob = base_frequency * (1 - freq_drop * sigmoid(day))

            if (
                user_id in churned_users
                and day >= OBSERVATION_DAYS
                and random.random() < 0.85
            ):
                continue

            if random.random() > usage_prob:
                continue

            decay = decay_strength * sigmoid(day)

            session_length = int(
                base_length * (1 - decay)
                * (onboarding_boost if day < 14 else 1)
                + random.randint(-5, 5)
            )

            session_events = int(
                base_events * (1 - decay)
                + random.randint(-4, 4)
            )

            if session_length < 4 or session_events < 3:
                continue

            sessions.append({
                "sessionId": f"{user_id}_{session_counter}",
                "sessionDate": date_int,
                "sessionLength": session_length,
                "sessionEvents": session_events
            })

            session_counter += 1

        Users[user_id] = {
            "userId": user_id,
            "churnLabel": churn_label,
            "sessions": sessions
        }

    # ---------------- SUMMARY ----------------
    print("\n--- Simulation Summary ---")
    print(f"Total users: {NUM_USERS}")
    print(f"Churned users: {len(churned_users)}")
    print(f"At-risk users: {len(at_risk_users)}")
    print(f"Healthy users: {NUM_USERS - TARGET_CHURNED - TARGET_AT_RISK}\n")

    return Users, churned_users, at_risk_users