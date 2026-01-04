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
Advanced Churn Simulation Engine (Realistic)
- 5000 users total
- ~500 historically churned users (labels)
- ~1000 at-risk users (high churn probability)
- No 0% or 100% churn certainty
------------------------------------------------------
"""

# ---------------- CONFIG ----------------
NUM_USERS = 50000
DAYS = 120

CHURNED_USERS = 1000
AT_RISK_USERS = 10000

START_DATE = datetime(2025, 1, 1)

# ---------------- HELPERS ----------------
def sigmoid(x, k=0.06):
    return 1 / (1 + math.exp(k * x))


# ---------------- MAIN ----------------
def generate_users():
    all_users = list(range(1, NUM_USERS + 1))

    churned_user_ids = set(random.sample(all_users, CHURNED_USERS))
    remaining = list(set(all_users) - churned_user_ids)
    at_risk_user_ids = set(random.sample(remaining, AT_RISK_USERS))

    Users = {}

    for user_id in all_users:
        sessions = []
        session_counter = 1

        # Baseline profile
        base_length = random.randint(20, 65)
        base_events = random.randint(10, 40)
        base_frequency = random.uniform(0.65, 0.95)
        onboarding_boost = random.uniform(1.1, 1.4)

        # ---------------- GROUP BEHAVIOR ----------------
        if user_id in churned_user_ids:
            churn_label = 1
            churn_day = random.randint(45, 95)
            decay_strength = random.uniform(0.55, 0.85)
            frequency_drop = random.uniform(0.25, 0.45)

        elif user_id in at_risk_user_ids:
            churn_label = 0
            churn_day = None
            decay_strength = random.uniform(0.30, 0.55)
            frequency_drop = random.uniform(0.15, 0.30)

        else:
            churn_label = 0
            churn_day = None
            decay_strength = random.uniform(0.05, 0.20)
            frequency_drop = random.uniform(0.02, 0.10)

        # ---------------- SESSION GENERATION ----------------
        for day in range(DAYS):
            if churn_day and day >= churn_day:
                break

            date = START_DATE + timedelta(days=day)
            date_int = int(date.strftime("%Y%m%d"))

            usage_prob = base_frequency * (1 - frequency_drop * sigmoid(day))
            if random.random() > usage_prob:
                continue

            decay = sigmoid(day) * decay_strength

            session_length = int(
                base_length * (1 - decay)
                * (onboarding_boost if day < 14 else 1)
                + random.randint(-5, 5)
            )

            session_events = int(
                base_events * (1 - decay)
                + random.randint(-4, 4)
            )

            if session_length <= 3 or session_events <= 2:
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
    print(f"Churned users (label=1): {len(churned_user_ids)}")
    print(f"At-risk users: {len(at_risk_user_ids)}")
    print(f"Healthy users: {NUM_USERS - CHURNED_USERS - AT_RISK_USERS}\n")

    return Users, churned_user_ids, at_risk_user_ids