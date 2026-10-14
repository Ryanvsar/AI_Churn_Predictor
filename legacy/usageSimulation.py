import random
from datetime import datetime, timedelta

"""
------------------------------------------------------
Author: Ryan Vandersar
Date (DD-MM-YYYY): 20-12-2025
------------------------------------------------------
Program Title: Churn Prediction AI
------------------------------------------------------
File Description: Simulates usage metrics for 500 users
implanting churn behaviours into 15 users 
------------------------------------------------------
"""

def generate_users():
    NUM_USERS = 500
    DAYS = 90
    CHURN_USERS = 15

    # Pick 15 users to simulate churn
    churn_user_ids = set(random.sample(range(1, NUM_USERS + 1), CHURN_USERS))

    Users = {}

    start_date = datetime(2025, 1, 1)

    for user_id in range(1, NUM_USERS + 1):
        sessions = []
        session_counter = 1

        # Baseline behavior
        avg_length = random.randint(20, 60)
        avg_events = random.randint(10, 40)
        usage_probability = 0.85

        # Churn behavior modifiers
        if user_id in churn_user_ids:
            length_decay = avg_length / DAYS
            event_decay = avg_events / DAYS
            usage_probability = 0.75
        else:
            length_decay = 0
            event_decay = 0

        for day in range(DAYS):
            date = start_date + timedelta(days=day)
            date_int = int(date.strftime("%Y%m%d"))

            # Simulate less consistent usage for churn users
            if random.random() > usage_probability:
                continue

            # Apply decay for churn users
            session_length = max(
                1,
                int(avg_length - length_decay * day + random.randint(-5, 5))
            )

            session_events = max(
                1,
                int(avg_events - event_decay * day + random.randint(-3, 3))
            )

            sessions.append({
                "sessionId": f"{user_id}_{session_counter}",
                "sessionDate": date_int,
                "sessionLength": session_length,
                "sessionEvents": session_events
            })

            session_counter += 1

        Users[user_id] = {
            "userId": user_id,
            "churnPred": None,
            "sessions": sessions
        }

    # Output churn users for validation
    print("Users showing simulated churn behavior:")
    print(sorted(churn_user_ids))
    print()
    print("--------------------------")
    print()

    return Users, churn_user_ids
