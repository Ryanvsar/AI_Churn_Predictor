"""
------------------------------------------------------
Author: Ryan Vandersar
Date (DD-MM-YYYY): 27-12-2025
------------------------------------------------------
Program Title: Churn Prediction AI
------------------------------------------------------
File Description:  Collect data from database and input
into Logistic Regression model to predict churn 
------------------------------------------------------
"""
import sqlite3
import pandas as pd
import numpy as np
from usageSimulation import churn_user_ids
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def runModel():
    CURRENT_DATE = 20250331

    conn = sqlite3.connect("app_usage.db")

    query = """
    SELECT
        userId,
        COUNT(*) AS total_sessions,
        AVG(sessionLength) AS avg_session_length,
        AVG(sessionEvents) AS avg_session_events,
        MAX(sessionDate) AS last_session_date,
        MIN(sessionDate) AS first_session_date
    FROM sessions
    GROUP BY userId;
    """

    df = pd.read_sql_query(query, conn)

    print(df.head())

    trend_query = """
    SELECT
        userId,
        AVG(CASE WHEN sessionDate <= 20250130 THEN sessionLength END) AS early_len,
        AVG(CASE WHEN sessionDate >= 20250301 THEN sessionLength END) AS recent_len,
        AVG(CASE WHEN sessionDate <= 20250130 THEN sessionEvents END) AS early_evt,
        AVG(CASE WHEN sessionDate >= 20250301 THEN sessionEvents END) AS recent_evt
    FROM sessions
    GROUP BY userId;
    """

    df_trends = pd.read_sql_query(trend_query, conn)
    conn.close()

    df = df.merge(df_trends, on="userId")

    df["session_length_change"] = df["recent_len"] - df["early_len"]
    df["session_event_change"] = df["recent_evt"] - df["early_evt"]

    df["days_since_last_session"] = CURRENT_DATE - df["last_session_date"]

    df["churn"] = df["userId"].apply(lambda x: 1 if x in churn_user_ids else 0)

    df["session_length_change"] = df["session_length_change"].fillna(0)
    df["session_event_change"] = df["session_event_change"].fillna(0)

    print(df["churn"].value_counts())

    FEATURES = [
        "avg_session_length",
        "avg_session_events",
        "total_sessions",
        "session_length_change",
        "session_event_change",
        "days_since_last_session"
    ]

    X = df[FEATURES]
    y = df["churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    # Initialize Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # important for churn imbalance
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # ------------------------------------------------------
    # CHURN RISK SCORING FOR ALL USERS
    # ------------------------------------------------------
    df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

    # Show highest-risk users
    top_risk_users = df.sort_values("churn_probability", ascending=False).head(15)

    print("\n--- Top At-Risk Users ---")
    print(top_risk_users[["userId", "churn_probability"]])