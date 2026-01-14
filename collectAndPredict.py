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
import numpy as np
import pandas as pd
from dbConfig import DB_PATH

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


def runModel(churn_user_ids):

    # --------------------------------------------------
    # Observation window
    # --------------------------------------------------
    OBSERVATION_START = 20250101
    OBSERVATION_END = 20250301  # 60 days

    conn = sqlite3.connect(DB_PATH)

    # --------------------------------------------------
    # Session-level data
    # --------------------------------------------------
    sessions_query = f"""
    SELECT
        userId,
        sessionLength,
        sessionEvents,
        sessionDate
    FROM sessions
    WHERE sessionDate >= {OBSERVATION_START}
      AND sessionDate <= {OBSERVATION_END};
    """

    df_sessions = pd.read_sql_query(sessions_query, conn)

    # --------------------------------------------------
    # Aggregate per user
    # --------------------------------------------------
    df = df_sessions.groupby("userId").agg({
        "sessionLength": ["mean", "std", "count"],
        "sessionEvents": ["mean", "std"],
        "sessionDate": ["min", "max"]
    }).reset_index()

    df.columns = [
        "userId",
        "avg_session_length",
        "std_session_length",
        "total_sessions",
        "avg_session_events",
        "std_session_events",
        "first_session_date",
        "last_session_date"
    ]

    # --------------------------------------------------
    # Trend features (4 periods)
    # --------------------------------------------------
    trend_query = f"""
    SELECT
        userId,

        AVG(CASE WHEN sessionDate <= 20250115 THEN sessionLength END) AS w1_length,
        COUNT(CASE WHEN sessionDate <= 20250115 THEN 1 END) AS w1_count,

        AVG(CASE WHEN sessionDate > 20250115 AND sessionDate <= 20250130 THEN sessionLength END) AS w2_length,
        COUNT(CASE WHEN sessionDate > 20250115 AND sessionDate <= 20250130 THEN 1 END) AS w2_count,

        AVG(CASE WHEN sessionDate > 20250130 AND sessionDate <= 20250214 THEN sessionLength END) AS w3_length,
        COUNT(CASE WHEN sessionDate > 20250130 AND sessionDate <= 20250214 THEN 1 END) AS w3_count,

        AVG(CASE WHEN sessionDate > 20250214 AND sessionDate <= {OBSERVATION_END} THEN sessionLength END) AS w4_length,
        COUNT(CASE WHEN sessionDate > 20250214 AND sessionDate <= {OBSERVATION_END} THEN 1 END) AS w4_count

    FROM sessions
    WHERE sessionDate >= {OBSERVATION_START}
      AND sessionDate <= {OBSERVATION_END}
    GROUP BY userId;
    """

    df_trends = pd.read_sql_query(trend_query, conn)
    conn.close()

    df = df.merge(df_trends, on="userId", how="left").fillna(0)

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    df["length_trend"] = (df["w4_length"] - df["w1_length"]) / (df["w1_length"] + 1)
    df["frequency_trend"] = (df["w4_count"] - df["w1_count"]) / (df["w1_count"] + 1)
    df["consistency_score"] = 1 / (df["std_session_length"] + 1)
    df["engagement_rate"] = df["avg_session_events"] / (df["avg_session_length"] + 1)
    df["sessions_per_day"] = df["total_sessions"] / 60

    # --------------------------------------------------
    # Churn label
    # --------------------------------------------------
    df["churn"] = df["userId"].apply(lambda x: 1 if x in churn_user_ids else 0)

    print("\nChurn distribution:")
    print(df["churn"].value_counts()) #.sort_index())

    # --------------------------------------------------
    # Feature selection
    # --------------------------------------------------
    FEATURES = [
        "avg_session_length",
        "avg_session_events",
        "total_sessions",
        "std_session_length",
        "length_trend",
        "frequency_trend",
        "consistency_score",
        "engagement_rate"
    ]

    X = df[FEATURES]
    y = df["churn"]

    # --------------------------------------------------
    # Add noise
    # --------------------------------------------------
    np.random.seed(42)
    X_noisy = X + np.random.normal(0, 0.05, X.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_noisy)

    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    # --------------------------------------------------
    # Handle class imbalance
    # --------------------------------------------------
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    # --------------------------------------------------
    # XGBoost model
    # --------------------------------------------------
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    y_prob = model.predict_proba(X_test)[:, 1]

    THRESHOLD = 0.30
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print("\n--- Model Evaluation (XGBoost) ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    
    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------
    feature_importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n--- Feature Importance ---")
    print(feature_importance.to_string(index=False))

    # --------------------------------------------------
    # Churn probability for all users
    # --------------------------------------------------
    df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

    at_risk = (
        df[df["churn"] == 0]
        .sort_values("churn_probability", ascending=False)
        .head(10)
    )

    print("\n--- Top 10 At-Risk Users (Not Yet Churned) ---")
    print(
        at_risk[
            ["userId", "churn_probability", "total_sessions",
             "length_trend", "frequency_trend"]
        ].round(4).to_string(index=False)
    )

    # --------------------------------------------------
    # Distribution diagnostics
    # --------------------------------------------------
    non_churned = df[df["churn"] == 0]

    print("\n--- Churn Probability Distribution (Non-Churned) ---")
    print(f"High risk (>0.7): {len(non_churned[non_churned['churn_probability'] > 0.7])}")
    print(f"Medium risk (0.4–0.7): {len(non_churned[(non_churned['churn_probability'] >= 0.4) & (non_churned['churn_probability'] <= 0.7)])}")
    print(f"Low risk (<0.4): {len(non_churned[non_churned['churn_probability'] < 0.4])}")

    print(f"\nMean probability: {non_churned['churn_probability'].mean():.4f}")
    print(f"Std deviation: {non_churned['churn_probability'].std():.4f}")