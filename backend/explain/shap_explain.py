from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShapArtifacts:
    feature_names: Tuple[str, ...]
    shap_values: np.ndarray  # shape: (n_rows, n_features)
    base_value: float
    global_importance: List[Dict]


def compute_shap_for_xgb(
    model,
    X_scaled: np.ndarray,
    feature_names: List[str],
    max_rows: Optional[int] = 5000,
) -> ShapArtifacts:
    # Import lazily so backend can still run without SHAP installed (UI will show limited explainability).
    import shap  # type: ignore

    n = int(X_scaled.shape[0])
    if max_rows is not None and n > int(max_rows):
        idx = np.linspace(0, n - 1, int(max_rows), dtype=int)
        X_use = X_scaled[idx]
        row_map = idx
    else:
        X_use = X_scaled
        row_map = None

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_use)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base_val = float(np.array(base).ravel()[0])
    else:
        base_val = float(base)

    shap_values = np.asarray(sv, dtype=float)
    # Global importance = mean |SHAP|
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)
    global_imp = [
        {"featureKey": feature_names[i], "importance": float(mean_abs[i])} for i in order
    ]

    if row_map is not None:
        # Expand to full shape with NaNs for non-computed rows
        full = np.full((n, shap_values.shape[1]), np.nan, dtype=float)
        full[row_map] = shap_values
        shap_values = full

    return ShapArtifacts(
        feature_names=tuple(feature_names),
        shap_values=shap_values,
        base_value=base_val,
        global_importance=global_imp,
    )


def top_factors_for_user(
    user_row: pd.Series,
    shap_row: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 8,
) -> List[Dict]:
    pairs = []
    for i, f in enumerate(feature_names):
        sv = shap_row[i]
        if np.isnan(sv):
            continue
        fv = float(user_row.get(f, 0.0))
        pairs.append((f, float(sv), fv))

    # Sort by absolute contribution, but keep sign for direction
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    out = []
    for f, sv, fv in pairs[: int(top_k)]:
        out.append(
            {
                "featureKey": f,
                "displayName": f.replace("_", " ").title(),
                "shapValue": float(sv),
                "featureValue": float(fv),
                "direction": "increases_risk" if sv > 0 else "decreases_risk",
            }
        )
    return out


def why_risky_summary(top_factors: List[Dict]) -> str:
    inc = [f for f in top_factors if f.get("direction") == "increases_risk"]
    if not inc:
        return "This user does not show strong risk-increasing signals among the top factors."
    names = [f.get("displayName", f.get("featureKey", "")) for f in inc[:3]]
    names = [n for n in names if n]
    if not names:
        return "Top risk drivers were not available."
    if len(names) == 1:
        return f"Highest-risk driver is {names[0]}."
    if len(names) == 2:
        return f"Highest-risk drivers are {names[0]} and {names[1]}."
    return f"Highest-risk drivers are {names[0]}, {names[1]}, and {names[2]}."

