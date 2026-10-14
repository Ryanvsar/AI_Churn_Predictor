"""Turn per-user TreeSHAP contributions into readable explanations."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def top_factors_for_user(
    user_row: pd.Series,
    contrib_row: np.ndarray,
    feature_names: Sequence[str],
    labels: Dict[str, str],
    top_k: int = 8,
) -> List[Dict]:
    pairs = [(f, float(contrib_row[i])) for i, f in enumerate(feature_names) if np.isfinite(contrib_row[i])]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return [
        {
            "featureKey": f,
            "displayName": labels.get(f, f.replace("_", " ")),
            "shapValue": sv,
            "featureValue": _num(user_row.get(f)),
            "direction": "increases_risk" if sv > 0 else "decreases_risk",
        }
        for f, sv in pairs[: int(top_k)]
        if sv != 0.0
    ]


def why_risky_summary(top_factors: List[Dict]) -> str:
    inc = [f for f in top_factors if f.get("direction") == "increases_risk"]
    if not inc:
        return "This user does not show strong risk-increasing signals among the top factors."
    names = [f.get("displayName") or f.get("featureKey", "") for f in inc[:3]]
    names = [n for n in names if n]
    if len(names) == 1:
        return f"Highest-risk driver is {names[0]}."
    if len(names) == 2:
        return f"Highest-risk drivers are {names[0]} and {names[1]}."
    return f"Highest-risk drivers are {names[0]}, {names[1]}, and {names[2]}."
