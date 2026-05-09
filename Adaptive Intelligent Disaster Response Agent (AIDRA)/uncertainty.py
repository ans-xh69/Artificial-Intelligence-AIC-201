"""
uncertainty.py
Fuzzy Logic system for uncertainty handling.
Inputs : road_blockage_probability, hazard_spread_level, victim_severity
Output : risk_score (0–10) → influences routing and prioritisation decisions
"""

from typing import Tuple


# ── Membership functions ───────────────────────────────────────────────────────

def trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership function."""
    if x <= a or x >= d:
        return 0.0
    if x >= b and x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


# ── Fuzzification ──────────────────────────────────────────────────────────────

def fuzzify_blockage(prob: float) -> dict:
    """prob in [0, 1]"""
    return {
        "low":    trapmf(prob, 0.0, 0.0, 0.2, 0.4),
        "medium": trimf(prob,  0.2, 0.5, 0.8),
        "high":   trapmf(prob, 0.6, 0.8, 1.0, 1.0),
    }


def fuzzify_hazard(level: float) -> dict:
    """level in [0, 10]"""
    return {
        "low":    trapmf(level, 0, 0, 2, 4),
        "medium": trimf(level,  2, 5, 8),
        "high":   trapmf(level, 6, 8, 10, 10),
    }


def fuzzify_severity(score: int) -> dict:
    """score in {1, 2, 3}"""
    return {
        "minor":    trapmf(score, 0, 0, 1, 1.5),
        "moderate": trimf(score,  1, 2, 3),
        "critical": trapmf(score, 2.5, 3, 3, 3),
    }


# ── Rule base ─────────────────────────────────────────────────────────────────
#
# Risk output membership:
#   low    [0–3],  medium [2–6],  high [5–8],  critical [7–10]
#
# Rules (a sample of 9 rules covering key combinations):
# R1:  blockage=low  ∧ hazard=low  ∧ severity=minor    → risk=low
# R2:  blockage=low  ∧ hazard=low  ∧ severity=critical → risk=medium
# R3:  blockage=low  ∧ hazard=high ∧ severity=*        → risk=high
# R4:  blockage=med  ∧ hazard=med  ∧ severity=moderate → risk=medium
# R5:  blockage=med  ∧ hazard=med  ∧ severity=critical → risk=high
# R6:  blockage=high ∧ hazard=*    ∧ severity=critical → risk=critical
# R7:  blockage=high ∧ hazard=high ∧ severity=*        → risk=critical
# R8:  blockage=med  ∧ hazard=low  ∧ severity=minor    → risk=low
# R9:  blockage=low  ∧ hazard=med  ∧ severity=moderate → risk=medium
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_SETS = {
    "low":      (0.0, 1.5),
    "medium":   (3.0, 5.0),
    "high":     (6.0, 7.5),
    "critical": (8.5, 10.0),
}


def _centroid(activation: float, center: float) -> Tuple[float, float]:
    """Returns (weighted_sum, weight) for centroid defuzzification."""
    return activation * center, activation


def evaluate_rules(b: dict, h: dict, s: dict) -> dict:
    """Returns dict of output_set → activation strength (max aggregation)."""
    activations = {"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0}

    rules = [
        # (blockage, hazard, severity, output)
        ("low",    "low",    "minor",    "low"),
        ("low",    "low",    "critical", "medium"),
        ("low",    "high",   "minor",    "high"),
        ("low",    "high",   "moderate", "high"),
        ("low",    "high",   "critical", "critical"),
        ("medium", "medium", "moderate", "medium"),
        ("medium", "medium", "critical", "high"),
        ("high",   "medium", "critical", "critical"),
        ("high",   "high",   "minor",    "critical"),
        ("high",   "high",   "critical", "critical"),
        ("medium", "low",    "minor",    "low"),
        ("low",    "medium", "moderate", "medium"),
        ("medium", "high",   "moderate", "high"),
        ("high",   "low",    "moderate", "medium"),
    ]

    for br, hr, sr, out in rules:
        strength = min(b[br], h[hr], s[sr])
        activations[out] = max(activations[out], strength)

    return activations


def defuzzify(activations: dict) -> float:
    """Centroid defuzzification → crisp risk score in [0, 10]."""
    num, den = 0.0, 0.0
    for label, strength in activations.items():
        lo, hi = OUTPUT_SETS[label]
        center = (lo + hi) / 2
        ws, w = _centroid(strength, center)
        num += ws
        den += w
    if den == 0:
        return 5.0   # neutral fallback
    return num / den


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_risk_score(road_blockage_prob: float,
                       hazard_spread_level: float,
                       victim_severity: int) -> float:
    """
    Compute crisp risk score using Mamdani fuzzy inference.

    Parameters
    ----------
    road_blockage_prob  : float in [0, 1]   e.g. 0.7 = 70% chance blocked
    hazard_spread_level : float in [0, 10]  e.g. 8 = severe fire spread
    victim_severity     : int in {1,2,3}    1=minor, 2=moderate, 3=critical

    Returns
    -------
    risk_score : float in [0, 10]  (higher → avoid this route / act urgently)
    """
    b = fuzzify_blockage(road_blockage_prob)
    h = fuzzify_hazard(hazard_spread_level)
    s = fuzzify_severity(victim_severity)
    activations = evaluate_rules(b, h, s)
    return defuzzify(activations)


def interpret_risk(score: float) -> str:
    if score < 3.0:
        return "LOW"
    elif score < 5.5:
        return "MEDIUM"
    elif score < 7.5:
        return "HIGH"
    return "CRITICAL"


def fuzzy_decision(road_blockage_prob: float,
                   hazard_spread_level: float,
                   victim_severity: int) -> Tuple[float, str, str]:
    """
    Returns (score, label, recommended_action).
    """
    score  = compute_risk_score(road_blockage_prob, hazard_spread_level, victim_severity)
    label  = interpret_risk(score)
    if label == "LOW":
        action = "Use fastest route (low risk justified)"
    elif label == "MEDIUM":
        action = "Use A* with mild risk penalty"
    elif label == "HIGH":
        action = "Use safe A* route (avoid risk zones)"
    else:
        action = "Dispatch rescue team first; use safest route; treat as top priority"
    return score, label, action