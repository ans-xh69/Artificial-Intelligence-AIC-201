from config import PATTERNS

def compute_risk_score(prompt: str):
   
    prompt_lower = prompt.lower()
    score = 0
    matched_patterns = []

    for category, data in PATTERNS.items():
        for keyword in data["keywords"]:
            if keyword in prompt_lower:
                score += data["weight"]
                matched_patterns.append(keyword)

    return score, matched_patterns