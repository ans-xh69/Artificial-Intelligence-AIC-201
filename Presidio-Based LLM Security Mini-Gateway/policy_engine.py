from config import LOW_THRESHOLD, HIGH_THRESHOLD
from presidio import detect_pii, anonymize_pii, detect_composite

def decide_action(score, text):
   
    pii_results = detect_pii(text)
    composite_flag, composite_entities = detect_composite(text)
    
    if score >= HIGH_THRESHOLD:
        return "BLOCK", "Request blocked due to security risk."
    
    elif score >= LOW_THRESHOLD or pii_results or composite_flag:
        masked_text = anonymize_pii(text)
        return "MASK", masked_text
    
    else:
        return "ALLOW", text