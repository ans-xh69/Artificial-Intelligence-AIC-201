import time
import ollama

from injection_detector import compute_risk_score
from presidio import detect_pii
from policy_engine import decide_action


def call_llm(prompt):
    response = ollama.chat(
        model='llama3',
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


def process_prompt(prompt):
    print("\nUser Input:", prompt)

    start_time = time.time()

    score, patterns = compute_risk_score(prompt)
    print("Risk Score:", score)
    print("Matched Patterns:", patterns)

    pii = detect_pii(prompt)
    print("PII Detected:", pii)

    action, safe_prompt = decide_action(score, prompt)
    print("Decision:", action)

    if action == "BLOCK":
        output = safe_prompt
    else:
        llm_response = call_llm(safe_prompt)
        output = llm_response

    end_time = time.time()
    latency = (end_time - start_time) * 1000

    print("Output:", output)
    print("Total Latency (ms):", round(latency, 2))

    return action, output, latency


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter prompt (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        process_prompt(user_input)