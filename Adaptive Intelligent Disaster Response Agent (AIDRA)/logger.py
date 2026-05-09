"""
logger.py
Decision log for all agent actions, trade-off justifications, and replanning events.
"""

from typing import List
from dataclasses import dataclass, field
import time


@dataclass
class LogEntry:
    step: int
    event_type: str         # 'rescue', 'replan', 'csp', 'risk_eval', 'env_event', 'ml'
    message: str
    justification: str = ""


class DecisionLogger:
    def __init__(self):
        self.entries: List[LogEntry] = []
        self.start_time = time.time()

    def log(self, step: int, event_type: str, message: str, justification: str = ""):
        entry = LogEntry(step=step, event_type=event_type,
                         message=message, justification=justification)
        self.entries.append(entry)
        tag = f"[Step {step:>2} | {event_type.upper():<10}]"
        print(f"{tag} {message}")
        if justification:
            print(f"{'':>25} ↳ Reason: {justification}")

    def print_full_log(self):
        print("\n" + "="*70)
        print("FULL DECISION LOG")
        print("="*70)
        for e in self.entries:
            print(f"[Step {e.step:>2}] [{e.event_type.upper()}] {e.message}")
            if e.justification:
                print(f"         Reason: {e.justification}")
        elapsed = time.time() - self.start_time
        print(f"\nTotal simulation time: {elapsed:.2f}s")
        print("="*70)

    def replan_events(self) -> List[LogEntry]:
        return [e for e in self.entries if e.event_type == "replan"]

    def summary(self):
        types = {}
        for e in self.entries:
            types[e.event_type] = types.get(e.event_type, 0) + 1
        print("\n[Log Summary]")
        for t, count in types.items():
            print(f"  {t:<12}: {count} events")
        print(f"  Total replanning events: {len(self.replan_events())}")
