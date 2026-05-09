"""
csp.py
Resource allocation modelled as a Constraint Satisfaction Problem.
Solver uses backtracking + MRV heuristic + forward checking.
"""

from typing import List, Dict, Optional, Tuple
from environment import Victim, Resources


# ── CSP Variables & Domains ───────────────────────────────────────────────────

class CSPState:
    """
    Variables : one per victim  →  which ambulance carries them
    Domain    : ambulance ids {0, 1}   (or -1 = rescue team on foot)
    Constraints:
      - Each ambulance carries at most 2 victims simultaneously
      - Rescue team (id=-1) handles at most 1 victim at a time
      - Critical victims must be assigned before moderate/minor if possible
    """

    def __init__(self, victims: List[Victim], resources: Resources):
        self.victims    = [v for v in victims if not v.rescued]
        self.resources  = resources
        self.num_amb    = resources.ambulances
        # assignment: victim_id -> ambulance_id (or -1 for rescue team)
        self.assignment: Dict[int, int] = {}
        self.backtrack_count = 0

    # ── Remaining domain for a victim ─────────────────────────────────────────
    def domain(self, victim: Victim, assignment: Dict[int,int]) -> List[int]:
        options = []
        for amb_id in range(self.num_amb):
            assigned = [v for v, a in assignment.items() if a == amb_id]
            if len(assigned) < self.resources.ambulance_capacity:
                options.append(amb_id)
        # rescue team as fallback (capacity 1)
        team_assigned = [v for v, a in assignment.items() if a == -1]
        if len(team_assigned) < 1:
            options.append(-1)
        return options

    # ── MRV: pick unassigned victim with smallest remaining domain ─────────────
    def select_unassigned(self, assignment: Dict[int,int]) -> Optional[Victim]:
        unassigned = [v for v in self.victims if v.id not in assignment]
        if not unassigned:
            return None
        # MRV: least domain size; break ties by highest severity (degree heuristic)
        return min(unassigned,
                   key=lambda v: (len(self.domain(v, assignment)), -v.priority))

    # ── Forward checking: ensure remaining victims still have a domain ─────────
    def forward_check(self, assignment: Dict[int,int]) -> bool:
        unassigned = [v for v in self.victims if v.id not in assignment]
        for v in unassigned:
            if not self.domain(v, assignment):
                return False
        return True

    # ── Constraint check for a single assignment ───────────────────────────────
    def is_consistent(self, victim: Victim, amb_id: int,
                       assignment: Dict[int,int]) -> bool:
        if amb_id == -1:
            team_count = sum(1 for a in assignment.values() if a == -1)
            return team_count < 1
        else:
            amb_count = sum(1 for a in assignment.values() if a == amb_id)
            return amb_count < self.resources.ambulance_capacity


# ── Backtracking Solver ────────────────────────────────────────────────────────

def backtrack(state: CSPState,
              assignment: Dict[int,int]) -> Optional[Dict[int,int]]:
    if len(assignment) == len(state.victims):
        return assignment  # complete assignment found

    victim = state.select_unassigned(assignment)
    if victim is None:
        return assignment

    for amb_id in state.domain(victim, assignment):
        if state.is_consistent(victim, amb_id, assignment):
            assignment[victim.id] = amb_id

            if state.forward_check(assignment):
                result = backtrack(state, assignment)
                if result is not None:
                    return result

            # backtrack
            del assignment[victim.id]
            state.backtrack_count += 1

    return None


def solve_csp(victims: List[Victim],
              resources: Resources) -> Tuple[Optional[Dict[int,int]], int]:
    """
    Entry point. Returns (assignment_dict, backtrack_count).
    assignment_dict: victim_id -> ambulance_id  (-1 = rescue team)
    """
    state  = CSPState(victims, resources)
    result = backtrack(state, {})
    return result, state.backtrack_count


# ── Solve without heuristics (pure backtracking, for comparison) ───────────────

class CSPStateNaive(CSPState):
    """Identical to CSPState but with no MRV/forward-checking."""

    def select_unassigned(self, assignment):
        unassigned = [v for v in self.victims if v.id not in assignment]
        return unassigned[0] if unassigned else None

    def forward_check(self, assignment):
        return True   # skip forward checking


def solve_csp_naive(victims: List[Victim],
                    resources: Resources) -> Tuple[Optional[Dict[int,int]], int]:
    state  = CSPStateNaive(victims, resources)
    result = backtrack(state, {})
    return result, state.backtrack_count


# ── Pretty print ───────────────────────────────────────────────────────────────

def print_csp_result(assignment: Optional[Dict[int,int]],
                     victims: List[Victim],
                     backtrack_count: int,
                     label: str = ""):
    tag = f"[CSP{' ' + label if label else ''}]"
    print(f"\n{tag} Backtrack count: {backtrack_count}")
    if assignment is None:
        print(f"{tag} No valid assignment found!")
        return
    victim_map = {v.id: v for v in victims}
    groups: Dict[int, List[int]] = {}
    for vid, amb in assignment.items():
        groups.setdefault(amb, []).append(vid)
    for amb_id, vids in sorted(groups.items()):
        label_str = f"Ambulance {amb_id}" if amb_id >= 0 else "Rescue Team"
        victim_descs = [
            f"V{vid}({victim_map[vid].severity})" for vid in vids
        ]
        print(f"  {label_str}: {', '.join(victim_descs)}")