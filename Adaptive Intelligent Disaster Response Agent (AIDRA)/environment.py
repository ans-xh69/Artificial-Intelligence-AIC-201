"""
environment.py
Grid map, victims, risk zones, medical centers, and dynamic event simulation.
"""

import csv
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# ── Victim ────────────────────────────────────────────────────────────────────

SEVERITY_LEVELS = {"critical": 3, "moderate": 2, "minor": 1}
KTAS_TO_SEVERITY = {1: "critical", 2: "critical", 3: "moderate", 4: "moderate", 5: "minor"}

@dataclass
class Victim:
    id: int
    position: Tuple[int, int]
    severity: str          # 'critical', 'moderate', 'minor'
    patient_record: Optional[Dict[str, str]] = None
    picked_up: bool = False
    rescued: bool = False
    rescue_time: Optional[float] = None

    @property
    def priority(self) -> int:
        return SEVERITY_LEVELS[self.severity]

    def __repr__(self):
        status = "RESCUED" if self.rescued else "WAITING"
        return f"Victim(id={self.id}, pos={self.position}, severity={self.severity}, {status})"


def _dataset_path() -> Path:
    return Path(__file__).with_name("dataset_clean.csv")


def _load_random_dataset_victims(count: int = 5) -> List[Victim]:
    path = _dataset_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Place the cleaned dataset next to environment.py."
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("KTAS_expert") or row.get("KTAS_RN")]

    if len(rows) < count:
        raise ValueError(f"Need at least {count} usable rows in {path.name}, found {len(rows)}")

    secure_random = random.SystemRandom()
    sampled_rows = secure_random.sample(rows, count)

    victims: List[Victim] = []
    for idx, row in enumerate(sampled_rows):
        ktas_raw = row.get("KTAS_expert") or row.get("KTAS_RN")
        try:
            ktas = int(float(str(ktas_raw).replace(",", ".")))
        except ValueError:
            ktas = 3
        ktas = max(1, min(5, ktas))
        severity = KTAS_TO_SEVERITY.get(ktas, "moderate")
        victims.append(Victim(id=idx, position=(0, 0), severity=severity, patient_record=row))

    return victims


# ── Grid Map ──────────────────────────────────────────────────────────────────

CELL_FREE      = 0
CELL_BLOCKED   = 1
CELL_HIGH_RISK = 2

@dataclass
class GridMap:
    rows: int
    cols: int
    grid: List[List[int]] = field(default_factory=list)
    medical_centers: List[Tuple[int, int]] = field(default_factory=list)
    rescue_base: Tuple[int, int] = (0, 0)

    def __post_init__(self):
        if not self.grid:
            self.grid = [[CELL_FREE] * self.cols for _ in range(self.rows)]

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return self.in_bounds(pos) and self.grid[r][c] != CELL_BLOCKED

    def is_high_risk(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return self.in_bounds(pos) and self.grid[r][c] == CELL_HIGH_RISK

    def block_road(self, pos: Tuple[int, int], 
                   victims: Optional[List[Victim]] = None):
        r, c = pos
    
        # Never block victim positions
        if victims:
            for v in victims:
                if not v.rescued and not v.picked_up and v.position == pos:
                    return
    
        if self.in_bounds(pos):
            self.grid[r][c] = CELL_BLOCKED

    def neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = pos
        moves = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
        return [p for p in moves if self.is_walkable(p)]

    def print_map(self, victims: List[Victim], agent_pos: Optional[Tuple[int,int]] = None):
        symbols = {CELL_FREE: '.', CELL_BLOCKED: '#', CELL_HIGH_RISK: '!'}
        victim_positions = {v.position: v for v in victims if not v.rescued and not v.picked_up}
        print("\n  " + " ".join(str(c) for c in range(self.cols)))
        for r in range(self.rows):
            row_str = f"{r} "
            for c in range(self.cols):
                pos = (r, c)
                if pos == agent_pos:
                    row_str += "A "
                elif pos == self.rescue_base:
                    row_str += "B "
                elif pos in self.medical_centers:
                    row_str += "M "
                elif pos in victim_positions:
                    v = victim_positions[pos]
                    row_str += f"V "
                else:
                    row_str += symbols[self.grid[r][c]] + " "
            print(row_str)
        print()


# ── Resources ─────────────────────────────────────────────────────────────────

@dataclass
class Resources:
    ambulances: int = 2
    rescue_teams: int = 1
    medical_kits: int = 10
    ambulance_capacity: int = 2   # max victims per ambulance

    ambulance_assignments: Dict[int, List[int]] = field(default_factory=dict)  # amb_id -> [victim_ids]
    team_busy: bool = False

    def __post_init__(self):
        for i in range(self.ambulances):
            self.ambulance_assignments[i] = []

    def available_ambulance(self) -> Optional[int]:
        for amb_id, victims in self.ambulance_assignments.items():
            if len(victims) < self.ambulance_capacity:
                return amb_id
        return None

    def assign_victim_to_ambulance(self, victim_id: int, amb_id: int) -> bool:
        if len(self.ambulance_assignments[amb_id]) < self.ambulance_capacity:
            self.ambulance_assignments[amb_id].append(victim_id)
            return True
        return False

    def release_ambulance(self, amb_id: int):
        self.ambulance_assignments[amb_id] = []

    def use_kit(self) -> bool:
        if self.medical_kits > 0:
            self.medical_kits -= 1
            return True
        return False

    def summary(self) -> str:
        lines = ["[Resources]"]
        for amb_id, victims in self.ambulance_assignments.items():
            lines.append(f"  Ambulance {amb_id}: victims {victims} ({len(victims)}/{self.ambulance_capacity})")
        lines.append(f"  Rescue Team busy: {self.team_busy}")
        lines.append(f"  Medical Kits remaining: {self.medical_kits}")
        return "\n".join(lines)


# ── Dynamic Events ─────────────────────────────────────────────────────────────

class EventSimulator:
    """Simulates dynamic environmental changes during a mission."""

    def __init__(self, grid_map: GridMap, seed: int = 42):
        self.map = grid_map
        random.seed(seed)
        self.event_log: List[str] = []

    def maybe_block_road(self, step: int, victims: Optional[List[Victim]] = None,
                         probability: float = 0.15) -> Optional[Tuple[int,int]]:
        """With some probability, randomly block a free cell."""
        if random.random() < probability:
    
            victim_positions = set()
            
            if victims:
                victim_positions = {
                    v.position
                    for v in victims
                    if not v.rescued
                }
            
            free_cells = [
                (r, c)
                for r in range(self.map.rows)
                for c in range(self.map.cols)
                if self.map.grid[r][c] == CELL_FREE
                and (r, c) not in victim_positions
            ]
            
            if free_cells:
                pos = random.choice(free_cells)
                self.map.block_road(pos, victims)
                msg = f"[Step {step}] DYNAMIC EVENT: Road blocked at {pos} (aftershock/fire)"
                self.event_log.append(msg)
                return pos
        return None

    def maybe_change_risk(self, step: int, victims: Optional[List[Victim]] = None,
                          probability: float = 0.10):
        """Randomly upgrade a free cell to high-risk."""
        if random.random() < probability:
            free_cells = [
                (r, c)
                for r in range(self.map.rows)
                for c in range(self.map.cols)
                if self.map.grid[r][c] == CELL_FREE
            ]
            if free_cells:
                r, c = random.choice(free_cells)
                self.map.grid[r][c] = CELL_HIGH_RISK
                msg = f"[Step {step}] DYNAMIC EVENT: New high-risk zone at ({r},{c}) (fire spreading)"
                self.event_log.append(msg)


# ── Scenario Builder ───────────────────────────────────────────────────────────

def build_default_scenario() -> Tuple[GridMap, List[Victim], Resources]:
    """
    Builds the baseline 10x10 grid scenario described in the CCP.
    Returns the map, victim list, and resource state.
    """
    rows, cols = 10, 10
    gmap = GridMap(rows=rows, cols=cols)

    # Rescue base at top-left
    gmap.rescue_base = (0, 0)

    # Two fixed medical centers
    gmap.medical_centers = [(0, 9), (9, 0)]

    # Pre-defined high-risk zones (fire / structural collapse)
    high_risk = [(2,2),(2,3),(3,2),(3,3),(5,5),(5,6),(6,5)]
    for r, c in high_risk:
        gmap.grid[r][c] = CELL_HIGH_RISK

    # Pre-defined blocked roads
    blocked = [(1,4),(4,1),(7,3),(3,7)]
    for r, c in blocked:
        gmap.grid[r][c] = CELL_BLOCKED

    # Five victims sampled from the real triage dataset.
    victims = _load_random_dataset_victims(count=5)

    # Sample different free cells every run so the scenario changes naturally.
    # Spread victims across different columns to avoid clustering.
    reserved_positions = set(gmap.medical_centers + [gmap.rescue_base] + high_risk + blocked)
    free_by_col = {
        c: [(r, c) for r in range(rows) if (r, c) not in reserved_positions]
        for c in range(cols)
    }
    eligible_cols = [c for c, cells in free_by_col.items() if cells]
    secure_random = random.SystemRandom()
    sampled_cols = secure_random.sample(eligible_cols, len(victims))
    sampled_positions = []
    for col in sampled_cols:
        sampled_positions.append(secure_random.choice(free_by_col[col]))
    for victim, position in zip(victims, sampled_positions):
        victim.position = position

    resources = Resources()

    return gmap, victims, resources
