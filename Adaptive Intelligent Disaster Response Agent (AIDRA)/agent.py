"""
agent.py
Intelligent Agent: integrates search, CSP, ML, and fuzzy logic
to plan and execute the disaster response mission.
"""

from typing import List, Tuple, Optional, Dict
from environment import GridMap, Victim, Resources, EventSimulator, CELL_HIGH_RISK
from search import astar, compare_algorithms, SearchResult
from csp import solve_csp, solve_csp_naive, print_csp_result
from ml_model import train_and_evaluate, victim_features
from uncertainty import fuzzy_decision
from logger import DecisionLogger
from metrics import KPITracker
from visualizer import generate_all, plot_grid


class AIDRAAgent:
    def __init__(self,
                 grid_map: GridMap,
                 victims: List[Victim],
                 resources: Resources,
                 event_sim: EventSimulator):
        self.map       = grid_map
        self.victims   = victims
        self.resources = resources
        self.events    = event_sim
        self.logger    = DecisionLogger()
        self.kpis      = KPITracker()
        self.kpis.victims_total = len(victims)
        self.step      = 0
        self._victim_routes: Dict[int, list] = {}   # victim_id -> route path

        # Train ML models once at startup
        print("\n" + "="*70)
        print("AIDRA — Adaptive Intelligent Disaster Response Agent")
        print("="*70)
        print("\n[Startup] Training ML models on synthetic disaster dataset…")
        self.knn, self.nb, knn_m, nb_m = train_and_evaluate()
        self.kpis.record_ml("kNN (k=5)", knn_m)
        self.kpis.record_ml("Naive Bayes", nb_m)

    # ── Victim prioritisation ──────────────────────────────────────────────────

    def prioritise_victims(self) -> List[Victim]:
        """Sort unrescued victims: critical first, then by ML survival probability."""
        unrescued = [v for v in self.victims if not v.rescued]
        scored = []
        for v in unrescued:
            dist = self._manhattan_to_nearest_center(v.position)
            risk_nearby = int(self.map.is_high_risk(v.position))
            feats = victim_features(v, dist, risk_nearby,
                                    time_elapsed=self.step,
                                    kits_remaining=self.resources.medical_kits)
            # Lower survival prob → higher urgency
            survival = self.knn.predict_proba(feats)
            scored.append((v, survival))
            self.logger.log(self.step, "ml",
                f"V{v.id} ({v.severity}): kNN survival_prob={survival:.2f}",
                f"Low survival probability raises urgency")

        # Sort: severity descending, then survival probability ascending (most at risk first)
        scored.sort(key=lambda x: (-x[0].priority, x[1]))
        return [v for v, _ in scored]

    # ── Route decision ─────────────────────────────────────────────────────────

    def choose_route(self, start: Tuple[int,int],
                     goal: Tuple[int,int],
                     victim: Victim) -> Tuple[SearchResult, str]:
        """
        Uses fuzzy logic to decide between fast (A* no penalty) and
        safe (A* with risk penalty) routing. Returns (result, strategy_used).
        """
        blockage_prob   = self._estimate_blockage_prob(start, goal)
        hazard_level    = self._estimate_hazard_level(start, goal)
        score, label, action = fuzzy_decision(blockage_prob, hazard_level, victim.priority)

        self.logger.log(self.step, "risk_eval",
            f"Fuzzy risk for route {start}→{goal}: score={score:.2f} [{label}]",
            action)

        # Choose strategy based on fuzzy output
        if label in ("HIGH", "CRITICAL"):
            result = astar(self.map, start, goal, avoid_risk=True)
            strategy = "A*(safe) — risk penalty applied"
            tradeoff = "Prioritised SAFETY over speed (longer path, fewer risk cells)"
        else:
            result = astar(self.map, start, goal, avoid_risk=False)
            strategy = "A*(fast) — shortest path"
            tradeoff = "Prioritised SPEED over safety (shorter path, acceptable risk)"

        if not result.found:
            self.logger.log(self.step, "replan",
                f"No path from {start} to {goal} — route blocked!",
                "Will attempt alternative medical center")
            self.kpis.record_replan()
            return result, strategy

        self.logger.log(self.step, "rescue",
            f"Route selected: {len(result.path)-1} steps, "
            f"risk_cells={result.risk_cells}, cost={result.cost:.1f}",
            tradeoff)
        return result, strategy

    # ── Resource allocation via CSP ────────────────────────────────────────────

    def allocate_resources(self) -> Optional[Dict[int,int]]:
        unrescued = [v for v in self.victims if not v.rescued]

        self.logger.log(self.step, "csp",
            f"Running CSP (with MRV + forward-checking) for {len(unrescued)} victims")
        assignment, bt = solve_csp(unrescued, self.resources)
        self.kpis.record_backtrack("with_heuristic", bt)
        print_csp_result(assignment, unrescued, bt, label="MRV+FC")

        self.logger.log(self.step, "csp",
            f"Running CSP (naive backtracking) for comparison")
        _, bt_naive = solve_csp_naive(unrescued, self.resources)
        self.kpis.record_backtrack("naive", bt_naive)
        print(f"  [CSP Naive] Backtrack count: {bt_naive}  "
              f"(vs heuristic: {bt})")

        return assignment

    # ── Search algorithm comparison ────────────────────────────────────────────

    def compare_search_algorithms(self, start, goal):
        print(f"\n[Search Comparison] {start} → {goal}")
        results = compare_algorithms(self.map, start, goal)
        for r in results:
            print(f"  {r.summary()}")
            self.kpis.record_search(r)
        return results

    # ── Main rescue loop ───────────────────────────────────────────────────────

    def run(self):
        self.logger.log(0, "rescue", "Mission started. Assessing victims…")
        self.map.print_map(self.victims, agent_pos=self.map.rescue_base)

        # Step 1: CSP allocation
        assignment = self.allocate_resources()
        if assignment is None:
            print("[AGENT] CSP failed — no valid allocation found.")
            return

        # Step 2: Prioritise victims
        priority_order = self.prioritise_victims()
        self.logger.log(self.step, "rescue",
            "Rescue priority order: " +
            ", ".join(f"V{v.id}({v.severity})" for v in priority_order),
            "Critical victims first, then sorted by ML-estimated survival urgency")

        # Step 3: Compare search on first victim as demo
        first_victim = priority_order[0]
        self._comparison_results = self.compare_search_algorithms(
            self.map.rescue_base, first_victim.position)

        # Step 4: Execute rescues in priority order
        for victim in priority_order:
            self.step += 1
            self._rescue_victim(victim, assignment)

        # Step 5: Final report
        self.kpis.print_report()
        self.logger.print_full_log()
        self.logger.summary()

        # Step 6: Generate all matplotlib figures
        generate_all(self.map, self.victims, self.kpis,
                     comparison_results=self._comparison_results,
                     knn_model=self.knn,
                     nb_model=self.nb,
                     victim_routes=self._victim_routes)

        # Step 7: Export Pandas CSVs
        self.kpis.export_csv("figures")

    # ── Single rescue operation ────────────────────────────────────────────────

    def _rescue_victim(self, victim: Victim, assignment: Dict[int,int]):
        amb_id = assignment.get(victim.id, 0)
        label  = f"Ambulance {amb_id}" if amb_id >= 0 else "Rescue Team"

        self.logger.log(self.step, "rescue",
            f"Dispatching {label} to V{victim.id} at {victim.position} "
            f"[{victim.severity.upper()}]",
            f"Assigned by CSP solver")

        # Dynamic event: maybe block a road
        blocked = self.events.maybe_block_road(self.step)
        if blocked:
            self.logger.log(self.step, "replan",
                f"Road blocked at {blocked} — replanning route",
                "Environmental change detected; A* reruns on updated map")
            self.kpis.record_replan()

        self.events.maybe_change_risk(self.step)

        # Choose route to victim
        result, strategy = self.choose_route(
            self.map.rescue_base, victim.position, victim)

        if not result.found:
            # Try second medical center
            alt_found = False
            for center in self.map.medical_centers:
                alt, _ = self.choose_route(self.map.rescue_base, center, victim)
                if alt.found:
                    self.logger.log(self.step, "replan",
                        f"Rerouted V{victim.id} to alternate center {center}",
                        "Primary route unavailable")
                    result = alt
                    alt_found = True
                    break
            if not alt_found:
                self.logger.log(self.step, "rescue",
                    f"V{victim.id} UNREACHABLE — skipping",
                    "All routes blocked")
                return

        # Route to nearest medical center after picking up victim
        center = self._nearest_center(victim.position)
        center_result, _ = self.choose_route(victim.position, center, victim)

        total_steps = (len(result.path) - 1) + (
            len(center_result.path) - 1 if center_result.found else 0)

        # Administer kit
        kit_ok = self.resources.use_kit()
        if kit_ok:
            self.kpis.kits_used += 1

        # Mark rescued
        victim.rescued    = True
        victim.rescue_time = total_steps
        self._victim_routes[victim.id] = result.path   # store for visualizer
        self.kpis.record_rescue(victim, total_steps)
        self.kpis.record_search(result)
        self.kpis.ambulance_trips += 1

        self.logger.log(self.step, "rescue",
            f"V{victim.id} RESCUED in {total_steps} steps via {strategy}",
            f"Kit administered: {kit_ok}. "
            f"Delivered to medical center {center}.")

        self.map.print_map(self.victims, agent_pos=center)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _manhattan_to_nearest_center(self, pos: Tuple[int,int]) -> float:
        return min(abs(pos[0]-c[0]) + abs(pos[1]-c[1])
                   for c in self.map.medical_centers)

    def _nearest_center(self, pos: Tuple[int,int]) -> Tuple[int,int]:
        return min(self.map.medical_centers,
                   key=lambda c: abs(pos[0]-c[0]) + abs(pos[1]-c[1]))

    def _estimate_blockage_prob(self, start, goal) -> float:
        """Heuristic: fraction of cells along Manhattan path that are high-risk."""
        r1, c1 = start
        r2, c2 = goal
        dist = abs(r1-r2) + abs(c1-c2)
        if dist == 0:
            return 0.0
        risk_count = 0
        # Sample along the straight-line path
        for t in range(dist + 1):
            r = r1 + round((r2 - r1) * t / dist)
            c = c1 + round((c2 - c1) * t / dist)
            if self.map.is_high_risk((r, c)):
                risk_count += 1
        return min(risk_count / dist, 1.0)

    def _estimate_hazard_level(self, start, goal) -> float:
        """Returns hazard level [0–10] based on risk cells near path."""
        prob = self._estimate_blockage_prob(start, goal)
        return prob * 10