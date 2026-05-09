"""
gui_simulator.py
Real-time Tkinter GUI for AIDRA simulation.
Shows animated grid, live decision log, KPI panel, and step-by-step execution.

Run: python gui_simulator.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import time
from typing import List, Tuple, Optional, Dict
from environment import GridMap, Victim, Resources, EventSimulator, CELL_BLOCKED, CELL_HIGH_RISK
from search import astar
from csp import solve_csp
from ml_model import train_and_evaluate, victim_features
from uncertainty import fuzzy_decision
from metrics import KPITracker


# ── Color Scheme ───────────────────────────────────────────────────────────────
COLORS = {
    "free":       "#f0f4f8",
    "blocked":    "#2d3748",
    "high_risk":  "#fc8181",
    "base":       "#4299e1",
    "center":     "#48bb78",
    "critical":   "#e53e3e",
    "moderate":   "#ed8936",
    "minor":      "#ecc94b",
    "route":      "#805ad5",
    "agent":      "#f6e05e",
    "rescued":    "#68d391",
}


class AIDRASimulatorGUI:
    def __init__(self, root, grid_map: GridMap, victims: List[Victim],
                 resources: Resources, event_sim: EventSimulator):
        self.root = root
        self.root.title("AIDRA — Real-Time Disaster Response Simulation")
        self.root.geometry("1400x900")
        self.root.configure(bg="#edf2f7")

        self.map = grid_map
        self.victims = victims
        self.resources = resources
        self.events = event_sim
        self.kpis = KPITracker()
        self.kpis.victims_total = len(victims)

        # Agent state
        self.step = 0
        self.current_route: List[Tuple[int, int]] = []
        self.agent_pos = grid_map.rescue_base
        self.animation_speed = 300  # ms per step
        self.is_running = False
        self.route_index = 0

        # ML models
        self.knn = None
        self.nb = None

        # UI elements
        self.cell_size = 50
        self.setup_ui()
        self.draw_grid()
        self.update_kpi_display()
        self.log("System initialized. Click START to begin simulation.")

    # ── UI Setup ───────────────────────────────────────────────────────────────

    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg="#edf2f7")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: Grid
        left_panel = tk.Frame(main_frame, bg="white", relief=tk.RIDGE, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        title = tk.Label(left_panel, text="AIDRA — Disaster Response Grid",
                         font=("Arial", 14, "bold"), bg="white", fg="#2d3748")
        title.pack(pady=10)

        self.canvas = tk.Canvas(left_panel, width=self.map.cols * self.cell_size,
                                height=self.map.rows * self.cell_size, bg="white")
        self.canvas.pack(padx=10, pady=10)

        # Right panel: Controls + Log + KPIs
        right_panel = tk.Frame(main_frame, bg="#edf2f7")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # Controls
        control_frame = tk.LabelFrame(right_panel, text="Simulation Controls",
                                      font=("Arial", 11, "bold"), bg="white",
                                      fg="#2d3748", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = tk.Button(control_frame, text="▶ START", command=self.start_simulation,
                                   bg="#48bb78", fg="white", font=("Arial", 11, "bold"),
                                   width=12, cursor="hand2")
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)

        self.pause_btn = tk.Button(control_frame, text="⏸ PAUSE", command=self.pause_simulation,
                                   bg="#ed8936", fg="white", font=("Arial", 11, "bold"),
                                   width=12, state=tk.DISABLED, cursor="hand2")
        self.pause_btn.grid(row=0, column=1, padx=5, pady=5)

        self.reset_btn = tk.Button(control_frame, text="↻ RESET", command=self.reset_simulation,
                                   bg="#4299e1", fg="white", font=("Arial", 11, "bold"),
                                   width=12, cursor="hand2")
        self.reset_btn.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Speed control
        speed_label = tk.Label(control_frame, text="Animation Speed:", bg="white",
                              font=("Arial", 10))
        speed_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.speed_var = tk.IntVar(value=300)
        speed_slider = tk.Scale(control_frame, from_=50, to=1000, orient=tk.HORIZONTAL,
                               variable=self.speed_var, command=self.update_speed,
                               bg="white", length=200)
        speed_slider.grid(row=3, column=0, columnspan=2, pady=(0, 5))

        # KPI Panel
        kpi_frame = tk.LabelFrame(right_panel, text="Live KPIs",
                                 font=("Arial", 11, "bold"), bg="white",
                                 fg="#2d3748", padx=10, pady=10)
        kpi_frame.pack(fill=tk.X, pady=(0, 10))

        self.kpi_labels = {}
        kpi_items = [
            ("Victims Saved", "0 / 5"),
            ("Current Step", "0"),
            ("Avg Rescue Time", "0.0 steps"),
            ("Kits Used", "0 / 10"),
            ("Replan Events", "0"),
        ]
        for i, (key, default_val) in enumerate(kpi_items):
            tk.Label(kpi_frame, text=f"{key}:", bg="white",
                    font=("Arial", 9, "bold"), anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            label = tk.Label(kpi_frame, text=default_val, bg="white",
                           font=("Arial", 9), fg="#4299e1", anchor="e")
            label.grid(row=i, column=1, sticky="e", pady=2)
            self.kpi_labels[key] = label

        # Decision Log
        log_frame = tk.LabelFrame(right_panel, text="Decision Log",
                                 font=("Arial", 11, "bold"), bg="white",
                                 fg="#2d3748", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=50,
                                                  font=("Courier", 9), bg="#f7fafc",
                                                  fg="#2d3748", wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ── Grid Drawing ───────────────────────────────────────────────────────────

    def draw_grid(self):
        self.canvas.delete("all")
        cs = self.cell_size

        # Draw cells
        for r in range(self.map.rows):
            for c in range(self.map.cols):
                x1, y1 = c * cs, r * cs
                x2, y2 = x1 + cs, y1 + cs
                cell_type = self.map.grid[r][c]

                if cell_type == CELL_BLOCKED:
                    color = COLORS["blocked"]
                elif cell_type == CELL_HIGH_RISK:
                    color = COLORS["high_risk"]
                else:
                    color = COLORS["free"]

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                                            outline="#cbd5e0", width=1)

        # Draw route
        if self.current_route and len(self.current_route) > 1:
            for i in range(len(self.current_route) - 1):
                r1, c1 = self.current_route[i]
                r2, c2 = self.current_route[i + 1]
                x1, y1 = c1 * cs + cs // 2, r1 * cs + cs // 2
                x2, y2 = c2 * cs + cs // 2, r2 * cs + cs // 2
                self.canvas.create_line(x1, y1, x2, y2, fill=COLORS["route"],
                                      width=4, dash=(5, 3))

        # Draw base
        br, bc = self.map.rescue_base
        self.draw_circle(br, bc, COLORS["base"], "B")

        # Draw medical centers
        for mr, mc in self.map.medical_centers:
            self.draw_circle(mr, mc, COLORS["center"], "M")

        # Draw victims
        for v in self.victims:
            vr, vc = v.position
            if v.rescued:
                self.draw_circle(vr, vc, COLORS["rescued"], "✓", text_color="white")
            else:
                sev_color = COLORS.get(v.severity, COLORS["moderate"])
                self.draw_circle(vr, vc, sev_color, f"V{v.id}")

        # Draw agent
        ar, ac = self.agent_pos
        self.draw_agent(ar, ac)

    def draw_circle(self, row, col, color, text, text_color="white"):
        cs = self.cell_size
        cx, cy = col * cs + cs // 2, row * cs + cs // 2
        r = cs // 3
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                               fill=color, outline="white", width=2)
        self.canvas.create_text(cx, cy, text=text, fill=text_color,
                               font=("Arial", 10, "bold"))

    def draw_agent(self, row, col):
        cs = self.cell_size
        cx, cy = col * cs + cs // 2, row * cs + cs // 2
        r = cs // 4
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                               fill=COLORS["agent"], outline="#744210", width=3)
        self.canvas.create_text(cx, cy, text="A", fill="#744210",
                               font=("Arial", 11, "bold"))

    # ── Logging ────────────────────────────────────────────────────────────────

    def log(self, message: str, color: str = "black"):
        self.log_text.insert(tk.END, f"{message}\n", color)
        self.log_text.see(tk.END)
        self.log_text.update()

    # ── Control Handlers ───────────────────────────────────────────────────────

    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.log("═" * 50, "blue")
            self.log("▶ SIMULATION STARTED", "green")
            self.log("═" * 50, "blue")

            # Train ML models
            if self.knn is None:
                self.log("Training ML models (kNN + Naive Bayes)...", "purple")
                self.knn, self.nb, _, _ = train_and_evaluate()
                self.log("✓ ML models trained successfully", "green")

            self.run_agent()

    def pause_simulation(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.log("⏸ SIMULATION PAUSED", "orange")

    def reset_simulation(self):
        self.is_running = False
        self.step = 0
        self.route_index = 0
        self.current_route = []
        self.agent_pos = self.map.rescue_base

        # Reset victims
        for v in self.victims:
            v.rescued = False
            v.rescue_time = None

        # Reset KPIs
        self.kpis = KPITracker()
        self.kpis.victims_total = len(self.victims)

        self.draw_grid()
        self.update_kpi_display()
        self.log_text.delete(1.0, tk.END)
        self.log("↻ SIMULATION RESET", "blue")
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)

    def update_speed(self, val):
        self.animation_speed = int(val)

    # ── Agent Logic ────────────────────────────────────────────────────────────

    def run_agent(self):
        """Main agent loop — executes one rescue at a time."""
        if not self.is_running:
            return

        unrescued = [v for v in self.victims if not v.rescued]
        if not unrescued:
            self.log("═" * 50, "blue")
            self.log("✓ ALL VICTIMS RESCUED", "green")
            self.log("═" * 50, "blue")
            self.is_running = False
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.DISABLED)
            return

        # CSP allocation (once at start)
        if self.step == 0:
            self.log("Running CSP resource allocation...", "purple")
            assignment, bt = solve_csp(unrescued, self.resources)
            self.kpis.record_backtrack("with_heuristic", bt)
            self.log(f"✓ CSP complete (backtracks: {bt})", "green")

        # Prioritize victims
        victim = self.prioritize_next_victim(unrescued)
        self.step += 1

        self.log(f"\n[Step {self.step}] Rescuing V{victim.id} ({victim.severity.upper()})", "blue")

        # Dynamic events
        blocked = self.events.maybe_block_road(self.step, probability=0.2)
        if blocked:
            self.log(f"⚠ ROAD BLOCKED at {blocked} (aftershock)", "red")
            self.kpis.record_replan()

        # Route planning
        result, strategy = self.plan_route(victim)
        if not result.found:
            self.log(f"✗ V{victim.id} UNREACHABLE — skipping", "red")
            self.root.after(1000, self.run_agent)
            return

        self.log(f"Route: {len(result.path)-1} steps | Strategy: {strategy}", "purple")
        self.current_route = result.path
        self.route_index = 0

        # Animate movement along route
        self.animate_route(victim, result)

    def prioritize_next_victim(self, unrescued: List[Victim]) -> Victim:
        """Pick next victim: critical first, then by survival probability."""
        scored = []
        for v in unrescued:
            dist = abs(v.position[0] - self.map.medical_centers[0][0]) + \
                   abs(v.position[1] - self.map.medical_centers[0][1])
            risk_nearby = int(self.map.is_high_risk(v.position))
            feats = victim_features(v, dist, risk_nearby, self.step,
                                   self.resources.medical_kits)
            survival = self.knn.predict_proba(feats)
            scored.append((v, survival))
        scored.sort(key=lambda x: (-x[0].priority, x[1]))
        return scored[0][0]

    def plan_route(self, victim: Victim):
        """Use fuzzy logic to choose route strategy."""
        start = self.agent_pos
        goal = victim.position

        # Estimate risk
        blockage_prob = self.estimate_blockage_prob(start, goal)
        hazard_level = blockage_prob * 10
        score, label, action = fuzzy_decision(blockage_prob, hazard_level,
                                              victim.priority)

        self.log(f"Fuzzy risk: {score:.2f} [{label}]", "orange")

        # Choose A* variant
        if label in ("HIGH", "CRITICAL"):
            result = astar(self.map, start, goal, avoid_risk=True)
            strategy = "A*(safe)"
        else:
            result = astar(self.map, start, goal, avoid_risk=False)
            strategy = "A*(fast)"

        self.kpis.record_search(result)
        return result, strategy

    def estimate_blockage_prob(self, start, goal):
        r1, c1 = start
        r2, c2 = goal
        dist = abs(r1 - r2) + abs(c1 - c2)
        if dist == 0:
            return 0.0
        risk_count = sum(
            1 for t in range(dist + 1)
            if self.map.is_high_risk((
                r1 + round((r2 - r1) * t / dist),
                c1 + round((c2 - c1) * t / dist)
            ))
        )
        return min(risk_count / dist, 1.0)

    def animate_route(self, victim: Victim, result):
        """Move agent step-by-step along the route."""
        if not self.is_running or self.route_index >= len(self.current_route):
            # Route complete — mark rescued
            self.finalize_rescue(victim, result)
            return

        # Move to next cell
        self.agent_pos = self.current_route[self.route_index]
        self.route_index += 1
        self.draw_grid()
        self.root.after(self.animation_speed, lambda: self.animate_route(victim, result))

    def finalize_rescue(self, victim: Victim, result):
        """Mark victim rescued and continue to next."""
        victim.rescued = True
        victim.rescue_time = len(result.path) - 1
        self.kpis.record_rescue(victim, victim.rescue_time)

        if self.resources.use_kit():
            self.kpis.kits_used += 1

        self.kpis.ambulance_trips += 1

        self.log(f"✓ V{victim.id} RESCUED in {victim.rescue_time} steps", "green")
        self.current_route = []
        self.update_kpi_display()
        self.draw_grid()

        # Continue to next victim after brief pause
        self.root.after(800, self.run_agent)

    # ── KPI Display ────────────────────────────────────────────────────────────

    def update_kpi_display(self):
        saved = self.kpis.victims_saved
        total = self.kpis.victims_total
        avg_time = self.kpis.average_rescue_time()

        self.kpi_labels["Victims Saved"].config(text=f"{saved} / {total}")
        self.kpi_labels["Current Step"].config(text=str(self.step))
        self.kpi_labels["Avg Rescue Time"].config(text=f"{avg_time:.1f} steps")
        self.kpi_labels["Kits Used"].config(text=f"{self.kpis.kits_used} / 10")
        self.kpi_labels["Replan Events"].config(text=str(self.kpis.replan_count))


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main():
    from environment import build_default_scenario

    grid_map, victims, resources = build_default_scenario()
    event_sim = EventSimulator(grid_map, seed=42)

    root = tk.Tk()
    app = AIDRASimulatorGUI(root, grid_map, victims, resources, event_sim)
    root.mainloop()


if __name__ == "__main__":
    main()
