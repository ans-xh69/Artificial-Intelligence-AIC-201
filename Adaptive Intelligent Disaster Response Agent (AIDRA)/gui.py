"""
gui_simulator.py
Enhanced AIDRA GUI with decision log, algorithm comparison, speed control.
Fixes: Hill Climbing/Simulated Annealing routing, plot graph metrics.

Run: python gui_simulator.py
"""

import tkinter as tk
import numpy as np  # For simulated annealing
from tkinter import ttk, scrolledtext, messagebox
from typing import List, Tuple, Optional, Dict
import random
from environment import GridMap, Victim, Resources, EventSimulator, CELL_BLOCKED, CELL_HIGH_RISK
from search import bfs, dfs, greedy_best_first, astar, hill_climbing, SearchResult
from csp import solve_csp
from ml_model import train_and_evaluate, victim_features
from uncertainty import fuzzy_decision
from metrics import KPITracker
from visualizer import generate_all


# ── Dark Theme Colors ──────────────────────────────────────────────────────────
COLORS = {
    "bg_dark":    "#0d1117",
    "bg_panel":   "#161b22",
    "bg_input":   "#0d1117",
    "border":     "#30363d",
    "text":       "#c9d1d9",
    "text_dim":   "#8b949e",
    "accent":     "#58a6ff",
    "success":    "#3fb950",
    "warning":    "#d29922",
    "danger":     "#f85149",
    # Grid colors
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
        self.root.title("AIDRA — Disaster Response Simulator")
        self.root.geometry("1600x950")
        self.root.configure(bg=COLORS["bg_dark"])

        self.map = grid_map
        self.victims = victims
        self.victims_backup = [
            Victim(
                v.id,
                v.position,
                v.severity,
                patient_record=v.patient_record,
                picked_up=v.picked_up,
                rescued=v.rescued,
                rescue_time=v.rescue_time,
            )
            for v in victims
        ]
        self.resources = resources
        self.events = event_sim
        self.kpis = KPITracker()
        self.kpis.victims_total = len(victims)

        # Agent state
        self.step = 0
        self.current_route: List[Tuple[int, int]] = []
        self.agent_pos = grid_map.rescue_base
        self.animation_speed = 300
        self.is_running = False
        self.route_index = 0
        self.ml_trained = False
        self.knn, self.nb = None, None
        self.pending_victims: List[Victim] = []
        self.current_trip_victims: List[Victim] = []
        self.current_leg_type: str = ""
        self.current_leg_goal: Optional[Tuple[int, int]] = None
        self.current_leg_victim: Optional[Victim] = None
        self.current_trip_load: int = 0

        # Comparison data
        self.comparison_data: Dict[str, Dict] = {}
        self.last_run_kpis: Dict = {}

        # UI setup
        self.cell_size = 50
        self.setup_ui()
        self.draw_grid()
        self.update_kpi_display()
        self.log("System initialized. Select algorithm and press RUN.", COLORS["accent"])

    # ── UI Setup ───────────────────────────────────────────────────────────────

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg=COLORS["bg_dark"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left: Grid
        left_panel = tk.Frame(main_frame, bg=COLORS["bg_panel"], relief=tk.FLAT, bd=0)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        self.canvas = tk.Canvas(left_panel, width=self.map.cols * self.cell_size,
                                height=self.map.rows * self.cell_size,
                                bg="white", highlightthickness=0)
        self.canvas.pack(padx=15, pady=15)

        # Right: Controls + Tables + Log
        right_panel = tk.Frame(main_frame, bg=COLORS["bg_dark"])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.setup_controls(right_panel)
        self.setup_last_run_kpis(right_panel)
        self.setup_comparison_table(right_panel)
        self.setup_decision_log(right_panel)
        self.setup_ml_section(right_panel)

    def setup_controls(self, parent):
        frame = tk.LabelFrame(parent, text="Controls", bg=COLORS["bg_panel"],
                             fg=COLORS["text"], font=("Consolas", 10, "bold"),
                             bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Row 0: Algorithm + ML Status
        tk.Label(frame, text="Algorithm:", bg=COLORS["bg_panel"],
                fg=COLORS["text"], font=("Consolas", 9)).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.algo_var = tk.StringVar(value="A*")
        algo_dropdown = ttk.Combobox(frame, textvariable=self.algo_var,
            values=["BFS", "DFS", "Greedy", "A*", "Hill Climbing", "Sim. Annealing"],
            state="readonly", width=18, font=("Consolas", 9))
        algo_dropdown.grid(row=0, column=1, padx=5, pady=5)

        self.ml_status = tk.Label(frame, text="● NOT TRAINED", bg=COLORS["bg_panel"],
                                 fg=COLORS["danger"], font=("Consolas", 9, "bold"))
        self.ml_status.grid(row=0, column=2, padx=10)

        # Row 1: Risk Mode
        tk.Label(frame, text="Risk Mode:", bg=COLORS["bg_panel"],
                fg=COLORS["text"], font=("Consolas", 9)).grid(row=1, column=0, sticky="w", padx=10, pady=5)

        self.risk_mode = tk.StringVar(value="Balanced")
        risk_frame = tk.Frame(frame, bg=COLORS["bg_panel"])
        risk_frame.grid(row=1, column=1, columnspan=2, sticky="w", padx=5)

        for mode in ["Fast", "Balanced", "Safe"]:
            rb = tk.Radiobutton(risk_frame, text=mode, variable=self.risk_mode,
                               value=mode, bg=COLORS["bg_panel"], fg=COLORS["text"],
                               selectcolor=COLORS["bg_input"], font=("Consolas", 9),
                               activebackground=COLORS["bg_panel"])
            rb.pack(side=tk.LEFT, padx=5)

        # Row 2: Buttons
        btn_frame = tk.Frame(frame, bg=COLORS["bg_panel"])
        btn_frame.grid(row=2, column=0, columnspan=3, pady=10)

        self.run_btn = tk.Button(btn_frame, text="▶ RUN", command=self.start_simulation,
                                bg=COLORS["success"], fg="white", font=("Consolas", 10, "bold"),
                                width=10, cursor="hand2", relief=tk.FLAT)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(btn_frame, text="⟲ RESET", command=self.reset_simulation,
                                   bg=COLORS["danger"], fg="white", font=("Consolas", 10, "bold"),
                                   width=10, cursor="hand2", relief=tk.FLAT)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.train_btn = tk.Button(btn_frame, text="🧠 TRAIN ML", command=self.train_ml_models,
                                   bg=COLORS["accent"], fg="white", font=("Consolas", 10, "bold"),
                                   width=12, cursor="hand2", relief=tk.FLAT)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.pdf_btn = tk.Button(btn_frame, text="PDFs", command=self.generate_pdf_reports,
                                 bg=COLORS["warning"], fg="white", font=("Consolas", 10, "bold"),
                                 width=10, cursor="hand2", relief=tk.FLAT)
        self.pdf_btn.pack(side=tk.LEFT, padx=5)

        # Row 3: Speed control
        tk.Label(frame, text="Speed:", bg=COLORS["bg_panel"],
                fg=COLORS["text"], font=("Consolas", 9)).grid(row=3, column=0, sticky="w", padx=10, pady=5)

        self.speed_var = tk.IntVar(value=300)
        speed_slider = tk.Scale(frame, from_=50, to=1000, orient=tk.HORIZONTAL,
                               variable=self.speed_var, command=self.update_speed,
                               bg=COLORS["bg_panel"], fg=COLORS["text"],
                               highlightthickness=0, length=250, troughcolor=COLORS["bg_input"])
        speed_slider.grid(row=3, column=1, columnspan=2, sticky="w", padx=5)

        # Row 4: Info
        info = tk.Label(frame, text="ⓘ Switch algorithm freely — no reset needed.",
                       bg=COLORS["bg_panel"], fg=COLORS["text_dim"], font=("Consolas", 8))
        info.grid(row=4, column=0, columnspan=3, pady=(0, 5))

    def setup_last_run_kpis(self, parent):
        frame = tk.LabelFrame(parent, text="Last Run — KPIs", bg=COLORS["bg_panel"],
                             fg=COLORS["text"], font=("Consolas", 10, "bold"), bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.X, pady=(0, 10))

        self.kpi_labels = {}
        kpis = [
            ("Victims Saved:", "—"),
            ("Avg Rescue Time:", "—"),
            ("Risk Exposure:", "—"),
            ("Resource Util:", "—"),
            ("CSP Backtracks:", "—"),
        ]
        for i, (key, val) in enumerate(kpis):
            tk.Label(frame, text=key, bg=COLORS["bg_panel"], fg=COLORS["text_dim"],
                    font=("Consolas", 9), anchor="w").grid(row=i, column=0, sticky="w", padx=10, pady=3)
            lbl = tk.Label(frame, text=val, bg=COLORS["bg_panel"], fg=COLORS["accent"],
                          font=("Consolas", 9, "bold"), anchor="e")
            lbl.grid(row=i, column=1, sticky="e", padx=10, pady=3)
            self.kpi_labels[key] = lbl

    def setup_comparison_table(self, parent):
        frame = tk.LabelFrame(parent, text="Algorithm Comparison", bg=COLORS["bg_panel"],
                             fg=COLORS["text"], font=("Consolas", 10, "bold"), bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        # Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=COLORS["bg_input"], foreground=COLORS["text"],
                       fieldbackground=COLORS["bg_input"], font=("Consolas", 9))
        style.configure("Treeview.Heading", background=COLORS["bg_panel"],
                       foreground=COLORS["warning"], font=("Consolas", 9, "bold"))
        style.map("Treeview", background=[("selected", COLORS["accent"])])

        cols = ("Saved", "AvgTime", "Risk", "Nodes", "BT")
        self.comp_table = ttk.Treeview(frame, columns=cols, height=6, show="tree headings")
        self.comp_table.heading("#0", text="Algorithm")
        self.comp_table.column("#0", width=120, anchor="w")
        for col in cols:
            self.comp_table.heading(col, text=col)
            self.comp_table.column(col, width=60, anchor="center")
        self.comp_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Buttons
        btn_frame = tk.Frame(frame, bg=COLORS["bg_panel"])
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Button(btn_frame, text="📊 Plot Graph", command=self.plot_comparison,
                 bg=COLORS["accent"], fg="white", font=("Consolas", 9, "bold"),
                 relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="✖ Clear", command=self.clear_table,
                 bg=COLORS["danger"], fg="white", font=("Consolas", 9, "bold"),
                 relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=5)

    def setup_decision_log(self, parent):
        frame = tk.LabelFrame(parent, text="Decision Log", bg=COLORS["bg_panel"],
                             fg=COLORS["text"], font=("Consolas", 10, "bold"), bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(frame, height=12, width=50,
                                                  font=("Consolas", 9), bg=COLORS["bg_input"],
                                                  fg=COLORS["text"], wrap=tk.WORD,
                                                  insertbackground=COLORS["text"])
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure tags for colored text
        self.log_text.tag_config("accent", foreground=COLORS["accent"])
        self.log_text.tag_config("success", foreground=COLORS["success"])
        self.log_text.tag_config("warning", foreground=COLORS["warning"])
        self.log_text.tag_config("danger", foreground=COLORS["danger"])

    def setup_ml_section(self, parent):
        frame = tk.LabelFrame(parent, text="ML Metrics (kNN | Naive Bayes)", bg=COLORS["bg_panel"],
                             fg=COLORS["text"], font=("Consolas", 10, "bold"), bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.X)

        self.ml_info = tk.Label(frame, text="Auto-trains on first RUN, or press 🧠 TRAIN ML.",
                               bg=COLORS["bg_panel"], fg=COLORS["success"],
                               font=("Consolas", 9), wraplength=400, justify="left")
        self.ml_info.pack(padx=10, pady=10)

    # ── Grid Drawing ───────────────────────────────────────────────────────────

    def draw_grid(self):
        self.canvas.delete("all")
        cs = self.cell_size

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

        # Route
        if self.current_route and len(self.current_route) > 1:
            for i in range(len(self.current_route) - 1):
                r1, c1 = self.current_route[i]
                r2, c2 = self.current_route[i + 1]
                x1, y1 = c1 * cs + cs // 2, r1 * cs + cs // 2
                x2, y2 = c2 * cs + cs // 2, r2 * cs + cs // 2
                self.canvas.create_line(x1, y1, x2, y2, fill=COLORS["route"],
                                      width=4, dash=(5, 3))

        # Base
        br, bc = self.map.rescue_base
        self.draw_circle(br, bc, COLORS["base"], "B")

        # Centers
        for mr, mc in self.map.medical_centers:
            self.draw_circle(mr, mc, COLORS["center"], "M")

        # Victims
        for v in self.victims:
            vr, vc = v.position
            if v.rescued:
                self.draw_circle(vr, vc, COLORS["rescued"], "✓", "white")
            else:
                self.draw_circle(vr, vc, COLORS[v.severity], f"V{v.id}")

        # Agent
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

    def log(self, msg: str, color_tag: str = None):
        if color_tag:
            # Extract color name from hex code
            tag_name = {v: k for k, v in COLORS.items()}.get(color_tag, "accent")
            self.log_text.insert(tk.END, f"{msg}\n", tag_name)
        else:
            self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.log_text.update()

    # ── Controls ───────────────────────────────────────────────────────────────

    def update_speed(self, val):
        self.animation_speed = int(val)

    def train_ml_models(self):
        if self.ml_trained:
            messagebox.showinfo("Already Trained", "ML models are already trained.")
            return
        self.log("Training ML models on the KTAS triage dataset...", COLORS["warning"])
        self.knn, self.nb, knn_m, nb_m = train_and_evaluate()
        self.kpis.record_ml("kNN", knn_m)
        self.kpis.record_ml("Naive Bayes", nb_m)
        self.ml_trained = True
        self.ml_status.config(text="✓ TRAINED", fg=COLORS["success"])
        self.train_btn.config(state=tk.DISABLED)
        self.ml_info.config(text=f"kNN Acc: {knn_m['accuracy']:.3f} | NB Acc: {nb_m['accuracy']:.3f}",
                           fg=COLORS["accent"])
        self.log("✓ ML training complete.", COLORS["success"])

    def prioritize_victims(self, unrescued):
        scored = []
        for v in unrescued:
            dist = abs(v.position[0] - self.map.medical_centers[0][0]) + \
                   abs(v.position[1] - self.map.medical_centers[0][1])
            risk_nearby = int(self.map.is_high_risk(v.position))
            feats = victim_features(v, dist, risk_nearby, self.step,
                                   self.resources.medical_kits)
            urgency = self.knn.predict_proba(feats)
            scored.append((v, urgency))
        scored.sort(key=lambda x: (-x[0].priority, -x[1]))
        return [v for v, _ in scored]

    def _nearest_center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        return min(
            self.map.medical_centers,
            key=lambda c: abs(pos[0] - c[0]) + abs(pos[1] - c[1]),
        )

    def start_simulation(self):
        if self.is_running:
            return

        # Auto-train if needed
        if not self.ml_trained:
            self.train_ml_models()

        self.pending_victims = self.prioritize_victims([v for v in self.victims if not v.rescued])
        self.current_trip_victims = []
        self.current_leg_type = ""
        self.current_leg_goal = None
        self.current_leg_victim = None
        self.current_trip_load = 0
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.log("═" * 50, COLORS["accent"])
        self.log(f"▶ SIMULATION STARTED ({self.algo_var.get()})", COLORS["success"])
        self.log("═" * 50, COLORS["accent"])
        self.run_agent()

    def reset_simulation(self):
        self.is_running = False
        self.step = 0
        self.route_index = 0
        self.current_route = []
        self.agent_pos = self.map.rescue_base
        self.pending_victims = []
        self.current_trip_victims = []
        self.current_leg_type = ""
        self.current_leg_goal = None
        self.current_leg_victim = None
        self.current_trip_load = 0

        # Reset victims
        self.victims = [
            Victim(
                v.id,
                v.position,
                v.severity,
                patient_record=v.patient_record,
                picked_up=v.picked_up,
                rescued=v.rescued,
                rescue_time=v.rescue_time,
            )
            for v in self.victims_backup
        ]

        # Reset KPIs
        self.kpis = KPITracker()
        self.kpis.victims_total = len(self.victims)

        self.draw_grid()
        self.update_kpi_display()
        self.run_btn.config(state=tk.NORMAL)
        self.log("↻ Simulation reset.", COLORS["accent"])

    def clear_table(self):
        for item in self.comp_table.get_children():
            self.comp_table.delete(item)
        self.comparison_data = {}
        self.log("✖ Comparison table cleared.", COLORS["warning"])

    def plot_comparison(self):
        if not self.comparison_data:
            messagebox.showwarning("No Data", "Run at least one simulation first.")
            return
        
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            algos = list(self.comparison_data.keys())
            avg_nodes = [float(self.comparison_data[a]["avg_nodes"]) for a in algos]
            avg_time = [float(self.comparison_data[a]["time"]) for a in algos]
            risk_exp = [int(self.comparison_data[a]["risk"]) for a in algos]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle("Algorithm Comparison", fontsize=14, fontweight="bold")

            colors = plt.cm.Set2(np.linspace(0, 0.9, len(algos)))

            # Plot 1: Avg Nodes Expanded
            axes[0].bar(range(len(algos)), avg_nodes, color=colors, edgecolor="white")
            axes[0].set_xticks(range(len(algos)))
            axes[0].set_xticklabels(algos, rotation=35, ha="right")
            axes[0].set_ylabel("Avg Nodes Expanded")
            axes[0].set_title("Search Efficiency\n(fewer = better)")
            axes[0].grid(axis="y", alpha=0.4)

            # Plot 2: Avg Rescue Time
            axes[1].bar(range(len(algos)), avg_time, color=colors, edgecolor="white")
            axes[1].set_xticks(range(len(algos)))
            axes[1].set_xticklabels(algos, rotation=35, ha="right")
            axes[1].set_ylabel("Avg Rescue Time (steps)")
            axes[1].set_title("Rescue Speed\n(lower = faster)")
            axes[1].grid(axis="y", alpha=0.4)

            # Plot 3: Risk Exposure
            axes[2].bar(range(len(algos)), risk_exp, color=colors, edgecolor="white")
            axes[2].set_xticks(range(len(algos)))
            axes[2].set_xticklabels(algos, rotation=35, ha="right")
            axes[2].set_ylabel("Risk Exposure (hazard steps)")
            axes[2].set_title("Safety\n(fewer = safer)")
            axes[2].grid(axis="y", alpha=0.4)

            plt.tight_layout()
            plt.savefig("figures/gui_comparison.png", dpi=150, bbox_inches="tight")
            plt.savefig("figures/gui_comparison.pdf", dpi=150, bbox_inches="tight")
            plt.close()

            messagebox.showinfo("Success", "Chart saved to figures/gui_comparison.png")
            self.log("📊 Comparison chart generated.", COLORS["success"])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot: {e}")
            self.log(f"✗ Plot error: {e}", COLORS["danger"])

    # ── Agent Logic ────────────────────────────────────────────────────────────

    def generate_pdf_reports(self):
        if not self.ml_trained:
            self.train_ml_models()

        if not self.kpis.search_results:
            messagebox.showwarning("No Results", "Run the simulation first so PDF reports can be generated.")
            return

        try:
            self.log("Generating full PDF and PNG report set...", COLORS["warning"])
            generate_all(
                self.map,
                self.victims,
                self.kpis,
                comparison_results=self.kpis.search_results,
                knn_model=self.knn,
                nb_model=self.nb,
                victim_routes=None,
            )
            messagebox.showinfo("Exported", "All report figures were saved as PNG and PDF in figures/.")
            self.log("✓ PDF/PNG reports generated.", COLORS["success"])
        except Exception as e:
            messagebox.showerror("PDF Export Failed", f"Could not generate reports: {e}")
            self.log(f"✗ PDF generation failed: {e}", COLORS["danger"])

    def run_agent(self):
        if not self.is_running:
            return

        unrescued = [v for v in self.victims if not v.rescued]
        if not unrescued:
            self.log("═" * 50, COLORS["accent"])
            self.log("✓ ALL VICTIMS RESCUED", COLORS["success"])
            self.log("═" * 50, COLORS["accent"])
            self.is_running = False
            self.save_run_results()
            return

        # CSP (once)
        if self.step == 0:
            self.log("Running CSP resource allocation...", COLORS["warning"])
            assignment, bt = solve_csp(unrescued, self.resources)
            self.kpis.record_backtrack("with_heuristic", bt)
            self.log(f"✓ CSP complete (backtracks: {bt})", COLORS["success"])
        if not self.current_trip_victims and self.pending_victims:
            self.current_trip_victims = self.pending_victims[:2]
            self.pending_victims = self.pending_victims[2:]
            self.current_trip_load = 0
            self.log(
                "Trip load: " + ", ".join(f"V{v.id}({v.severity.upper()})" for v in self.current_trip_victims),
                COLORS["warning"],
            )

        if not self.current_trip_victims:
            self.is_running = False
            self.run_btn.config(state=tk.NORMAL)
            self.save_run_results()
            return

        if self.current_trip_load < len(self.current_trip_victims):
            victim = self.current_trip_victims[self.current_trip_load]
            self.step += 1
            self.log(f"\n[Step {self.step}] Picking up V{victim.id} ({victim.severity.upper()})", COLORS["accent"])
            result = self.plan_route_to_goal(victim.position, victim)
            if not result.found:
                self.log(f"✗ V{victim.id} UNREACHABLE — skipping", COLORS["danger"])
                self.current_trip_load += 1
                self.root.after(300, self.run_agent)
                return
            self.current_leg_type = "victim"
            self.current_leg_goal = victim.position
            self.current_leg_victim = victim
            self.current_route = result.path
            self.route_index = 0
            self.animate_route()
            return

        if not any(v.picked_up for v in self.current_trip_victims):
            self.current_trip_victims = []
            self.current_trip_load = 0
            self.root.after(300, self.run_agent)
            return

        self.step += 1
        center = self._nearest_center(self.agent_pos)
        self.log(f"\n[Step {self.step}] Delivering batch to medical center {center}", COLORS["accent"])
        result = self.plan_route_to_goal(center, None)
        if not result.found:
            self.log("✗ No valid route to medical center — stopping batch", COLORS["danger"])
            self.is_running = False
            self.run_btn.config(state=tk.NORMAL)
            self.save_run_results()
            return
        self.current_leg_type = "center"
        self.current_leg_goal = center
        self.current_leg_victim = None
        self.current_route = result.path
        self.route_index = 0
        self.animate_route()

    def plan_route_to_goal(self, goal, victim=None) -> SearchResult:
        start = self.agent_pos
        algo = self.algo_var.get()
        risk_mode = self.risk_mode.get()

        avoid_risk = (risk_mode == "Safe")

        # Select algorithm
        if algo == "A*":
            result = astar(self.map, start, goal, avoid_risk)
        elif algo == "BFS":
            result = bfs(self.map, start, goal)
        elif algo == "DFS":
            result = dfs(self.map, start, goal)
        elif algo == "Greedy":
            result = greedy_best_first(self.map, start, goal)
        elif algo == "Hill Climbing":
            result = hill_climbing(self.map, start, goal, max_restarts=10)
        else:  # Sim. Annealing
            result = self.simulated_annealing(start, goal)

        self.kpis.record_search(result)
        self.log(f"Algorithm: {algo} ({'safe' if avoid_risk else 'fast'} mode)", COLORS["warning"])
        return result

    def simulated_annealing(self, start, goal, max_iter=500) -> SearchResult:
        """Simulated Annealing for path finding (fixed version)."""
        current = start
        current_path = [current]
        temperature = 100.0
        cooling_rate = 0.95
        expanded = 0

        for iteration in range(max_iter):
            if current == goal:
                return SearchResult("Sim. Annealing", current_path, expanded,
                                   len(current_path) - 1,
                                   sum(1 for p in current_path if self.map.is_high_risk(p)))

            neighbors = self.map.neighbors(current)
            if not neighbors:
                break

            # Pick random neighbor
            next_pos = random.choice(neighbors)
            expanded += 1

            # Cost = distance to goal (lower is better)
            current_cost = abs(current[0] - goal[0]) + abs(current[1] - goal[1])
            next_cost = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
            delta = next_cost - current_cost

            # Accept if better, or probabilistically if worse
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                if next_pos not in current_path:  # Avoid cycles
                    current = next_pos
                    current_path.append(current)

            temperature *= cooling_rate

            if temperature < 0.1:
                break

        # If didn't reach goal, use A* as fallback
        return astar(self.map, start, goal, avoid_risk=False)

    def animate_route(self):
        if not self.is_running or self.route_index >= len(self.current_route):
            self.finalize_current_leg()
            return

        self.agent_pos = self.current_route[self.route_index]
        self.route_index += 1
        self.draw_grid()
        self.root.after(self.animation_speed, self.animate_route)

    def finalize_current_leg(self):
        if self.current_leg_type == "victim" and self.current_leg_victim is not None:
            victim = self.current_leg_victim
            victim.picked_up = True
            if self.resources.use_kit():
                self.kpis.kits_used += 1
            self.current_trip_load += 1
            self.agent_pos = victim.position
            self.log(f"✓ V{victim.id} PICKED UP ({victim.severity.upper()})", COLORS["success"])
            self.draw_grid()

            if self.current_trip_load < len(self.current_trip_victims):
                self.root.after(300, self.run_agent)
                return

            self.root.after(300, self.run_agent)
            return

        if self.current_leg_type == "center" and self.current_leg_goal is not None:
            delivered = [v for v in self.current_trip_victims if v.picked_up]
            trip_time = len(self.current_route) - 1
            for victim in delivered:
                victim.rescue_time = trip_time
                self.kpis.record_rescue(victim, trip_time)
                self.log(f"✓ V{victim.id} DELIVERED to medical center in {trip_time} steps", COLORS["success"])

            self.kpis.ambulance_trips += 1
            self.agent_pos = self.current_leg_goal
            for victim in self.current_trip_victims:
                victim.rescued = True
                victim.picked_up = False
            self.update_kpi_display()
            self.draw_grid()
            self.current_trip_victims = []
            self.current_trip_load = 0
            self.current_leg_type = ""
            self.current_leg_goal = None
            self.current_leg_victim = None
            self.root.after(300, self.run_agent)
            return

        self.root.after(300, self.run_agent)

    def save_run_results(self):
        algo = self.algo_var.get()
        saved = self.kpis.victims_saved
        avg_time = self.kpis.average_rescue_time()
        risk = sum(r.risk_cells for r in self.kpis.search_results if r.found)
        total_nodes = sum(r.nodes_expanded for r in self.kpis.search_results if r.found)
        avg_nodes = total_nodes / len([r for r in self.kpis.search_results if r.found]) if self.kpis.search_results else 0
        bt = self.kpis.backtrack_counts.get("with_heuristic", 0)

        # Save to comparison
        self.comparison_data[algo] = {
            "saved": f"{saved}/5",
            "time": f"{avg_time:.1f}",
            "risk": str(risk),
            "nodes": str(total_nodes),
            "avg_nodes": f"{avg_nodes:.1f}",
            "bt": str(bt),
        }

        # Update table
        self.update_comparison_table()

        # Update last run KPIs
        self.last_run_kpis = {
            "Victims Saved:": f"{saved} / 5",
            "Avg Rescue Time:": f"{avg_time:.1f}",
            "Risk Exposure:": str(risk),
            "Resource Util:": f"{self.kpis.kits_used}/10 kits",
            "CSP Backtracks:": str(bt),
        }
        for key, val in self.last_run_kpis.items():
            self.kpi_labels[key].config(text=val)

        self.log(f"\nRun saved: {saved}/5 victims | Avg time: {avg_time:.1f} | Risk: {risk}", COLORS["accent"])

    def update_comparison_table(self):
        for item in self.comp_table.get_children():
            self.comp_table.delete(item)
        for algo, data in sorted(self.comparison_data.items()):
            self.comp_table.insert("", "end", text=algo,
                values=(data["saved"], data["time"], data["risk"],
                       data["nodes"], data["bt"]))

    def update_kpi_display(self):
        pass

# ── Entry Point ────────────────────────────────────────────────────────────────

def launch_gui(grid_map, victims, resources, event_sim):
    root = tk.Tk()

    app = AIDRASimulatorGUI(
        root,
        grid_map,
        victims,
        resources,
        event_sim
    )

    root.mainloop()


if __name__ == "__main__":
    from environment import build_default_scenario

    grid_map, victims, resources = build_default_scenario()
    event_sim = EventSimulator(grid_map, seed=42)

    launch_gui(grid_map, victims, resources, event_sim)
