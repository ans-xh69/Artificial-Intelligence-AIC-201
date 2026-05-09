"""
visualizer.py
All Matplotlib visualizations for AIDRA.
Uses NumPy for data arrays and Matplotlib for all plots.

Figures generated:
  1.  grid_initial.png        — initial disaster scenario map
  2.  grid_final.png          — final state with rescued victims
  3.  grid_route_V{id}.png    — per-victim A* route overlay
  4.  search_comparison.png   — BFS/DFS/Greedy/A* bar charts (3 metrics)
  5.  cm_{model}.png          — confusion matrix heatmap per ML model
  6.  ml_comparison.png       — grouped bar: accuracy/precision/recall/F1
  7.  ml_survival_scatter.png — survival probability curves by severity
  8.  rescue_timeline.png     — horizontal bar chart of rescue times
  9.  kpi_dashboard.png       — 6-panel summary dashboard
  10. fuzzy_surface.png       — 3-D fuzzy risk score surface
  11. risk_heatmap.png        — agent risk heatmap over the grid
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works in any terminal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple

from environment import GridMap, Victim, CELL_BLOCKED, CELL_HIGH_RISK
from search import SearchResult
from uncertainty import compute_risk_score

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "free":      "#f0f4f8",
    "blocked":   "#2d3748",
    "high_risk": "#fc8181",
    "base":      "#4299e1",
    "center":    "#48bb78",
    "crit":      "#e53e3e",
    "mod":       "#ed8936",
    "minor":     "#ecc94b",
    "route":     "#805ad5",
    "bg":        "#ffffff",
    "panel_bg":  "#f7fafc",
}
SEV_COLOR = {"critical": C["crit"], "moderate": C["mod"], "minor": C["minor"]}


def _save(fig, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Saved: {path}")
    return path


# ── 1 & 2. Grid map ────────────────────────────────────────────────────────────

def plot_grid(grid_map: GridMap,
              victims: List[Victim],
              route: Optional[List[Tuple[int, int]]] = None,
              title: str = "AIDRA — Disaster Grid",
              filename: str = "grid_map.png",
              agent_pos: Optional[Tuple[int, int]] = None) -> str:

    rows, cols = grid_map.rows, grid_map.cols
    grid = np.array(grid_map.grid, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(C["bg"])

    cmap = ListedColormap([C["free"], C["blocked"], C["high_risk"]])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, origin="upper", aspect="equal")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#cbd5e0", linewidth=0.5)
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols)); ax.set_yticklabels(range(rows))

    # Rescue base
    br, bc = grid_map.rescue_base
    ax.add_patch(plt.Circle((bc, br), 0.38, color=C["base"], zorder=4))
    ax.text(bc, br, "B", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=5)

    # Medical centers
    for (mr, mc) in grid_map.medical_centers:
        ax.add_patch(plt.Circle((mc, mr), 0.38, color=C["center"], zorder=4))
        ax.text(mc, mr, "M", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=5)

    # Route overlay
    if route and len(route) > 1:
        ry = np.array([p[0] for p in route])
        rx = np.array([p[1] for p in route])
        ax.plot(rx, ry, color=C["route"], linewidth=2.5,
                zorder=3, alpha=0.85, linestyle="--")
        ax.plot(rx[0], ry[0], "o", color=C["route"], markersize=7, zorder=4)
        ax.plot(rx[-1], ry[-1], "*", color=C["route"], markersize=13, zorder=4)

    # Victims
    for v in victims:
        vr, vc = v.position
        color  = SEV_COLOR[v.severity]
        marker = "X" if v.rescued else "o"
        ax.plot(vc, vr, marker, markersize=14, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=6)
        ax.text(vc + 0.42, vr - 0.42, f"V{v.id}",
                fontsize=7.5, color="#2d3748", fontweight="bold", zorder=7)

    if agent_pos:
        ar, ac = agent_pos
        ax.plot(ac, ar, "D", markersize=11, color="#f6e05e",
                markeredgecolor="#744210", markeredgewidth=1.5, zorder=7)

    legend_elements = [
        mpatches.Patch(facecolor=C["free"],      label="Free cell"),
        mpatches.Patch(facecolor=C["blocked"],   label="Blocked road"),
        mpatches.Patch(facecolor=C["high_risk"], label="High-risk zone"),
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=C["crit"],  markersize=10, label="Critical victim"),
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=C["mod"],   markersize=10, label="Moderate victim"),
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=C["minor"], markersize=10, label="Minor victim"),
        mpatches.Patch(facecolor=C["base"],   label="Rescue base"),
        mpatches.Patch(facecolor=C["center"], label="Medical center"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=8, framealpha=0.92, edgecolor="#e2e8f0")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    plt.tight_layout()
    return _save(fig, filename)


# ── 3. Per-victim route overlays ──────────────────────────────────────────────

def plot_victim_routes(grid_map: GridMap, victims: List[Victim],
                       routes: Dict[int, List[Tuple[int, int]]]):
    for vid, route in routes.items():
        v = next((x for x in victims if x.id == vid), None)
        if v:
            plot_grid(grid_map, victims, route=route,
                      title=f"Route — V{vid} ({v.severity})",
                      filename=f"grid_route_V{vid}.png")


# ── 4. Search algorithm comparison ────────────────────────────────────────────

def plot_search_comparison(search_results: List[SearchResult],
                           filename: str = "search_comparison.png") -> str:
    seen: Dict[str, SearchResult] = {}
    for r in search_results:
        if r.algorithm not in seen and r.found:
            seen[r.algorithm] = r
    if not seen:
        return ""

    algos      = list(seen.keys())
    costs      = np.array([seen[a].cost           for a in algos])
    nodes      = np.array([seen[a].nodes_expanded for a in algos])
    risk_cells = np.array([seen[a].risk_cells     for a in algos])
    x          = np.arange(len(algos))
    palette    = plt.cm.Set2(np.linspace(0, 0.9, len(algos)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Search Algorithm Comparison", fontsize=14, fontweight="bold")

    for ax, vals, ylabel, title_ in zip(
        axes,
        [costs, nodes, risk_cells],
        ["Path Cost (steps)", "Nodes Expanded", "High-Risk Cells on Path"],
        ["Path Cost", "Search Efficiency\n(fewer nodes = better)",
         "Risk Exposure\n(fewer = safer)"]
    ):
        bars = ax.bar(x, vals, color=palette, edgecolor="white",
                      linewidth=0.8, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title_, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        ax.set_facecolor(C["panel_bg"])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        # Highlight best (minimum)
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor("#2d3748")
        bars[best_idx].set_linewidth(2.2)

    plt.tight_layout()
    return _save(fig, filename)


# ── 5. Confusion matrix heatmap ───────────────────────────────────────────────

def plot_confusion_matrix(metrics: Dict, model_name: str,
                          filename: Optional[str] = None) -> str:
    cm = np.array([
        [metrics["tp"], metrics["fp"]],
        [metrics["fn"], metrics["tn"]],
    ])
    cell_labels = np.array([["TP", "FP"], ["FN", "TN"]])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Actual Positive", "Actual Negative"], fontsize=10)
    ax.set_yticklabels(["Predicted Positive", "Predicted Negative"], fontsize=10)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()

    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "#2d3748"
            ax.text(j, i, f"{cell_labels[i, j]}\n{cm[i, j]}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold", color=text_color)

    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontsize=12, fontweight="bold", pad=18)
    plt.tight_layout()
    fname = filename or f"cm_{model_name.replace(' ', '_')}.png"
    return _save(fig, fname)


# ── 6. ML metrics grouped bar ─────────────────────────────────────────────────

def plot_ml_comparison(ml_metrics: Dict,
                       filename: str = "ml_comparison.png") -> str:
    if not ml_metrics:
        return ""
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    models      = list(ml_metrics.keys())
    x           = np.arange(len(metric_keys))
    width       = 0.35
    colors      = ["#4299e1", "#48bb78"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (model, md) in enumerate(ml_metrics.items()):
        vals   = np.array([md[k] for k in metric_keys])
        offset = (i - len(models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=model,
                        color=colors[i % len(colors)],
                        edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_keys], fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("ML Model Performance Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.92)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.set_facecolor(C["panel_bg"])
    plt.tight_layout()
    return _save(fig, filename)


# ── 7. Survival probability curves ────────────────────────────────────────────

def plot_survival_scatter(knn_model, nb_model,
                          filename: str = "ml_survival_scatter.png") -> str:
    """
    Plots survival probability vs distance for each severity level.
    Uses NumPy linspace to generate smooth evaluation curves.
    """
    distances  = np.linspace(1, 14, 60)
    severities = [1, 2, 3]
    sev_labels = {1: "Minor", 2: "Moderate", 3: "Critical"}
    sev_colors = {1: C["minor"], 2: C["mod"], 3: C["crit"]}
    linestyles = {1: ":", 2: "--", 3: "-"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Survival Probability vs Distance to Medical Center",
                 fontsize=13, fontweight="bold")

    for ax, model, model_name in zip(axes,
                                     [knn_model, nb_model],
                                     ["kNN (k=5)", "Gaussian Naive Bayes"]):
        for sev in severities:
            probs = np.array([
                model.predict_proba(np.array([float(sev), d, 0, 10.0, 5.0]))
                for d in distances
            ])
            ax.plot(distances, probs,
                    color=sev_colors[sev],
                    linestyle=linestyles[sev],
                    linewidth=2.2,
                    label=sev_labels[sev])

        ax.set_xlabel("Distance to Medical Center (steps)", fontsize=10)
        ax.set_ylabel("Survival Probability", fontsize=10)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(0.5, color="#a0aec0", linestyle="--",
                   linewidth=1, label="Decision boundary")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.4, linestyle="--")
        ax.set_facecolor(C["panel_bg"])

    plt.tight_layout()
    return _save(fig, filename)


# ── 8. Rescue timeline ────────────────────────────────────────────────────────

def plot_rescue_timeline(victims: List[Victim],
                         filename: str = "rescue_timeline.png") -> str:
    rescued = [v for v in victims if v.rescued and v.rescue_time is not None]
    if not rescued:
        return ""

    labels = [f"V{v.id}  ({v.severity[:3].upper()})" for v in rescued]
    times  = np.array([v.rescue_time for v in rescued], dtype=float)
    colors = [SEV_COLOR[v.severity] for v in rescued]
    avg    = float(np.mean(times))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, times, color=colors, edgecolor="white", height=0.52)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.25,
                bar.get_y() + bar.get_height() / 2,
                f"{t:.0f} steps", va="center", fontsize=10)

    ax.axvline(avg, color="#2d3748", linestyle="--",
               linewidth=1.8, label=f"Average: {avg:.1f} steps")
    ax.set_xlabel("Rescue Time (steps)", fontsize=11)
    ax.set_title("Rescue Timeline by Victim", fontsize=13, fontweight="bold")
    ax.set_facecolor(C["panel_bg"])
    ax.grid(axis="x", alpha=0.4, linestyle="--")
    ax.set_xlim(0, float(times.max()) * 1.15)

    patches = [mpatches.Patch(color=SEV_COLOR[s], label=s.capitalize())
               for s in ["critical", "moderate", "minor"]]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color="#2d3748", linestyle="--",
                   label=f"Avg: {avg:.1f} steps")
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    return _save(fig, filename)


# ── 9. KPI dashboard ──────────────────────────────────────────────────────────

def plot_kpi_dashboard(kpi_tracker, victims: List[Victim],
                       filename: str = "kpi_dashboard.png") -> str:
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#edf2f7")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    # Panel 1: Victims saved donut
    ax1 = fig.add_subplot(gs[0, 0])
    saved   = kpi_tracker.victims_saved
    unsaved = max(kpi_tracker.victims_total - saved, 0)
    wd = [saved, unsaved] if unsaved > 0 else [saved]
    wl = ["Saved", "Unsaved"] if unsaved > 0 else ["All Saved"]
    wc = ["#48bb78", "#fc8181"] if unsaved > 0 else ["#48bb78"]
    ax1.pie(wd, labels=wl, colors=wc, autopct="%1.0f%%",
            startangle=90, wedgeprops=dict(width=0.55, edgecolor="white"))
    ax1.set_title(f"Victims Saved ({saved}/{kpi_tracker.victims_total})",
                  fontweight="bold")

    # Panel 2: Resource utilization
    ax2 = fig.add_subplot(gs[0, 1])
    res_labels = ["Kits Used", "Kits Left", "Amb Trips", "Replannings"]
    res_vals   = np.array([kpi_tracker.kits_used,
                            10 - kpi_tracker.kits_used,
                            kpi_tracker.ambulance_trips,
                            kpi_tracker.replan_count], dtype=float)
    bars2 = ax2.bar(res_labels, res_vals,
                    color=["#4299e1","#bee3f8","#9f7aea","#fc8181"],
                    edgecolor="white")
    for bar, v in zip(bars2, res_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                 str(int(v)), ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("Resource Utilization", fontweight="bold")
    ax2.set_facecolor(C["panel_bg"])
    ax2.grid(axis="y", alpha=0.4, linestyle="--")
    ax2.tick_params(axis="x", rotation=15)

    # Panel 3: Rescue time per victim
    ax3 = fig.add_subplot(gs[0, 2])
    rescued = [v for v in victims if v.rescued and v.rescue_time is not None]
    v_times = np.array([v.rescue_time for v in rescued], dtype=float)
    ax3.bar([f"V{v.id}" for v in rescued], v_times,
            color=[SEV_COLOR[v.severity] for v in rescued], edgecolor="white")
    if len(v_times) > 0:
        ax3.axhline(float(np.mean(v_times)), color="#2d3748",
                    linestyle="--", linewidth=1.5,
                    label=f"Avg: {np.mean(v_times):.1f}")
    ax3.set_title("Rescue Time per Victim", fontweight="bold")
    ax3.set_ylabel("Steps"); ax3.legend(fontsize=8)
    ax3.set_facecolor(C["panel_bg"])
    ax3.grid(axis="y", alpha=0.4, linestyle="--")

    # Panel 4: Nodes expanded by algorithm
    ax4 = fig.add_subplot(gs[1, 0])
    seen: Dict = {}
    for r in kpi_tracker.search_results:
        if r.algorithm not in seen and r.found:
            seen[r.algorithm] = r
    if seen:
        algos = list(seen.keys())
        n_exp = np.array([seen[a].nodes_expanded for a in algos])
        palette = plt.cm.Pastel1(np.linspace(0, 1, len(algos)))
        bars4 = ax4.bar(range(len(algos)), n_exp, color=palette,
                        edgecolor="white", tick_label=algos)
        for bar, v in zip(bars4, n_exp):
            ax4.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                     str(int(v)), ha="center", fontsize=8)
        ax4.tick_params(axis="x", rotation=30)
        ax4.set_title("Nodes Expanded by Algorithm", fontweight="bold")
        ax4.set_ylabel("Nodes Expanded")
        ax4.set_facecolor(C["panel_bg"])
        ax4.grid(axis="y", alpha=0.4, linestyle="--")

    # Panel 5: ML comparison
    ax5 = fig.add_subplot(gs[1, 1])
    if kpi_tracker.ml_metrics:
        metric_keys = ["accuracy", "precision", "recall", "f1"]
        x5 = np.arange(len(metric_keys))
        w5 = 0.35
        ml_colors = ["#4299e1", "#48bb78"]
        for i, (model, md) in enumerate(kpi_tracker.ml_metrics.items()):
            vals   = np.array([md[k] for k in metric_keys])
            offset = (i - len(kpi_tracker.ml_metrics) / 2 + 0.5) * w5
            ax5.bar(x5 + offset, vals, w5, label=model,
                    color=ml_colors[i % 2], edgecolor="white")
        ax5.set_xticks(x5)
        ax5.set_xticklabels([m.upper() for m in metric_keys], fontsize=9)
        ax5.set_ylim(0, 1.25); ax5.set_title("ML Model Comparison", fontweight="bold")
        ax5.legend(fontsize=7, framealpha=0.9)
        ax5.set_facecolor(C["panel_bg"])
        ax5.grid(axis="y", alpha=0.4, linestyle="--")

    # Panel 6: CSP backtrack comparison
    ax6 = fig.add_subplot(gs[1, 2])
    if kpi_tracker.backtrack_counts:
        bt_labels = list(kpi_tracker.backtrack_counts.keys())
        bt_vals   = np.array(list(kpi_tracker.backtrack_counts.values()), dtype=float)
        ax6.bar(bt_labels, bt_vals,
                color=["#4299e1", "#fc8181"], edgecolor="white", width=0.5)
        for i, v in enumerate(bt_vals):
            ax6.text(i, v + 0.05, str(int(v)),
                     ha="center", fontsize=13, fontweight="bold")
        ax6.set_title("CSP Backtracks: Heuristic vs Naive", fontweight="bold")
        ax6.set_ylabel("Backtrack Count")
        ax6.set_facecolor(C["panel_bg"])
        ax6.grid(axis="y", alpha=0.4, linestyle="--")
        ax6.set_ylim(0, max(float(bt_vals.max()) + 1, 2))

    fig.suptitle("AIDRA — KPI Dashboard", fontsize=16,
                 fontweight="bold", y=1.01)
    return _save(fig, filename)


# ── 10. Fuzzy logic risk surface ──────────────────────────────────────────────

def plot_fuzzy_surface(filename: str = "fuzzy_surface.png") -> str:
    """
    3-D surface of fuzzy risk score across blockage probability × hazard level.
    NumPy meshgrid used for vectorised evaluation.
    """
    blockage_range = np.linspace(0, 1, 40)
    hazard_range   = np.linspace(0, 10, 40)
    B, H = np.meshgrid(blockage_range, hazard_range)
    Z = np.vectorize(lambda b, h: compute_risk_score(b, h, 3))(B, H)

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(B, H, Z, cmap="RdYlGn_r",
                           linewidth=0, antialiased=True, alpha=0.88)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12,
                 label="Risk Score (0=low, 10=critical)")
    ax.set_xlabel("Road Blockage Probability", fontsize=10, labelpad=8)
    ax.set_ylabel("Hazard Spread Level (0–10)", fontsize=10, labelpad=8)
    ax.set_zlabel("Fuzzy Risk Score", fontsize=10, labelpad=6)
    ax.set_title("Fuzzy Logic — Risk Score Surface\n(Victim Severity = Critical)",
                 fontsize=12, fontweight="bold")
    ax.view_init(elev=28, azim=225)
    plt.tight_layout()
    return _save(fig, filename)


# ── 11. Agent risk heatmap ────────────────────────────────────────────────────

def plot_risk_heatmap(grid_map: GridMap, victims: List[Victim],
                      filename: str = "risk_heatmap.png") -> str:
    rows, cols = grid_map.rows, grid_map.cols
    risk_matrix = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            if grid_map.grid[r][c] == CELL_BLOCKED:
                risk_matrix[r, c] = 11.0
                continue
            blockage = 0.7 if grid_map.is_high_risk((r, c)) else 0.1
            hazard   = 8.0 if grid_map.is_high_risk((r, c)) else 1.0
            risk_matrix[r, c] = compute_risk_score(blockage, hazard, 2)

    masked = np.ma.masked_where(risk_matrix > 10, risk_matrix)
    cmap_risk = LinearSegmentedColormap.from_list(
        "risk", ["#c6f6d5", "#fefcbf", "#fed7d7", "#fc8181", "#e53e3e"])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(masked, cmap=cmap_risk, vmin=0, vmax=10,
                   origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax,
                 label="Fuzzy Risk Score (0=safe, 10=critical)",
                 fraction=0.046, pad=0.04)

    # Blocked cells
    for r in range(rows):
        for c in range(cols):
            if grid_map.grid[r][c] == CELL_BLOCKED:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                             color=C["blocked"], zorder=2))

    # Victims
    for v in victims:
        vr, vc = v.position
        ax.plot(vc, vr, "o", markersize=13, color=SEV_COLOR[v.severity],
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.text(vc + 0.42, vr - 0.42, f"V{v.id}",
                fontsize=8, color="white", fontweight="bold", zorder=6)

    # Base & centers
    br, bc = grid_map.rescue_base
    ax.add_patch(plt.Circle((bc, br), 0.38, color=C["base"], zorder=4))
    ax.text(bc, br, "B", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=5)
    for (mr, mc) in grid_map.medical_centers:
        ax.add_patch(plt.Circle((mc, mr), 0.38, color=C["center"], zorder=4))
        ax.text(mc, mr, "M", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=5)

    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_title("Agent Risk Heatmap (Fuzzy Evaluation per Cell)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    plt.tight_layout()
    return _save(fig, filename)


# ── Master generate_all ───────────────────────────────────────────────────────

def generate_all(grid_map: GridMap,
                 victims: List[Victim],
                 kpi_tracker,
                 comparison_results: List[SearchResult],
                 knn_model=None,
                 nb_model=None,
                 victim_routes: Optional[Dict[int, List[Tuple[int, int]]]] = None):

    print("\n[Visualizer] Generating all figures…")
    plot_grid(grid_map, victims, title="Initial Disaster Scenario",
              filename="grid_initial.png")

    if any(v.rescued for v in victims):
        plot_grid(grid_map, victims, title="Final State — All Victims Rescued",
                  filename="grid_final.png")

    if victim_routes:
        plot_victim_routes(grid_map, victims, victim_routes)

    plot_search_comparison(comparison_results)
    plot_rescue_timeline(victims)
    plot_kpi_dashboard(kpi_tracker, victims)
    plot_fuzzy_surface()
    plot_risk_heatmap(grid_map, victims)

    for model_name, metrics in kpi_tracker.ml_metrics.items():
        safe = model_name.replace(" ","_").replace("(","").replace(")","")
        plot_confusion_matrix(metrics, model_name, f"cm_{safe}.png")

    plot_ml_comparison(kpi_tracker.ml_metrics)

    if knn_model is not None and nb_model is not None:
        plot_survival_scatter(knn_model, nb_model)

    print(f"[Visualizer] All figures saved to '{OUTPUT_DIR}/'")
