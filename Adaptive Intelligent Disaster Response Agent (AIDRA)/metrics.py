"""
metrics.py
KPI computation, Pandas reporting tables, and CSV export.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from environment import Victim, GridMap
from search import SearchResult


class KPITracker:
    def __init__(self):
        self.rescue_times:     List[float]        = []
        self.search_results:   List[SearchResult] = []
        self.risk_exposure:    float              = 0.0
        self.victims_saved:    int                = 0
        self.victims_total:    int                = 0
        self.kits_used:        int                = 0
        self.ambulance_trips:  int                = 0
        self.replan_count:     int                = 0
        self.backtrack_counts: Dict[str, int]     = {}
        self.ml_metrics:       Dict[str, Dict]    = {}
        self.rescue_log:       List[Dict]         = []   # per-victim records

    # ── Recording helpers ──────────────────────────────────────────────────────

    def record_rescue(self, victim: Victim, rescue_time: float):
        self.victims_saved += 1
        self.rescue_times.append(rescue_time)
        self.rescue_log.append({
            "victim_id":   victim.id,
            "severity":    victim.severity,
            "rescue_time": rescue_time,
            "position":    str(victim.position),
        })

    def record_search(self, result: SearchResult):
        self.search_results.append(result)
        self.risk_exposure += result.risk_cells

    def record_replan(self):
        self.replan_count += 1

    def record_backtrack(self, label: str, count: int):
        self.backtrack_counts[label] = count

    def record_ml(self, model_name: str, metrics: Dict):
        self.ml_metrics[model_name] = metrics

    # ── KPI calculations ───────────────────────────────────────────────────────

    def average_rescue_time(self) -> float:
        return float(np.mean(self.rescue_times)) if self.rescue_times else 0.0

    def path_optimality_ratio(self) -> Dict[str, float]:
        trips: Dict[tuple, List[SearchResult]] = {}
        for r in self.search_results:
            if r.found and r.path:
                goal = r.path[-1]
                trips.setdefault(goal, []).append(r)
        ratios = {}
        for goal, results in trips.items():
            best = min(r.cost for r in results)
            for r in results:
                key = f"{r.algorithm}→{goal}"
                ratios[key] = round(r.cost / best, 3) if best > 0 else 1.0
        return ratios

    def resource_utilization(self, total_kits: int = 10) -> Dict:
        kit_util = (self.kits_used / total_kits * 100) if total_kits > 0 else 0
        return {
            "ambulance_trips":     self.ambulance_trips,
            "kits_used":           self.kits_used,
            "kit_utilization_pct": round(kit_util, 1),
        }

    # ── Pandas DataFrames ──────────────────────────────────────────────────────

    def rescue_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rescue_log)

    def search_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.search_results:
            rows.append({
                "algorithm":      r.algorithm,
                "path_length":    len(r.path) - 1 if r.found else None,
                "cost":           r.cost,
                "nodes_expanded": r.nodes_expanded,
                "risk_cells":     r.risk_cells,
                "found":          r.found,
            })
        return pd.DataFrame(rows)

    def ml_dataframe(self) -> pd.DataFrame:
        rows = []
        for model, m in self.ml_metrics.items():
            rows.append({
                "model":     model,
                "accuracy":  m["accuracy"],
                "precision": m["precision"],
                "recall":    m["recall"],
                "f1":        m["f1"],
                "tp": m["tp"], "tn": m["tn"],
                "fp": m["fp"], "fn": m["fn"],
            })
        return pd.DataFrame(rows)

    def export_csv(self, folder: str = "."):
        self.rescue_dataframe().to_csv(f"{folder}/kpi_rescue.csv",      index=False)
        self.search_dataframe().to_csv(f"{folder}/kpi_search.csv",      index=False)
        self.ml_dataframe().to_csv(f"{folder}/kpi_ml.csv",              index=False)
        print(f"[Metrics] CSVs exported to '{folder}/'")

    # ── Final printed report ───────────────────────────────────────────────────

    def print_report(self):
        sep = "=" * 70
        print(f"\n{sep}")
        print("PERFORMANCE REPORT — AIDRA")
        print(sep)

        # Rescue outcomes
        print(f"\n{'─'*30} Rescue Outcomes {'─'*23}")
        print(f"  Victims saved          : {self.victims_saved} / {self.victims_total}")
        print(f"  Average rescue time    : {self.average_rescue_time():.2f} steps")
        print(f"  Total replanning events: {self.replan_count}")

        df_rescue = self.rescue_dataframe()
        if not df_rescue.empty:
            print("\n" + df_rescue.to_string(index=False))

        # Search comparison (aggregated by algorithm)
        df_search = self.search_dataframe()
        if not df_search.empty:
            print(f"\n{'─'*30} Search Algorithm Comparison {'─'*12}")
            agg = (df_search[df_search["found"]]
                   .groupby("algorithm")
                   .agg(
                       avg_cost=("cost", "mean"),
                       avg_nodes=("nodes_expanded", "mean"),
                       avg_risk_cells=("risk_cells", "mean"),
                       runs=("cost", "count"),
                   )
                   .round(2)
                   .reset_index())
            print(agg.to_string(index=False))

        # Path optimality
        ratios = self.path_optimality_ratio()
        if ratios:
            df_ratio = pd.DataFrame(
                [{"route": k, "optimality_ratio": v} for k, v in ratios.items()])
            print(f"\n  Path Optimality Ratios (1.000 = optimal):")
            print(df_ratio.to_string(index=False))

        # Risk & resources
        print(f"\n{'─'*30} Risk & Resources {'─'*23}")
        print(f"  Total high-risk cells traversed: {self.risk_exposure}")
        ru = self.resource_utilization()
        print(f"  Ambulance trips  : {ru['ambulance_trips']}")
        print(f"  Kits used        : {ru['kits_used']} / 10  ({ru['kit_utilization_pct']}%)")

        # CSP
        if self.backtrack_counts:
            print(f"\n{'─'*30} CSP Solver {'─'*29}")
            df_csp = pd.DataFrame([
                {"solver": k, "backtracks": v}
                for k, v in self.backtrack_counts.items()
            ])
            print(df_csp.to_string(index=False))

        # ML comparison
        df_ml = self.ml_dataframe()
        if not df_ml.empty:
            print(f"\n{'─'*30} ML Model Comparison {'─'*20}")
            cols = ["model", "accuracy", "precision", "recall", "f1"]
            print(df_ml[cols].round(3).to_string(index=False))

        print(f"\n{sep}\n")