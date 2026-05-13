"""
metrics.py
KPI computation and CSV export helpers for AIDRA.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from environment import Victim
from search import SearchResult


def _format_table(rows: List[Dict], columns: List[str]) -> str:
    if not rows:
        return "  (no rows)"

    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    lines = []
    lines.append("  " + " | ".join(col.ljust(widths[col]) for col in columns))
    lines.append("  " + "-+-".join("-" * widths[col] for col in columns))
    for row in rows:
        lines.append("  " + " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
    return "\n".join(lines)


class KPITracker:
    def __init__(self):
        self.rescue_times: List[float] = []
        self.search_results: List[SearchResult] = []
        self.risk_exposure: float = 0.0
        self.victims_saved: int = 0
        self.victims_total: int = 0
        self.kits_used: int = 0
        self.ambulance_trips: int = 0
        self.replan_count: int = 0
        self.backtrack_counts: Dict[str, int] = {}
        self.ml_metrics: Dict[str, Dict] = {}
        self.rescue_log: List[Dict] = []

    def record_rescue(self, victim: Victim, rescue_time: float):
        self.victims_saved += 1
        self.rescue_times.append(rescue_time)
        self.rescue_log.append(
            {
                "victim_id": victim.id,
                "severity": victim.severity,
                "rescue_time": rescue_time,
                "position": str(victim.position),
            }
        )

    def record_search(self, result: SearchResult):
        self.search_results.append(result)
        self.risk_exposure += result.risk_cells

    def record_replan(self):
        self.replan_count += 1

    def record_backtrack(self, label: str, count: int):
        self.backtrack_counts[label] = count

    def record_ml(self, model_name: str, metrics: Dict):
        self.ml_metrics[model_name] = metrics

    def average_rescue_time(self) -> float:
        return float(np.mean(self.rescue_times)) if self.rescue_times else 0.0

    def path_optimality_ratio(self) -> Dict[str, float]:
        trips: Dict[tuple, List[SearchResult]] = {}
        for r in self.search_results:
            if r.found and r.path:
                trips.setdefault(r.path[-1], []).append(r)
        ratios = {}
        for goal, results in trips.items():
            best = min(r.cost for r in results)
            for r in results:
                ratios[f"{r.algorithm}->{goal}"] = round(r.cost / best, 3) if best > 0 else 1.0
        return ratios

    def resource_utilization(self, total_kits: int = 10) -> Dict:
        kit_util = (self.kits_used / total_kits * 100) if total_kits > 0 else 0
        return {
            "ambulance_trips": self.ambulance_trips,
            "kits_used": self.kits_used,
            "kit_utilization_pct": round(kit_util, 1),
        }

    def rescue_dataframe(self) -> List[Dict]:
        return list(self.rescue_log)

    def search_dataframe(self) -> List[Dict]:
        return [
            {
                "algorithm": r.algorithm,
                "path_length": len(r.path) - 1 if r.found else None,
                "cost": r.cost,
                "nodes_expanded": r.nodes_expanded,
                "risk_cells": r.risk_cells,
                "found": r.found,
            }
            for r in self.search_results
        ]

    def ml_dataframe(self) -> List[Dict]:
        rows = []
        for model, m in self.ml_metrics.items():
            rows.append(
                {
                    "model": model,
                    "accuracy": m.get("accuracy", 0),
                    "precision": m.get("precision", 0),
                    "recall": m.get("recall", 0),
                    "f1": m.get("f1", 0),
                    "tp": m.get("tp", ""),
                    "tn": m.get("tn", ""),
                    "fp": m.get("fp", ""),
                    "fn": m.get("fn", ""),
                }
            )
        return rows

    def _write_csv(self, path: Path, rows: List[Dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            if not rows:
                handle.write("")
                return
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def export_csv(self, folder: str = "."):
        folder_path = Path(folder)
        self._write_csv(folder_path / "kpi_rescue.csv", self.rescue_dataframe())
        self._write_csv(folder_path / "kpi_search.csv", self.search_dataframe())
        self._write_csv(folder_path / "kpi_ml.csv", self.ml_dataframe())
        print(f"[Metrics] CSVs exported to '{folder_path}/'")

    def print_report(self):
        sep = "=" * 70
        print(f"\n{sep}")
        print("PERFORMANCE REPORT - AIDRA")
        print(sep)

        print(f"\n{'-' * 30} Rescue Outcomes {'-' * 23}")
        print(f"  Victims saved          : {self.victims_saved} / {self.victims_total}")
        print(f"  Average rescue time    : {self.average_rescue_time():.2f} steps")
        print(f"  Total replanning events: {self.replan_count}")

        rescue_rows = self.rescue_dataframe()
        if rescue_rows:
            print(_format_table(rescue_rows, ["victim_id", "severity", "rescue_time", "position"]))

        search_rows = self.search_dataframe()
        if search_rows:
            print(f"\n{'-' * 30} Search Algorithm Comparison {'-' * 12}")
            summary = {}
            for row in search_rows:
                if row["found"]:
                    stats = summary.setdefault(row["algorithm"], {"cost": [], "nodes": [], "risk": [], "runs": 0})
                    stats["cost"].append(float(row["cost"]))
                    stats["nodes"].append(float(row["nodes_expanded"]))
                    stats["risk"].append(float(row["risk_cells"]))
                    stats["runs"] += 1
            summary_rows = []
            for algo, stats in sorted(summary.items()):
                summary_rows.append(
                    {
                        "algorithm": algo,
                        "avg_cost": round(float(np.mean(stats["cost"])), 2) if stats["cost"] else 0,
                        "avg_nodes": round(float(np.mean(stats["nodes"])), 2) if stats["nodes"] else 0,
                        "avg_risk_cells": round(float(np.mean(stats["risk"])), 2) if stats["risk"] else 0,
                        "runs": stats["runs"],
                    }
                )
            print(_format_table(summary_rows, ["algorithm", "avg_cost", "avg_nodes", "avg_risk_cells", "runs"]))

        ratios = self.path_optimality_ratio()
        if ratios:
            print("\n  Path Optimality Ratios (1.000 = optimal):")
            print(_format_table([{"route": k, "optimality_ratio": v} for k, v in ratios.items()], ["route", "optimality_ratio"]))

        print(f"\n{'-' * 30} Risk & Resources {'-' * 23}")
        print(f"  Total high-risk cells traversed: {self.risk_exposure}")
        ru = self.resource_utilization()
        print(f"  Ambulance trips  : {ru['ambulance_trips']}")
        print(f"  Kits used        : {ru['kits_used']} / 10  ({ru['kit_utilization_pct']}%)")

        if self.backtrack_counts:
            print(f"\n{'-' * 30} CSP Solver {'-' * 29}")
            print(_format_table([{"solver": k, "backtracks": v} for k, v in self.backtrack_counts.items()], ["solver", "backtracks"]))

        ml_rows = self.ml_dataframe()
        if ml_rows:
            print(f"\n{'-' * 30} ML Model Comparison {'-' * 20}")
            print(_format_table(ml_rows, ["model", "accuracy", "precision", "recall", "f1"]))

        print(f"\n{sep}\n")
