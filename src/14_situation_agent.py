
"""Task 15: Situation Awareness Agent."""

import json
import pandas as pd


class SituationAwarenessAgent:
    def __init__(self, results_path, stats_path):
        self.results_path = results_path
        self.stats_path = stats_path

    def run(self):
        df = pd.read_csv(self.results_path)
        with open(self.stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        shortest = df[df["route_type"] == "shortest_path"]
        risk = df[df["route_type"] == "risk_aware_astar"]

        situation = {
            "origins_total": stats["origins_total"],
            "origins_routed": stats["origins_routed"],
            "origins_skipped": len(stats["origins_skipped"]),
            "routes_changed": stats["routes_changed_by_risk_awareness"],
            "lambda": stats["lambda"],
            "mean_hazard_shortest": stats["mean_hazard_shortest"],
            "mean_hazard_risk_aware": stats["mean_hazard_risk_aware"],
            "high_hazard_share_shortest": stats["high_hazard_share_shortest"],
            "high_hazard_share_risk_aware": stats["high_hazard_share_risk_aware"],
            "poorly_connected_origins": stats["poorly_connected_routed"],
            "mean_risk_route_length_m": round(
                float(risk["total_length_with_access_m"].mean()), 2
            ),
            "mean_shortest_route_length_m": round(
                float(shortest["total_length_with_access_m"].mean()), 2
            ),
        }

        if situation["mean_hazard_risk_aware"] < situation["mean_hazard_shortest"]:
            situation["overall_assessment"] = (
                "Risk-aware routing reduces average route hazard."
            )
        else:
            situation["overall_assessment"] = (
                "Risk-aware routing does not reduce average route hazard."
            )

        situation["connectivity_warning"] = (
            "Some origins have low-confidence road access."
            if situation["poorly_connected_origins"] > 0
            else "No poorly connected origins reported."
        )

        return situation
