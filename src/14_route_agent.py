
"""Task 15: Route Optimisation Agent."""

import pandas as pd


class RouteOptimisationAgent:
    def __init__(self, results_path):
        self.results_path = results_path

    def run(self):
        df = pd.read_csv(self.results_path)

        required = {
            "village", "population_2018", "route_type",
            "length_m", "mean_hazard", "max_hazard",
            "high_hazard_share", "total_length_with_access_m",
            "poorly_connected", "camp_node"
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Routing results missing columns: {sorted(missing)}")

        risk = df[df["route_type"] == "risk_aware_astar"].copy()
        shortest = df[df["route_type"] == "shortest_path"].copy()

        if risk.empty:
            raise ValueError("No risk-aware routes found.")

        # Rank by hazard first, then high-hazard exposure, then journey distance.
        risk["priority_score"] = (
            0.50 * risk["mean_hazard"].fillna(1)
            + 0.30 * risk["high_hazard_share"].fillna(1)
            + 0.20 * (
                risk["total_length_with_access_m"].fillna(
                    risk["total_length_with_access_m"].median()
                ) /
                max(risk["total_length_with_access_m"].max(), 1)
            )
        )

        risk = risk.sort_values("priority_score").reset_index(drop=True)
        risk["route_rank"] = risk.index + 1

        shortest_lookup = shortest.set_index("village")

        records = []

        for row in risk.itertuples(index=False):
            village = row.village
            baseline = shortest_lookup.loc[village]

            records.append({
                "village": village,
                "population_2018": row.population_2018,
                "camp_node": row.camp_node,
                "route_rank": row.route_rank,
                "priority_score": round(row.priority_score, 4),
                "route_length_m": row.length_m,
                "total_journey_length_m": row.total_length_with_access_m,
                "mean_hazard": row.mean_hazard,
                "max_hazard": row.max_hazard,
                "high_hazard_share": row.high_hazard_share,
                "shortest_route_length_m": baseline["length_m"],
                "poorly_connected": bool(row.poorly_connected),
            })

        return pd.DataFrame(records)
