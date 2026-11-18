
"""Task 15: Communication Agent."""

import pandas as pd


class CommunicationAgent:
    def run(self, routes, situation):
        advisories = []

        for row in routes.itertuples(index=False):
            if row.poorly_connected:
                priority = "LOW_CONFIDENCE"
                message = (
                    "Road connectivity is uncertain. Verify local access "
                    "conditions before using the suggested route."
                )
            elif row.mean_hazard >= 0.5 or row.high_hazard_share >= 0.5:
                priority = "HIGH_ATTENTION"
                message = (
                    "The suggested route has substantial hazard exposure. "
                    "Verify road conditions and consider local alternatives."
                )
            elif row.mean_hazard >= 0.25:
                priority = "MODERATE_ATTENTION"
                message = (
                    "Moderate hazard exposure is present. Follow the "
                    "risk-aware route and monitor local conditions."
                )
            else:
                priority = "LOWER_RELATIVE_HAZARD"
                message = (
                    "The selected route has comparatively lower modelled "
                    "hazard exposure. Continue monitoring local conditions."
                )

            advisories.append({
                "village": row.village,
                "population_2018": row.population_2018,
                "camp_node": row.camp_node,
                "priority": priority,
                "route_rank": row.route_rank,
                "mean_hazard": row.mean_hazard,
                "total_journey_length_m": row.total_journey_length_m,
                "poorly_connected": row.poorly_connected,
                "advisory": message,
            })

        return pd.DataFrame(advisories)
