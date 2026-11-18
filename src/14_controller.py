
"""Task 15: Multi-agent controller.

Run from project root:
    python src/14_controller.py
"""

import importlib.util
import json
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_agent(filename, class_name):
    path = SRC_DIR / filename
    spec = importlib.util.spec_from_file_location(class_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def main():
    SituationAwarenessAgent = load_agent(
        "14_situation_agent.py", "SituationAwarenessAgent"
    )
    RouteOptimisationAgent = load_agent(
        "14_route_agent.py", "RouteOptimisationAgent"
    )
    CommunicationAgent = load_agent(
        "14_comm_agent.py", "CommunicationAgent"
    )

    tables = OUTPUT_DIR / "tables"
    stats_dir = OUTPUT_DIR / "stats"

    results_path = tables / "08_routing_results.csv"
    stats_path = tables / "08_routing_stats.json"

    if not results_path.exists():
        raise FileNotFoundError(results_path)

    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    logger.info("Agent 1: Situation Awareness")
    situation_agent = SituationAwarenessAgent(results_path, stats_path)
    situation = situation_agent.run()

    logger.info("Agent 2: Route Optimisation")
    route_agent = RouteOptimisationAgent(results_path)
    routes = route_agent.run()

    logger.info("Agent 3: Communication")
    communication_agent = CommunicationAgent()
    advisories = communication_agent.run(routes, situation)

    # Save outputs
    stats_dir.mkdir(parents=True, exist_ok=True)

    routes.to_csv(
        tables / "14_agent_route_recommendations.csv", index=False
    )

    advisories.to_csv(
        tables / "14_agent_village_advisories.csv", index=False
    )

    with open(
        stats_dir / "14_agent_situation_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(situation, f, indent=2)

    report = [
        "# Multi-Agent Flood Evacuation Decision Support",
        "",
        "## Situation Assessment",
        "",
    ]

    for key, value in situation.items():
        report.append(f"- **{key.replace('_', ' ').title()}:** {value}")

    report.extend([
        "",
        "## Village-Level Recommendations",
        "",
        "| Village | Priority | Route length (m) | Mean hazard | Advisory |",
        "|---|---|---:|---:|---|",
    ])

    for row in advisories.itertuples(index=False):
        message = row.advisory.replace("|", "/")
        report.append(
            f"| {row.village} | {row.priority} | "
            f"{row.total_journey_length_m:.1f} | "
            f"{row.mean_hazard:.4f} | {message} |"
        )

    report.extend([
        "",
        "## Limitations",
        "",
        "- These are model-based scenario recommendations, not live evacuation instructions.",
        "- The 2018 flood extent is a hazard proxy, not a road-closure dataset.",
        "- Poorly connected origins have low-confidence access estimates.",
        "- Local authorities must verify road and shelter conditions before action.",
    ])

    (OUTPUT_DIR / "14_multi_agent_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    logger.info("Task 15 completed.")
    logger.info("Generated %d village advisories.", len(advisories))


if __name__ == "__main__":
    main()
