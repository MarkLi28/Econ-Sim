import json
import os
import logging
from dataclasses import asdict
from datetime import datetime

logger = logging.getLogger(__name__)


class SimulationRecorder:
    """Records simulation data to JSON for post-run analysis."""

    def __init__(self, output_dir: str, config):
        self.output_dir = output_dir
        self.config = config
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rounds_data: list[dict] = []

    def record_round(
        self,
        round_num: int,
        world_state: dict,
        govt_action: dict,
        firm_actions: dict,
        worker_actions: dict,
        labor_results: dict,
        goods_results: dict,
        fiscal_results: dict,
    ):
        self.rounds_data.append({
            "round": round_num,
            "statistics": world_state["statistics"],
            "actions": {
                "government": govt_action,
                "firms": firm_actions,
                "workers": worker_actions,
            },
            "outcomes": {
                "labor_market": labor_results,
                "goods_market": {
                    "firm_sales": goods_results["firm_sales"],
                    "firm_revenue": goods_results["firm_revenue"],
                    "unmet_demand": goods_results["unmet_demand"],
                },
                "fiscal": fiscal_results,
            },
        })

    def save(self, government, firms, workers):
        """Save the full simulation record to a JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{self.config.system.value}_{self.run_id}.json"
        path = os.path.join(self.output_dir, filename)

        data = {
            "run_id": self.run_id,
            "config": {
                "system": self.config.system.value,
                "num_rounds": self.config.num_rounds,
                "num_firms": self.config.num_firms,
                "num_workers": self.config.num_workers,
                "productivity_factor": self.config.productivity_factor,
                "income_tax_rate_range": list(self.config.income_tax_rate_range),
                "corporate_tax_rate_range": list(self.config.corporate_tax_rate_range),
            },
            "rounds": self.rounds_data,
            "agent_logs": {
                government.name: {
                    "action_log": government.action_log,
                },
                **{
                    f.name: {"action_log": f.action_log}
                    for f in firms
                },
                **{
                    w.name: {"action_log": w.action_log}
                    for w in workers
                },
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Simulation saved to {path}")
        print(f"\nResults saved to: {path}")
