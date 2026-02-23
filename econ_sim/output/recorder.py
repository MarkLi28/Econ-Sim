import json
import os
import logging
from dataclasses import asdict
from datetime import datetime

logger = logging.getLogger(__name__)


class SimulationRecorder:
    """Records simulation data to JSON for post-run analysis and phase diagram research."""

    def __init__(self, config, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rounds_data: list[dict] = []
        os.makedirs(output_dir, exist_ok=True)

    def record_round(
        self,
        round_num: int,
        stats,
        gov_action: dict,
        firm_states: list,
        worker_states: list,
        institutional_integrity: float,
        labor_results: dict,
    ):
        self.rounds_data.append({
            "round": round_num,
            "institutional_integrity": institutional_integrity,
            "statistics": asdict(stats),
            "government_action": gov_action,
            "firm_states": firm_states,
            "worker_states": worker_states,
            "labor_results": {
                "assignments": labor_results.get("assignments", {}),
                "unemployed": labor_results.get("unemployed", []),
            },
        })

    def finalize(
        self,
        stats_history: list,
        regime: str,
        integrity_history: list,
        cost_summary: dict,
    ) -> dict:
        """Write full JSON output. Returns the result dict."""
        label = self.config.label()
        filename = f"{label}_{self.run_id}.json"
        path = os.path.join(self.output_dir, filename)

        # Time-series data for phase diagram analysis
        ts = {
            "gdp":                  [s.gdp for s in stats_history],
            "unemployment":         [s.unemployment_rate for s in stats_history],
            "gini":                 [s.gini for s in stats_history],
            "welfare_score":        [s.welfare_score for s in stats_history],
            "basic_needs_rate":     [s.basic_needs_fulfillment_rate for s in stats_history],
            "total_suffering":      [s.total_suffering for s in stats_history],
            "institutional_integrity": integrity_history,
            "total_lobby_spending": [s.total_lobby_spending for s in stats_history],
            "industry_gdp":         [s.industry_gdp for s in stats_history],
        }

        result = {
            "run_id": self.run_id,
            "regime": regime,

            # Research axes — key for phase diagram
            "axes": {
                "coordination_mechanism": self.config.coordination_mechanism,
                "planning_structure":     self.config.planning_structure,
                "meta_game_constraint":   self.config.meta_game_constraint,
                "system_label":           label,
            },

            # Summary statistics (for quick phase diagram lookup)
            "summary": {
                "final_welfare":        ts["welfare_score"][-1] if ts["welfare_score"] else 0,
                "final_integrity":      integrity_history[-1] if integrity_history else 1,
                "final_gini":           ts["gini"][-1] if ts["gini"] else 0,
                "final_unemployment":   ts["unemployment"][-1] if ts["unemployment"] else 0,
                "final_needs_rate":     ts["basic_needs_rate"][-1] if ts["basic_needs_rate"] else 1,
                "total_suffering":      ts["total_suffering"][-1] if ts["total_suffering"] else 0,
                "integrity_trend":      (integrity_history[-1] - integrity_history[0]) if len(integrity_history) > 1 else 0,
                "welfare_trend":        (ts["welfare_score"][-1] - ts["welfare_score"][0]) if len(ts["welfare_score"]) > 1 else 0,
            },

            "time_series": ts,
            "rounds": self.rounds_data,
            "llm_cost": cost_summary,

            "config": {
                "coordination_mechanism":       self.config.coordination_mechanism,
                "planning_structure":           self.config.planning_structure,
                "meta_game_constraint":         self.config.meta_game_constraint,
                "num_rounds":                   self.config.num_rounds,
                "initial_institutional_integrity": self.config.initial_institutional_integrity,
                "industries": {
                    ind.value: {
                        "num_firms":            ic.num_firms,
                        "base_price":           ic.base_price,
                        "productivity_factor":  ic.productivity_factor,
                        "model_tier":           ic.model_tier,
                        "is_necessity":         ic.is_necessity,
                    }
                    for ind, ic in self.config.industry_configs.items()
                },
                "worker_profiles": [
                    {
                        "name":                 p.name,
                        "skill_level":          p.skill_level,
                        "representative_count": p.representative_count,
                    }
                    for p in self.config.worker_profiles
                ],
            },
        }

        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Simulation saved to {path}")
        print(f"\nResults saved to: {path}")
        print(f"Regime classification: {regime}")

        total_cost = sum(
            v.get("cost_usd", 0) for v in cost_summary.values()
        )
        print(f"Estimated LLM cost: ${total_cost:.3f}")

        return result
