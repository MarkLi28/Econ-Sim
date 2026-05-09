"""Economic Simulation with AI Agents — Phase Diagram Research Edition.

Single run:
    python -m econ_sim --system capitalist --rounds 15
    python -m econ_sim --system nordic --rounds 15
    python -m econ_sim --coordination 0.1 --structure 0.2 --meta-game 0.1 --rounds 15

Parameter sweep (phase diagram):
    python -m econ_sim --sweep --rounds 10 --output-dir results/sweep

Named systems: capitalist, crony, socialist, nordic, mixed
"""

import argparse
import itertools
import json
import logging
import os
import uuid

from econ_sim.config import SYSTEM_PRESETS, EconomicConfig, make_config
from econ_sim.engine.environment import EconomicEnvironment
from econ_sim import registry


def _run_one(
    config: EconomicConfig,
    output_dir: str,
    sweep_id: str | None = None,
    sweep_name: str | None = None,
    seed: int | None = None,
) -> dict:
    """Run one simulation under the registry. Returns the result dict."""
    env = EconomicEnvironment(config, output_dir=output_dir)
    with registry.start_run(
        config,
        sweep_id=sweep_id,
        sweep_name=sweep_name,
        seed=seed,
        architecture="llm_only",
        tier="toy",
    ) as run:
        result = env.run()
        run.complete(
            result,
            cost_summary=env.llm.cost_summary(),
            results_path=os.path.join(
                output_dir, f"{config.label()}_simulation_results.json"
            ),
        )
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Economic Simulation with AI Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Named preset OR explicit axes
    parser.add_argument(
        "--system",
        choices=list(SYSTEM_PRESETS.keys()),
        default=None,
        help="Named system preset (capitalist, crony, socialist, nordic, mixed)",
    )
    parser.add_argument("--coordination", type=float, default=None,
                        help="coordination_mechanism axis (0.0=market, 1.0=planning)")
    parser.add_argument("--structure", type=float, default=None,
                        help="planning_structure axis (0.0=distributed, 1.0=concentrated)")
    parser.add_argument("--meta-game", type=float, default=None,
                        help="meta_game_constraint axis (0.0=open lobbying, 1.0=suppressed)")

    # Sweep mode
    parser.add_argument("--sweep", action="store_true",
                        help="Run a parameter grid sweep to map the phase diagram")
    parser.add_argument("--sweep-steps", type=int, default=3,
                        help="Number of steps per axis in sweep (3=0/0.5/1, 5=0/0.25/0.5/0.75/1)")

    # Simulation settings
    parser.add_argument("--rounds", type=int, default=60,
                        help="Max rounds per simulation (convergence detection may stop earlier; "
                             "default 60 with persistence requirement of 5 consecutive matching rounds)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.sweep:
        _run_sweep(args)
    else:
        _run_single(args)


def _build_config(args) -> EconomicConfig:
    """Build config from named preset or explicit axes."""
    if args.system:
        cfg = SYSTEM_PRESETS[args.system]
        cfg.num_rounds = args.rounds
        return cfg

    # Explicit axes
    c = args.coordination if args.coordination is not None else 0.1
    s = args.structure if args.structure is not None else 0.2
    m = args.meta_game if args.meta_game is not None else 0.1

    return make_config(coordination=c, structure=s, meta_game=m, num_rounds=args.rounds)


def _run_single(args):
    config = _build_config(args)

    print(f"\n{'='*60}")
    print(f"ECONOMIC SIMULATION — {config.label().upper()}")
    print(f"{'='*60}")
    print(f"  Axes:   coord={config.coordination_mechanism:.2f}  "
          f"struct={config.planning_structure:.2f}  meta={config.meta_game_constraint:.2f}")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Initial integrity: {config.initial_institutional_integrity:.2f}")
    print(f"  Industries: {', '.join(i.value for i in config.industry_configs)}")
    print(f"  Worker archetypes: {len(config.worker_profiles)}")
    print(f"  Max lobby fraction: {config.max_lobby_fraction:.1%}")
    print(f"  Lobby detection prob: {config.lobby_detection_prob:.1%}")
    _print_cost_estimate(config)
    print()

    result = _run_one(config, output_dir=args.output_dir)

    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Regime:          {result['regime']}")
    print(f"  Rounds run:      {result['summary'].get('rounds_run', '?')}")
    conv = result.get("convergence", {})
    if conv:
        print(f"  Trend:           {conv.get('trend', '?')} (confidence={conv.get('confidence', 0):.0%})")
        print(f"  Convergence:     {conv.get('message', '')}")
    print(f"  Final welfare:   {result['summary']['final_welfare']:.3f}")
    print(f"  Final integrity: {result['summary']['final_integrity']:.2f}")
    print(f"  Final Gini:      {result['summary']['final_gini']:.3f}")
    print(f"  Welfare trend:   {result['summary']['welfare_trend']:+.3f}")
    print(f"  Integrity trend: {result['summary']['integrity_trend']:+.3f}")


def _run_sweep(args):
    """Grid search across all three axes. Saves a sweep_results.json summary."""
    n = args.sweep_steps
    axis_values = [round(i / (n - 1), 2) for i in range(n)] if n > 1 else [0.5]

    combos = list(itertools.product(axis_values, axis_values, axis_values))
    total = len(combos)

    print(f"\n{'='*60}")
    print(f"PHASE DIAGRAM SWEEP — {total} parameter combinations")
    print(f"Axis values: {axis_values}")
    print(f"Rounds per run: {args.rounds}")
    print(f"{'='*60}\n")

    sweep_dir = os.path.join(args.output_dir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_results = []

    sweep_id = str(uuid.uuid4())
    sweep_name = f"grid{n}_r{args.rounds}"
    print(f"Sweep ID: {sweep_id}  ({sweep_name})")

    for i, (c, s, m) in enumerate(combos, 1):
        print(f"\n[{i}/{total}] coord={c:.2f} struct={s:.2f} meta={m:.2f}")
        config = make_config(coordination=c, structure=s, meta_game=m, num_rounds=args.rounds)

        try:
            result = _run_one(
                config,
                output_dir=sweep_dir,
                sweep_id=sweep_id,
                sweep_name=sweep_name,
            )
            sweep_results.append({
                "coordination": c,
                "planning_structure": s,
                "meta_game_constraint": m,
                "system_label": config.label(),
                "regime": result["regime"],
                **result["summary"],
            })
            print(f"  → {result['regime']} | W={result['summary']['final_welfare']:.3f} "
                  f"| integrity={result['summary']['final_integrity']:.2f}")
        except Exception as e:
            logging.error(f"Run failed for ({c},{s},{m}): {e}")
            sweep_results.append({
                "coordination": c,
                "planning_structure": s,
                "meta_game_constraint": m,
                "regime": "error",
                "error": str(e),
            })

    # Save sweep summary
    summary_path = os.path.join(sweep_dir, "phase_diagram.json")
    with open(summary_path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\nPhase diagram saved to: {summary_path}")
    _print_phase_summary(sweep_results)


def _print_phase_summary(results: list):
    from collections import Counter
    regime_counts = Counter(r.get("regime", "error") for r in results)
    print(f"\nRegime distribution across {len(results)} parameter combinations:")
    for regime, count in regime_counts.most_common():
        pct = 100 * count / len(results)
        bar = "█" * int(pct / 5)
        print(f"  {regime:<25} {count:>3} ({pct:.0f}%) {bar}")


def _print_cost_estimate(config: EconomicConfig):
    # Rough estimate per round
    num_firms = sum(ic.num_firms for ic in config.industry_configs.values())
    num_workers = len(config.worker_profiles)
    opus_calls = 1 + sum(
        1 for ic in config.industry_configs.values()
        if ic.model_tier == "opus" for _ in range(ic.num_firms)
    )
    sonnet_calls = sum(
        1 for ic in config.industry_configs.values()
        if ic.model_tier == "sonnet" for _ in range(ic.num_firms)
    )
    haiku_calls = (
        sum(1 for ic in config.industry_configs.values()
            if ic.model_tier == "haiku" for _ in range(ic.num_firms))
        + num_workers * 2  # employment + consumer
    )
    total_rounds = config.num_rounds
    opus_cost = opus_calls * total_rounds * 0.015 * 2   # rough: 2k tokens avg
    sonnet_cost = sonnet_calls * total_rounds * 0.003 * 2
    haiku_cost = haiku_calls * total_rounds * 0.0008 * 2
    est_total = opus_cost + sonnet_cost + haiku_cost
    print(f"  Cost estimate: ~${est_total:.2f} "
          f"({opus_calls} Opus + {sonnet_calls} Sonnet + {haiku_calls} Haiku calls/round)")


if __name__ == "__main__":
    main()
