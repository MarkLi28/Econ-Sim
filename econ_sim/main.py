"""Economic Simulation with AI Agents.

Usage:
    python -m econ_sim.main --system capitalist --rounds 20
    python -m econ_sim.main --system socialist --rounds 20
    python -m econ_sim.main --system mixed --rounds 5 --firms 2 --workers 3
"""

import argparse
import logging

from econ_sim.config import SYSTEM_PRESETS, EconomicConfig
from econ_sim.llm.client import AnthropicLLMClient
from econ_sim.engine.environment import EconomicEnvironment


def main():
    parser = argparse.ArgumentParser(description="Economic Simulation with AI Agents")
    parser.add_argument(
        "--system",
        choices=["capitalist", "socialist", "mixed"],
        default="capitalist",
        help="Economic system preset",
    )
    parser.add_argument("--rounds", type=int, default=20, help="Number of simulation rounds")
    parser.add_argument("--firms", type=int, default=3, help="Number of firms")
    parser.add_argument("--workers", type=int, default=5, help="Number of workers")
    parser.add_argument("--model", type=str, default="claude-opus-4-6", help="Anthropic model ID")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build config from preset + overrides
    config = SYSTEM_PRESETS[args.system]
    config.num_rounds = args.rounds
    config.num_firms = args.firms
    config.num_workers = args.workers
    config.model = args.model
    config.temperature = args.temperature

    print(f"Economic Simulation — {args.system.upper()}")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Firms: {config.num_firms}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Model: {config.model}")
    print()

    llm_client = AnthropicLLMClient(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    env = EconomicEnvironment(config, llm_client, output_dir=args.output_dir)
    env.run_simulation()


if __name__ == "__main__":
    main()
