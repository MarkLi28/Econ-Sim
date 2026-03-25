# Econ-Sim — Project Context for Claude Code

## Research Question
**"Is there a mixed-system equilibrium where the meta-game (lobbying/corruption) exists but is bounded?"**

The "meta-game" is the universal pattern where entities shift from competing within the rules to reshaping the rules themselves. This appears everywhere: autocatalysis spawning life, niche construction in evolution, lobbying/corruption in economies. The simulation maps a phase diagram across 3 axes to find where institutions hold the meta-game bounded vs. where they collapse into capture.

## Three-Axis Parameter Space
- `coordination_mechanism` (0=pure market, 1=central planning) — how prices/wages are set
- `planning_structure` (0=distributed agencies, 1=single concentrated planner) — governance shape
- `meta_game_constraint` (0=open lobbying, 1=fully suppressed) — how hard it is to corrupt

These mechanically shape: agent action spaces, lobbying success probabilities, government constraint levels, and institutional integrity dynamics.

## Architecture

### Agents (all LLM-powered)
- **Government** (1 agent, Opus) — sets tax rates, subsidies, wage/price controls. Has `institutional_integrity` state variable (0-1) that erodes via lobbying and recovers via public pressure.
- **Firms** (8 total, 2 per industry) — set prices, wages, hiring targets, lobby spending. Industry-specific prompts and strategies. Opus for manufacturing/tech, Sonnet for housing, Haiku for agriculture.
- **Workers** (10 agents, 4 archetypes) — make employment decisions (accept/quit/switch jobs) and consumption decisions (necessity-first, then discretionary). All use Haiku.

### Industries (4-sector supply chain)
- **Agriculture** — produces food (necessity, 1.0 units/worker/round). No upstream dependency.
- **Manufacturing** — produces materials (B2B only). Upstream supplier to Housing and Technology. Strategic leverage position.
- **Housing** — produces shelter (necessity, 0.5 units/worker/round). Depends on Manufacturing materials (0.4 ratio).
- **Technology** — produces tech services (discretionary consumer good). Depends on Manufacturing materials (0.3 ratio). High-margin but first to lose sales in downturn.

### 13-Phase Round Structure
1. Government policy decisions
2. Firm decisions (parallel LLM calls)
3. Input market (B2B: manufacturing sells to housing/tech)
4. Worker employment decisions (parallel LLM calls)
5. Labor market clearing (hiring/firing with friction costs)
6. Production (output = workers x productivity x skill)
7. Necessity market clearing (food + shelter auto-buy)
8. Worker consumption decisions
9. Discretionary market clearing
10. Fiscal phase (taxes collected)
11. Wage payments
12. Institutional integrity dynamics (lobbying erosion + public pressure recovery)
13. Statistics computation + convergence check

### Convergence Detection
Simulation runs up to 40 rounds but stops early when welfare trend is classifiable:
- **Collapse**: welfare < 0.10 for 3 consecutive rounds
- **Oscillation**: 70%+ sign changes in diffs + amplitude > 0.02
- **Stable**: slope < 0.005, low variance on welfare and integrity
- **Strong trend**: R^2 > 0.80 and |slope| > 0.008
- **Weak trend**: R^2 > 0.60 and |slope| > 0.005 after 15+ rounds

### Welfare Function
W = 0.30*(1-unemployment) + 0.25*(1-gini) + 0.25*basic_needs_rate + 0.10*gdp_growth + 0.10*fiscal_health

### Labor Market Friction
Hiring cost: 0.5x wage per new hire. Firing cost: 1.0x wage per layoff. Prevents employment cycling.

## Named Presets
| Preset | coordination | structure | meta_game | Intent |
|--------|-------------|-----------|-----------|--------|
| capitalist | 0.10 | 0.20 | 0.10 | Laissez-faire with weak institutions |
| crony | 0.10 | 0.30 | 0.00 | No lobbying constraints at all |
| socialist | 0.85 | 0.80 | 0.65 | Central planning with concentrated power |
| nordic | 0.50 | 0.30 | 0.80 | Mixed economy with strong anti-corruption |
| mixed | 0.40 | 0.40 | 0.45 | Middle of the parameter space |

## Key Files
- `econ_sim/config.py` — 3-axis config, industry/worker/necessity definitions, presets
- `econ_sim/agents/government.py` — Opus agent, institutional integrity, lobby buffer, necessity diagnostics
- `econ_sim/agents/firm.py` — Industry-specific firms, hiring/firing cost tracking
- `econ_sim/agents/worker.py` — Dual-decide: employment + consumption
- `econ_sim/engine/environment.py` — 13-phase loop, convergence check
- `econ_sim/engine/statistics.py` — ConvergenceResult, check_convergence(), welfare formula, regime classification
- `econ_sim/main.py` — CLI: `--system`, `--rounds`, `--sweep`
- `econ_sim/output/recorder.py` — JSON output with convergence info

## Current State (March 2026)
- Full simulation implemented and working
- Convergence detection added
- Hiring/firing costs added
- Government diagnostic prompt improved (shows causal chain for necessity sector failures)
- **NOT YET RUN** with all improvements — the first full test run is the immediate next step

## Pending Tasks (in priority order)
1. **Run capitalist preset** (`python -m econ_sim --system capitalist`) — first full test with all improvements
2. **Analyze results** and iterate on any issues
3. **Add confucian and islamic presets** — discussed values: confucian (0.70, 0.75, 0.70), islamic (0.55, 0.50, 0.85)
4. **Run comparative analysis** across presets
5. **Phase diagram sweep** across parameter space

## Intellectual Context
- The meta-game concept connects to: autocatalysis (chemistry→life), niche construction / extended phenotype / Baldwin Effect (biology), lobbying/corruption (economics), geopolitics (countries reshaping international order)
- Arrow's Impossibility Theorem informed governance design — no perfect voting system exists, so institutional design is always about tradeoffs
- Nordic preset tests whether you can engineer institutions that hold the meta-game bounded
- Cooperation substrates evolved: kin selection → reciprocal altruism → religion (guilt/shame) → secular law → financial infrastructure — each solving cooperation at larger scales with different failure modes

## V2 Roadmap: Global Geopolitics Simulation
Pinned for after V1 is complete. Same 3-axis framework one level up:
- Countries as agents with resource dependency graphs (oil, food, semiconductors, rare earths)
- Trade flows, military deterrence, chokepoints (Hormuz, Suez, Taiwan Strait, Malacca)
- Hybrid RL+LLM architecture: RL for fast equilibrium-finding, LLM for narrative/justification
- Canonical test case: Japan/Hormuz oil cascade (Japan has 8-9 months of oil reserves)
- Research question: "Is there a stable multipolar world order?"
- Estimated cost: ~$30-50/run

## Cost Estimates
- Single simulation run (V1): ~$2-5 depending on rounds and model mix
- Phase diagram sweep (25 points): ~$50-125
- V2 geopolitics run: ~$30-50/run
