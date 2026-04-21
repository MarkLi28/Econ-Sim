# Econ-Sim — Session Notes

A running log of what's been built, the design decisions made along the way, and the reasoning behind each one. Meant as continuity context for future sessions (and future forks).

---

## 1. The Core Idea

**Research question:** *Is there a mixed-system equilibrium where the meta-game (lobbying/corruption) exists but is bounded?*

The "meta-game" is a universal pattern: entities stop competing *within* the rules and start reshaping the rules themselves. Autocatalysis in chemistry, niche construction in biology, lobbying/regulatory capture in economics, geopolitics between countries — they're all instances of the same phenomenon at different scales.

**Why an economic simulation?** Because economies are the cleanest testbed we have: the rules (taxes, subsidies, prices, wages) are explicit, the actors (firms, workers, government) have well-defined incentives, and the meta-game (firms lobbying the government) is already a well-studied real-world phenomenon.

**Rationale for the 3-axis framework:**
- `coordination_mechanism` (0=market, 1=planning) — this is the classic capitalism/socialism axis
- `planning_structure` (0=distributed, 1=concentrated) — but that axis alone collapses "Nordic social democracy" and "Soviet central planning" into the same bucket, so we need a second axis for *how concentrated* governance power is
- `meta_game_constraint` (0=open lobbying, 1=suppressed) — this is the actual variable the research question is about. Everything else exists to set up the conditions under which this variable's effect is interesting.

---

## 2. Architecture Decisions

### Why LLM agents instead of hand-coded rules?

Traditional agent-based economic models hard-code decision rules (e.g., "firms set price = marginal cost + 10%"). That bakes in the assumption of what "rational" behavior is.

We wanted emergent behavior — firms that *might* try to lobby, collude, deceive, or cooperate in unexpected ways. LLMs give us that because they can read the game state, reason about it, and pick actions from a rich space. Cost is the tradeoff (~$2–5/run), but it's what makes the meta-game observable rather than scripted.

### Model tiering (Opus / Sonnet / Haiku)

Not every agent needs a $75/MTok model. We picked by decision complexity:
- **Opus**: Government (policy reasoning over many variables), Manufacturing (upstream supplier, strategic leverage), Technology (high-margin, complex demand forecasting)
- **Sonnet**: Housing (supply chain dependent, medium complexity)
- **Haiku**: Agriculture (simple commodity), all Workers (individual decisions are mostly local — do I quit this job, do I buy food)

Rationale: worker and agriculture decisions are narrow; government and leverage-point firms drive emergent dynamics and need the reasoning headroom.

### 4-sector supply chain, not 1

A one-industry economy can't exhibit the dynamics we care about. Four sectors give us:
- **Necessities vs discretionary** (food/shelter vs tech) — so basic_needs_rate is a meaningful welfare signal
- **B2B dependency** (Manufacturing → Housing/Tech) — so supply chain failures cascade
- **Asymmetric leverage** — Manufacturing can hold up downstream industries, which is the kind of structural power that motivates lobbying

---

## 3. The Welfare Function

```
W = 0.30*(1-unemployment)
  + 0.25*(1-gini)
  + 0.25*basic_needs_rate
  + 0.10*gdp_growth
  + 0.10*fiscal_health
```

**Why these weights?** The first three (employment, equality, basic needs) are the outcomes people actually experience day-to-day, so they dominate. Growth and fiscal health matter but are means, not ends — they enable the first three. We deliberately did *not* use GDP as the headline metric, because GDP-maximizing policies can pass even when half the population is unemployed.

---

## 4. Problems We Hit and How We Fixed Them

### Problem: Employment cycling
Early runs showed firms oscillating between "hire everyone" and "fire everyone" every other round. Pure profit-maximization over one round treats labor as costlessly adjustable.

**Fix:** Hiring/firing costs — 0.5× wage per new hire, 1.0× wage per layoff. Firing costs > hiring costs because that's also empirically true (severance, morale, institutional-knowledge loss).

**Why not just smooth the policy?** Because the cycling was a real artifact of the decision rule, not a plotting issue. Making layoffs expensive is how real labor markets prevent this, so we replicated the mechanism, not the output.

### Problem: Arbitrary round cutoff
Originally we capped runs at 10 or 15 rounds. But a scenario that's still transient at round 10 and reaches equilibrium at round 20 would be misclassified as whatever its round-10 state happened to be.

**Fix:** Convergence detection. The simulation runs up to 40 rounds but stops as soon as the welfare trajectory is statistically classifiable:
- **Collapse**: welfare < 0.10 for 3 consecutive rounds
- **Oscillation**: 70%+ sign changes in round-to-round diffs + amplitude > 0.02
- **Stable**: slope < 0.005, low variance on welfare and integrity
- **Strong trend**: R² > 0.80 and |slope| > 0.008
- **Weak trend**: R² > 0.60 and |slope| > 0.005 after 15+ rounds

**Why these thresholds?** They're deliberately conservative — the simulation would rather run longer than falsely declare convergence. R² > 0.8 is a high bar for trend confidence; the oscillation detector requires *both* sign-change frequency and minimum amplitude so tiny numerical wobble doesn't get labeled oscillating.

`min_rounds=10` is a floor because early rounds always have transient dynamics (firms setting initial prices, workers sorting into jobs) and we don't want those classified as "stable."

### Problem: Government couldn't diagnose necessity-sector failures
Government saw `basic_needs_rate=0%` but not *why* — was food missing? Shelter? Was the issue production, employment, or pricing?

**Fix:** Added a NECESSITY SECTOR DIAGNOSIS section to the government prompt that walks the causal chain: employees → production → inventory → price → demand. So when food production is zero, the gov sees that Agriculture has zero employees, not just "people are hungry."

---

## 5. The Presets

| Preset | coord | struct | meta | What it tests |
|--------|-------|--------|------|---------------|
| capitalist | 0.10 | 0.20 | 0.10 | Can laissez-faire with weak institutions hold together? |
| crony | 0.10 | 0.30 | 0.00 | No lobbying constraints — how fast does capture happen? |
| socialist | 0.85 | 0.80 | 0.65 | Central planning, concentrated, moderately corruption-resistant |
| nordic | 0.50 | 0.30 | 0.80 | Mixed economy, distributed governance, hardened against lobbying |
| mixed | 0.40 | 0.40 | 0.45 | Middle of the parameter space |

**Why Nordic at (0.50, 0.30, 0.80)?** That's the preset that directly tests the research question. High meta_game_constraint says "lobbying is hard." Moderate coordination and distributed structure say "the state is active but not monolithic." If any preset should exhibit a bounded meta-game, this one should. If Nordic fails, the research question probably doesn't have a positive answer — at least not in this parameter space.

**Pending:** confucian (0.70, 0.75, 0.70) and islamic (0.55, 0.50, 0.85) — to broaden beyond Western archetypes.

---

## 6. Intellectual Threads Worth Preserving

These came up in conversation and shape how to think about extensions:

- **Meta-game is substrate-agnostic.** Autocatalysis (chemistry → life), niche construction (species shaping their environment), extended phenotype, Baldwin Effect, cultural evolution — all the same structural move. That's what makes this project more than economic modeling: it's a testbed for a universal pattern.
- **Cooperation substrates stack.** Kin selection → reciprocal altruism → religion (guilt/shame enforcement) → secular law → financial/institutional infrastructure. Each one solves cooperation at a larger scale, and each has its own failure mode. The meta-game is the pattern by which substrates *get* reshaped — agents escape one substrate by building the next.
- **Arrow's Impossibility Theorem** informed governance design: there is no perfect voting/aggregation rule, so every institutional design is a tradeoff, not an optimization. The `planning_structure` axis is basically "where on the Arrow-tradeoff surface does this system sit."
- **Religion-to-government transition.** Governments historically emerged *alongside or after* religion, not before. Religion solved cooperation via internalized guilt; government solved it via externalized enforcement. Modern secular moral codes are a third substrate — personal experience + dense social interaction instead of doctrinal authority.

---

## 7. V2: Global Geopolitics

Pinned for after V1 converges. Same 3-axis framework one level up, applied to countries instead of firms:

- **Agents:** countries, with resource dependency graphs (oil, food, semiconductors, rare earths)
- **Mechanics:** trade flows, military deterrence, chokepoints (Hormuz, Suez, Taiwan Strait, Malacca)
- **Architecture:** hybrid RL + LLM. RL handles fast equilibrium-finding on economic/military sub-games; LLM handles narrative, justification, and coalition-level strategy. Pure LLM is too slow and expensive for the search; pure RL can't produce the kind of meta-move reasoning we care about.
- **Canonical test case:** Japan/Hormuz oil cascade. Japan holds ~8–9 months of oil reserves; if Hormuz closes, what's the cascade?
- **Research question:** *Is there a stable multipolar world order?* — direct analog of V1's question at a larger scale.
- **Cost estimate:** ~$30–50/run. Phase diagram sweeps would need the RL portion to hold most of the compute load.

---

## 8. Where We Are Right Now

- All V1 code is implemented: 13-phase loop, 4 industries, LLM agents across 3 model tiers, convergence detection, hiring/firing costs, government necessity-diagnostic prompt.
- `CLAUDE.md` written at repo root so future Claude Code sessions have full project context.
- **Next concrete step:** run `python -m econ_sim --system capitalist` — first full test with every improvement in place. Then analyze, iterate, add the confucian/islamic presets, and do the cross-preset comparison.
