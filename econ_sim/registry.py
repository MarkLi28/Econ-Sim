"""Run registry for the econ-sim project.

Tracks every simulation run for reproducibility, phase-diagram analysis,
and cost accountability. Backed by SQLite (single-writer assumption).

Usage as context manager (recommended):

    from econ_sim import registry

    with registry.start_run(config, sweep_id="m4_validation", seed=42) as run:
        result = env.run()
        run.complete(result, cost_summary=env.llm.cost_summary())
    # On exception inside the block, the run is auto-marked status='failed'.

CLI:

    python -m econ_sim.registry init
    python -m econ_sim.registry list [--preset nordic] [--limit 20]
    python -m econ_sim.registry stats

DB location:
  Default: ~/econ-sim-data/runs.db
  Override: set ECON_SIM_REGISTRY_DB to an absolute path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1

DEFAULT_DB_PATH = Path(
    os.environ.get(
        "ECON_SIM_REGISTRY_DB",
        str(Path.home() / "econ-sim-data" / "runs.db"),
    )
)

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id                  TEXT PRIMARY KEY,

    -- Provenance
    config_hash             TEXT NOT NULL,
    git_sha                 TEXT NOT NULL,
    git_dirty               INTEGER NOT NULL DEFAULT 0,
    seed                    INTEGER,

    -- Sweep grouping
    sweep_id                TEXT,
    sweep_name              TEXT,

    -- Lifecycle
    started_at              TEXT NOT NULL,
    ended_at                TEXT,
    duration_seconds        REAL,
    status                  TEXT NOT NULL CHECK (
                                status IN ('running', 'completed', 'failed', 'cancelled')
                            ),
    error_message           TEXT,

    -- Architecture (LLM-only vs hybrid validation)
    architecture            TEXT NOT NULL DEFAULT 'llm_only' CHECK (
                                architecture IN ('llm_only', 'hybrid_v1', 'rl_only')
                            ),
    tier                    TEXT NOT NULL DEFAULT 'toy' CHECK (
                                tier IN ('toy', 'us_calibrated')
                            ),

    -- Three research axes (denormalized for indexed phase-diagram queries)
    coordination            REAL NOT NULL,
    structure               REAL NOT NULL,
    meta_game               REAL NOT NULL,
    preset_label            TEXT,

    -- Scale
    num_workers             INTEGER NOT NULL,
    num_firms               INTEGER NOT NULL,
    rounds_max              INTEGER,
    rounds_run              INTEGER,

    -- Outcome (denormalized)
    regime                  TEXT,
    convergence_trend       TEXT,
    convergence_confidence  REAL,
    final_welfare           REAL,
    final_integrity         REAL,
    final_gini              REAL,
    final_unemployment      REAL,
    welfare_trend           REAL,
    integrity_trend         REAL,

    -- Cost & resource accounting
    total_cost_usd          REAL,
    total_llm_calls         INTEGER,
    total_input_tokens      INTEGER,
    total_output_tokens     INTEGER,

    -- Models used (so we can stratify results by model version later)
    model_opus              TEXT,
    model_sonnet            TEXT,
    model_haiku             TEXT,

    -- Blobs (full reconstruction)
    config_json             TEXT NOT NULL,
    summary_json            TEXT,
    cost_breakdown_json     TEXT,

    -- External reference to the recorder's full per-round JSON dump
    results_path            TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_axes      ON runs(coordination, structure, meta_game);
CREATE INDEX IF NOT EXISTS idx_runs_preset    ON runs(preset_label);
CREATE INDEX IF NOT EXISTS idx_runs_regime    ON runs(regime);
CREATE INDEX IF NOT EXISTS idx_runs_status    ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_sweep     ON runs(sweep_id);
CREATE INDEX IF NOT EXISTS idx_runs_arch_tier ON runs(architecture, tier);
CREATE INDEX IF NOT EXISTS idx_runs_started   ON runs(started_at);

INSERT OR IGNORE INTO schema_version (version, applied_at)
VALUES (1, datetime('now'));
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> Path:
    """Create the database file and apply the schema (idempotent)."""
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _git_info() -> tuple[str, bool]:
    """Return (sha, dirty). Falls back to ('unknown', False) outside a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha, bool(dirty_out)
    except Exception:
        return "unknown", False


def _serializable_config(config) -> dict:
    """Extract the parameters that define a run for hashing and storage."""
    return {
        "coordination_mechanism":      config.coordination_mechanism,
        "planning_structure":          config.planning_structure,
        "meta_game_constraint":        config.meta_game_constraint,
        "num_rounds":                  config.num_rounds,
        "min_rounds":                  config.min_rounds,
        "initial_firm_capital":        config.initial_firm_capital,
        "initial_government_treasury": config.initial_government_treasury,
        "base_wage_level":             config.base_wage_level,
        "hiring_cost_factor":          config.hiring_cost_factor,
        "firing_cost_factor":          config.firing_cost_factor,
        "integrity_recovery_rate":     config.integrity_recovery_rate,
        "suffering_pressure_threshold": config.suffering_pressure_threshold,
        "temperature":                 config.temperature,
        "max_tokens":                  config.max_tokens,
        "model_opus":                  config.model_opus,
        "model_sonnet":                config.model_sonnet,
        "model_haiku":                 config.model_haiku,
        "industry_configs": {
            i.value: {
                "num_firms":           ic.num_firms,
                "base_price":          ic.base_price,
                "productivity_factor": ic.productivity_factor,
                "input_industry":      ic.input_industry.value if ic.input_industry else None,
                "input_ratio":         ic.input_ratio,
                "model_tier":          ic.model_tier,
                "is_consumer_good":    ic.is_consumer_good,
                "is_necessity":        ic.is_necessity,
                "necessity_units":     ic.necessity_units,
            }
            for i, ic in config.industry_configs.items()
        },
        "worker_profiles": [
            {
                "name":                p.name,
                "skill_level":          p.skill_level,
                "representative_count": p.representative_count,
                "initial_savings":      p.initial_savings,
                "industry_preferences": [i.value for i in p.industry_preferences],
            }
            for p in config.worker_profiles
        ],
    }


def _config_hash(config) -> str:
    payload = json.dumps(_serializable_config(config), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Run lifecycle: RunHandle + start_run()
# ---------------------------------------------------------------------------

class RunHandle:
    """Context manager wrapping a single simulation run.

    On enter:  insert a row with status='running'.
    On exit:   if complete() was called, status stays 'completed';
               otherwise the run is marked 'failed' with the exception text.
    """

    def __init__(
        self,
        run_id: str,
        config,
        db_path: Path,
        sweep_id: Optional[str],
        sweep_name: Optional[str],
        seed: Optional[int],
        architecture: str,
        tier: str,
    ):
        self.run_id = run_id
        self.config = config
        self.db_path = db_path
        self.sweep_id = sweep_id
        self.sweep_name = sweep_name
        self.seed = seed
        self.architecture = architecture
        self.tier = tier
        self._completed = False
        self._start_time: Optional[float] = None

    def __enter__(self) -> "RunHandle":
        init_db(self.db_path)
        self._start_time = time.time()
        sha, dirty = _git_info()
        num_workers = sum(p.representative_count for p in self.config.worker_profiles)
        num_firms = sum(ic.num_firms for ic in self.config.industry_configs.values())

        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO runs (
                    run_id, config_hash, git_sha, git_dirty, seed,
                    sweep_id, sweep_name,
                    started_at, status, architecture, tier,
                    coordination, structure, meta_game, preset_label,
                    num_workers, num_firms, rounds_max,
                    model_opus, model_sonnet, model_haiku,
                    config_json
                ) VALUES (?,?,?,?,?, ?,?, ?,?,?,?, ?,?,?,?, ?,?,?, ?,?,?, ?)""",
                (
                    self.run_id, _config_hash(self.config), sha, int(dirty), self.seed,
                    self.sweep_id, self.sweep_name,
                    _now_iso(), "running", self.architecture, self.tier,
                    self.config.coordination_mechanism,
                    self.config.planning_structure,
                    self.config.meta_game_constraint,
                    self.config.label(),
                    num_workers, num_firms, self.config.num_rounds,
                    self.config.model_opus, self.config.model_sonnet, self.config.model_haiku,
                    json.dumps(_serializable_config(self.config), sort_keys=True),
                ),
            )
            conn.commit()
        return self

    def complete(
        self,
        result: dict,
        cost_summary: Optional[dict] = None,
        results_path: Optional[str] = None,
    ) -> None:
        """Record successful completion. Pulls fields from result dict."""
        summary = result.get("summary", {}) or {}
        convergence = result.get("convergence", {}) or {}
        regime = result.get("regime")

        total_cost = 0.0
        total_calls = 0
        total_in = 0
        total_out = 0
        if cost_summary:
            for stats in cost_summary.values():
                total_cost += float(stats.get("cost_usd", 0.0))
                total_calls += int(stats.get("calls", 0))
                total_in += int(stats.get("input_tokens", 0))
                total_out += int(stats.get("output_tokens", 0))

        with _connect(self.db_path) as conn:
            conn.execute(
                """UPDATE runs SET
                    ended_at=?, duration_seconds=?, status='completed',
                    rounds_run=?, regime=?,
                    convergence_trend=?, convergence_confidence=?,
                    final_welfare=?, final_integrity=?, final_gini=?, final_unemployment=?,
                    welfare_trend=?, integrity_trend=?,
                    total_cost_usd=?, total_llm_calls=?,
                    total_input_tokens=?, total_output_tokens=?,
                    summary_json=?, cost_breakdown_json=?, results_path=?
                WHERE run_id=?""",
                (
                    _now_iso(),
                    time.time() - (self._start_time or time.time()),
                    summary.get("rounds_run"),
                    regime,
                    convergence.get("trend"),
                    convergence.get("confidence"),
                    summary.get("final_welfare"),
                    summary.get("final_integrity"),
                    summary.get("final_gini"),
                    summary.get("final_unemployment"),
                    summary.get("welfare_trend"),
                    summary.get("integrity_trend"),
                    total_cost, total_calls, total_in, total_out,
                    json.dumps(summary),
                    json.dumps(cost_summary or {}),
                    results_path,
                    self.run_id,
                ),
            )
            conn.commit()
        self._completed = True

    def fail(self, error: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """UPDATE runs SET
                    ended_at=?, duration_seconds=?, status='failed', error_message=?
                WHERE run_id=?""",
                (
                    _now_iso(),
                    time.time() - (self._start_time or time.time()),
                    error[:2000],
                    self.run_id,
                ),
            )
            conn.commit()
        self._completed = True

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._completed:
            if exc_val is not None:
                self.fail(f"{exc_type.__name__}: {exc_val}")
            else:
                self.fail("Run exited without complete() being called")
        return False  # never suppress exceptions


def start_run(
    config,
    sweep_id: Optional[str] = None,
    sweep_name: Optional[str] = None,
    seed: Optional[int] = None,
    architecture: str = "llm_only",
    tier: str = "toy",
    db_path: Path = DEFAULT_DB_PATH,
) -> RunHandle:
    """Begin tracking a simulation run; returns a context-manager handle."""
    return RunHandle(
        run_id=str(uuid.uuid4()),
        config=config,
        db_path=db_path,
        sweep_id=sweep_id,
        sweep_name=sweep_name,
        seed=seed,
        architecture=architecture,
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def list_runs(
    db_path: Path = DEFAULT_DB_PATH,
    preset: Optional[str] = None,
    regime: Optional[str] = None,
    sweep_id: Optional[str] = None,
    status: Optional[str] = None,
    architecture: Optional[str] = None,
    tier: Optional[str] = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    where: list[str] = []
    params: list[Any] = []
    for col, val in [
        ("preset_label", preset),
        ("regime", regime),
        ("sweep_id", sweep_id),
        ("status", status),
        ("architecture", architecture),
        ("tier", tier),
    ]:
        if val is not None:
            where.append(f"{col} = ?")
            params.append(val)
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT run_id, started_at, preset_label, coordination, structure, meta_game,
               status, regime, convergence_trend,
               final_welfare, final_integrity, final_gini, total_cost_usd
        FROM runs
        {where_clause}
        ORDER BY started_at DESC
        LIMIT ?
    """
    params.append(limit)
    with _connect(db_path) as conn:
        return list(conn.execute(sql, params))


def stats_summary(db_path: Path = DEFAULT_DB_PATH) -> dict:
    with _connect(db_path) as conn:
        by_status = conn.execute(
            """SELECT status, COUNT(*) AS n,
                      COALESCE(SUM(total_cost_usd), 0) AS cost
               FROM runs GROUP BY status"""
        ).fetchall()
        by_regime = conn.execute(
            """SELECT regime, COUNT(*) AS n
               FROM runs WHERE status = 'completed' GROUP BY regime"""
        ).fetchall()
        totals = conn.execute(
            """SELECT COUNT(*) AS n,
                      COALESCE(SUM(total_cost_usd), 0) AS cost,
                      COALESCE(SUM(total_llm_calls), 0) AS calls
               FROM runs"""
        ).fetchone()
    return {
        "total_runs": totals["n"],
        "total_cost_usd": totals["cost"],
        "total_llm_calls": totals["calls"],
        "by_status": {
            r["status"]: {"count": r["n"], "cost_usd": r["cost"]}
            for r in by_status
        },
        "by_regime": {(r["regime"] or "(null)"): r["n"] for r in by_regime},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_runs_table(rows: list[sqlite3.Row]) -> str:
    if not rows:
        return "(no runs)"
    header = (
        f"{'started':<20} {'preset':<22} {'c':>4} {'s':>4} {'m':>4} "
        f"{'status':<10} {'regime':<22} {'trend':<12} "
        f"{'W':>5} {'I':>5} {'G':>5} {'$':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{(r['started_at'] or '')[:19]:<20} "
            f"{(r['preset_label'] or '-')[:22]:<22} "
            f"{r['coordination']:>4.2f} {r['structure']:>4.2f} {r['meta_game']:>4.2f} "
            f"{r['status']:<10} "
            f"{(r['regime'] or '-')[:22]:<22} "
            f"{(r['convergence_trend'] or '-')[:12]:<12} "
            f"{(r['final_welfare']    or 0):>5.3f} "
            f"{(r['final_integrity']  or 0):>5.2f} "
            f"{(r['final_gini']       or 0):>5.2f} "
            f"{(r['total_cost_usd']   or 0):>7.4f}"
        )
    return "\n".join(lines)


def _cli() -> None:
    p = argparse.ArgumentParser(prog="python -m econ_sim.registry")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH),
                   help=f"Registry SQLite path (default: {DEFAULT_DB_PATH})")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Create the registry database (idempotent)")

    p_list = sub.add_parser("list", help="List recent runs")
    p_list.add_argument("--preset")
    p_list.add_argument("--regime")
    p_list.add_argument("--sweep-id")
    p_list.add_argument("--status",
                        choices=["running", "completed", "failed", "cancelled"])
    p_list.add_argument("--architecture",
                        choices=["llm_only", "hybrid_v1", "rl_only"])
    p_list.add_argument("--tier", choices=["toy", "us_calibrated"])
    p_list.add_argument("--limit", type=int, default=20)

    sub.add_parser("stats", help="Aggregate counts and costs")

    args = p.parse_args()
    db_path = Path(args.db)

    if args.cmd == "init":
        path = init_db(db_path)
        print(f"Registry initialized at: {path}")
    elif args.cmd == "list":
        rows = list_runs(
            db_path,
            preset=args.preset,
            regime=args.regime,
            sweep_id=args.sweep_id,
            status=args.status,
            architecture=args.architecture,
            tier=args.tier,
            limit=args.limit,
        )
        print(_format_runs_table(rows))
    elif args.cmd == "stats":
        s = stats_summary(db_path)
        print(f"Total runs:      {s['total_runs']}")
        print(f"Total LLM calls: {s['total_llm_calls']}")
        print(f"Total cost:      ${s['total_cost_usd']:.2f}")
        print()
        print("By status:")
        for status, info in sorted(s["by_status"].items()):
            print(f"  {status:<12} {info['count']:>5}   ${info['cost_usd']:>8.2f}")
        if s["by_regime"]:
            print()
            print("By regime (completed only):")
            for regime, n in sorted(s["by_regime"].items(), key=lambda kv: -kv[1]):
                print(f"  {regime:<28} {n:>5}")


if __name__ == "__main__":
    _cli()
