"""Backtest subpackage."""
from core.backtest.attribution import (
    excess_stats,
    market_attribution,
    ols_nw,
    run_attribution_suite,
    sector_neutral_returns,
    turnover_variants,
)
from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.robustness import (
    Panel,
    build_panel,
    cost_sensitivity,
    evaluate_panel,
    parameter_sweep,
    run_robustness_suite,
    universe_bootstrap,
)
from core.backtest.sectors import get_sectors, sector_weights
from core.backtest.sector_neutral import (
    add_sector_scores,
    largest_remainder,
    sector_neutral_target,
    sector_tilt,
)

__all__ = [
    "BacktestConfig",
    "run_backtest",
    "Panel",
    "build_panel",
    "evaluate_panel",
    "universe_bootstrap",
    "parameter_sweep",
    "cost_sensitivity",
    "run_robustness_suite",
    "ols_nw",
    "market_attribution",
    "excess_stats",
    "sector_neutral_returns",
    "turnover_variants",
    "run_attribution_suite",
    "get_sectors",
    "sector_weights",
    "add_sector_scores",
    "largest_remainder",
    "sector_neutral_target",
    "sector_tilt",
]
