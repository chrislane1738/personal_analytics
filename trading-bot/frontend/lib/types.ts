/** A single backtest run summary */
export interface Run {
  id: string;
  run_id: string;
  mode: string;
  strategy: string;
  strategy_name: string;
  symbols: string[];
  config: string | null;
  start_date: string;
  end_date: string;
  timeframe: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  initial_capital: number;
  final_value: number;
  total_return: number | null;
  sharpe: number | null;
  sharpe_ratio: number | null;
  max_drawdown: number | null;
  total_trades: number | null;
  win_rate: number | null;
  created_at: string;
  completed_at: string | null;
  parameters: Record<string, unknown>;
  full_metrics: Record<string, unknown> | null;
}

/** Paginated run list response */
export interface RunListResponse {
  runs: Run[];
  total: number;
}

/** A single trade record */
export interface Trade {
  id: string;
  trade_id: string;
  run_id: string;
  symbol: string;
  side: "long" | "short";
  direction: string;
  entry_time: string;
  exit_time: string | null;
  entry_date: string | null;
  exit_date: string | null;
  entry_price: number;
  exit_price: number | null;
  quantity: number;
  pnl: number | null;
  pnl_pct: number | null;
  commission: number;
  slippage: number;
  mae: number | null;
  mfe: number | null;
  duration_seconds: number | null;
  entry_reason: string;
  exit_reason: string;
  option_type: string | null;
  strike: number | null;
  expiration: string | null;
}

/** A single point on the equity curve */
export interface EquityCurvePoint {
  date: string;
  strategy_value: number;
  benchmark_value: number | null;
  drawdown?: number;
  cash?: number;
  positions_value?: number;
}

/** Performance statistics per market regime */
export interface RegimeStat {
  regime: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  trade_count: number;
  trades: number;
  avg_pnl: number;
  total_pnl: number;
  best_trade: number;
  worst_trade: number;
}

/** Monte Carlo simulation result */
export interface MonteCarloResult {
  simulations: number;
  percentiles: {
    p5: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
  };
  probability_of_profit: number;
  probability_of_ruin: number;
  expected_max_drawdown: number;
  median_final_equity: number;
  is_outlier: boolean;
  fan_chart: FanChartPoint[];
  drawdown_distribution: number[];
  paths: number[][];
}

/** A single point on the Monte Carlo fan chart */
export interface FanChartPoint {
  step: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  actual: number | null;
}

/** Options analytics response */
export interface OptionsAnalyticsResponse {
  total_collected: number;
  total_paid: number;
  net_premium: number;
  assignment_count: number;
  total_short_options: number;
  assignment_rate: number;
  win_rate_by_dte: DteBucketWinRate[];
  greeks_timeline: GreeksPoint[];
}

/** Win rate for a DTE bucket */
export interface DteBucketWinRate {
  bucket: string;
  win_rate: number;
  trade_count: number;
}

/** A single point on the Greeks timeline */
export interface GreeksPoint {
  timestamp: string;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
}

/** A single walk-forward OOS window with params, metrics, and equity */
export interface WalkForwardWindow {
  window_index: number;
  train_start: string;
  train_end: string;
  oos_start: string;
  oos_end: string;
  best_params: Record<string, number>;
  train_metrics: Record<string, number>;
  oos_metrics: Record<string, number>;
  oos_trades: Trade[];
  oos_equity_curve: EquityCurvePoint[];
}

/** Summary of a walk-forward study (used in list views) */
export interface WalkForwardStudySummary {
  study_id: string;
  strategy_name: string;
  start_date: string;
  end_date: string;
  train_months: number;
  oos_months: number;
  step_months: number;
  objective: string;
  status: string;
  created_at: string;
  aggregate?: Record<string, number>;
}

/** Full walk-forward study with windows and analysis */
export interface WalkForwardStudy extends WalkForwardStudySummary {
  config: string;
  initial_capital: number;
  windows: WalkForwardWindow[];
  stitched_equity_curve: EquityCurvePoint[];
  parameter_stability: Record<string, number[]>;
  monte_carlo: Record<string, MonteCarloResult>;
}

/** Paginated walk-forward study list response */
export interface WalkForwardListResponse {
  studies: WalkForwardStudySummary[];
  total: number;
}

/** Walk-forward study execution status */
export interface WalkForwardStatusResponse {
  study_id: string;
  status: string;
  windows_completed: number;
  windows_total: number;
  current_phase: string;
}
