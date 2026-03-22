/** A single backtest run summary */
export interface Run {
  id: string;
  strategy: string;
  symbols: string[];
  start_date: string;
  end_date: string;
  timeframe: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  total_return: number | null;
  sharpe_ratio: number | null;
  max_drawdown: number | null;
  total_trades: number | null;
  win_rate: number | null;
  created_at: string;
  completed_at: string | null;
  parameters: Record<string, unknown>;
}

/** Paginated run list response */
export interface RunListResponse {
  items: Run[];
  total: number;
  page: number;
  page_size: number;
}

/** A single trade record */
export interface Trade {
  id: string;
  run_id: string;
  symbol: string;
  side: "long" | "short";
  entry_time: string;
  exit_time: string | null;
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
}

/** A single point on the equity curve */
export interface EquityCurvePoint {
  timestamp: string;
  equity: number;
  drawdown: number;
  benchmark: number | null;
  cash: number;
  positions_value: number;
}

/** Performance statistics per market regime */
export interface RegimeStat {
  regime: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  trade_count: number;
  avg_pnl: number;
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
