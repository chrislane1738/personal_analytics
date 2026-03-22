"""Report generation: console text and interactive HTML reports."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


class ReportGenerator:
    def __init__(
        self,
        run_id: str,
        strategy_name: str,
        metrics: dict,
        trade_log=None,
        monte_carlo_result=None,
        regime_stats=None,
        options_analytics=None,
        equity_curve=None,
        benchmark_curve=None,
    ):
        self.run_id = run_id
        self.strategy_name = strategy_name
        self.metrics = metrics
        self.trade_log = trade_log
        self.monte_carlo = monte_carlo_result
        self.regime_stats = regime_stats
        self.options_analytics = options_analytics
        self.equity_curve = equity_curve or []
        self.benchmark_curve = benchmark_curve or []

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    def generate_console_report(self) -> str:
        """Generate formatted text report for terminal display."""
        m = self.metrics
        lines = [
            "=" * 60,
            f"  BACKTEST REPORT: {self.strategy_name}",
            f"  Period: {m.get('start_date', '?')} \u2192 {m.get('end_date', '?')}",
            "=" * 60,
            "",
            "  PERFORMANCE",
            f"  Total Return:    {m.get('total_return', 0)*100:>8.1f}%",
            f"  CAGR:            {m.get('cagr', 0)*100:>8.1f}%",
            f"  Sharpe Ratio:    {m.get('sharpe', 0):>8.2f}",
            f"  Sortino Ratio:   {m.get('sortino', 0):>8.2f}",
            f"  Calmar Ratio:    {m.get('calmar', 0):>8.2f}",
            "",
            "  RISK",
            f"  Max Drawdown:    {m.get('max_drawdown_pct', 0)*100:>8.1f}%",
            f"  Volatility:      {m.get('annualized_volatility', 0)*100:>8.1f}%",
            f"  VaR (95%):       {m.get('var_95', 0)*100:>8.2f}%",
            "",
            "  TRADES",
            f"  Total Trades:    {m.get('total_trades', 0):>8d}",
            f"  Win Rate:        {m.get('win_rate', 0)*100:>8.1f}%",
            f"  Profit Factor:   {m.get('profit_factor', 0):>8.2f}",
            f"  Expectancy:      ${m.get('expectancy', 0):>7.2f}",
            "",
        ]

        # Benchmark comparison
        if "benchmark_total_return" in m:
            lines.extend(
                [
                    "  BENCHMARK COMPARISON (SPY Buy & Hold)",
                    f"  Benchmark Return:{m.get('benchmark_total_return', 0)*100:>8.1f}%",
                    f"  Alpha:           {m.get('alpha', 0)*100:>8.2f}%",
                    f"  Beta:            {m.get('beta', 0):>8.2f}",
                    "",
                ]
            )

        # Monte Carlo
        if self.monte_carlo:
            mc = self.monte_carlo
            lines.extend(
                [
                    "  MONTE CARLO",
                    f"  Median Equity:   ${mc.median_final_equity:>12,.0f}",
                    f"  5th Percentile:  ${mc.percentile_5:>12,.0f}",
                    f"  95th Percentile: ${mc.percentile_95:>12,.0f}",
                    f"  P(Ruin):         {mc.probability_of_ruin*100:>8.2f}%",
                    f"  Outlier:         {'YES \u26a0\ufe0f' if mc.is_outlier else 'No'}",
                    "",
                ]
            )

        # Regime stats
        if self.regime_stats:
            lines.append("  REGIME PERFORMANCE")
            lines.append(f"  {'Regime':<12} {'Trades':>7} {'Win%':>7} {'Avg P&L':>10}")
            lines.append(f"  {'-'*38}")
            for regime, stats in self.regime_stats.items():
                lines.append(
                    f"  {regime:<12} {stats.get('trades', 0):>7d} "
                    f"{stats.get('win_rate', 0)*100:>6.1f}% "
                    f"${stats.get('avg_pnl', 0):>9.2f}"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_html_report(self, output_dir: str = "reports") -> str:
        """Generate interactive HTML report with Plotly charts.
        Returns path to generated file."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/{self.run_id}.html"

        # Build charts
        figs = []

        # 1. Equity curve with benchmark
        if self.equity_curve:
            fig = go.Figure()
            dates = [d for d, v in self.equity_curve]
            values = [v for d, v in self.equity_curve]
            fig.add_trace(
                go.Scatter(x=dates, y=values, name="Strategy", line=dict(color="blue"))
            )
            if self.benchmark_curve:
                b_dates = [d for d, v in self.benchmark_curve]
                b_values = [v for d, v in self.benchmark_curve]
                fig.add_trace(
                    go.Scatter(
                        x=b_dates,
                        y=b_values,
                        name="Benchmark (SPY)",
                        line=dict(color="gray", dash="dash"),
                    )
                )
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
            )
            figs.append(("equity-curve", fig))

        # 2. Drawdown chart
        if self.equity_curve:
            dates = [d for d, v in self.equity_curve]
            values = [v for d, v in self.equity_curve]
            running_max = []
            peak = 0
            for v in values:
                peak = max(peak, v)
                running_max.append(peak)
            drawdowns = [
                (v - p) / p * 100 if p > 0 else 0
                for v, p in zip(values, running_max)
            ]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=dates, y=drawdowns, fill="tozeroy", line=dict(color="red"))
            )
            fig.update_layout(
                title="Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown %"
            )
            figs.append(("drawdown", fig))

        # 3. Monthly returns heatmap
        if self.equity_curve and len(self.equity_curve) > 30:
            monthly = self._compute_monthly_returns()
            if monthly:
                years = sorted(set(m[0] for m in monthly))
                months = list(range(1, 13))
                z = [
                    [
                        next(
                            (r for y2, m2, r in monthly if y2 == y and m2 == mo),
                            None,
                        )
                        for mo in months
                    ]
                    for y in years
                ]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=z,
                        x=[
                            "Jan",
                            "Feb",
                            "Mar",
                            "Apr",
                            "May",
                            "Jun",
                            "Jul",
                            "Aug",
                            "Sep",
                            "Oct",
                            "Nov",
                            "Dec",
                        ],
                        y=[str(y) for y in years],
                        colorscale="RdYlGn",
                        zmid=0,
                        text=[
                            [f"{v:.1f}%" if v is not None else "" for v in row]
                            for row in z
                        ],
                        texttemplate="%{text}",
                        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%",
                    )
                )
                fig.update_layout(title="Monthly Returns Heatmap")
                figs.append(("monthly-returns", fig))

        # 4. Trade log table (basic HTML table)
        trade_table_html = self._build_trade_table()

        # Assemble HTML
        console_report = self.generate_console_report()
        charts_html = ""
        for div_id, fig in figs:
            charts_html += (
                f'<div id="{div_id}">'
                f"{fig.to_html(full_html=False, include_plotlyjs=False)}"
                f"</div>\n"
            )

        html = f"""<!DOCTYPE html>
<html><head>
<title>Backtest Report: {self.strategy_name}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ font-family: monospace; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
h1, h2 {{ color: #0f3460; }}
pre {{ background: #16213e; padding: 15px; border-radius: 5px; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #333; padding: 8px; text-align: right; }}
th {{ background: #0f3460; }}
.positive {{ color: #00b894; }}
.negative {{ color: #d63031; }}
</style>
</head><body>
<h1>Backtest Report: {self.strategy_name}</h1>
<pre>{console_report}</pre>
{charts_html}
<h2>Trade Log</h2>
{trade_table_html}
</body></html>"""

        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_monthly_returns(self):
        """Returns list of (year, month, return_pct) tuples."""
        if not self.equity_curve:
            return []
        monthly = []
        prev_value = self.equity_curve[0][1]
        current_month = self.equity_curve[0][0].month
        current_year = self.equity_curve[0][0].year
        month_start_value = prev_value

        for d, v in self.equity_curve[1:]:
            if d.month != current_month or d.year != current_year:
                ret = (
                    (prev_value - month_start_value) / month_start_value * 100
                    if month_start_value > 0
                    else 0
                )
                monthly.append((current_year, current_month, ret))
                month_start_value = prev_value
                current_month = d.month
                current_year = d.year
            prev_value = v
        # Last month
        ret = (
            (prev_value - month_start_value) / month_start_value * 100
            if month_start_value > 0
            else 0
        )
        monthly.append((current_year, current_month, ret))
        return monthly

    def _build_trade_table(self):
        if not self.trade_log or not self.trade_log.trades:
            return "<p>No trades.</p>"
        rows = ""
        for t in self.trade_log.trades:
            pnl_class = "positive" if t.pnl > 0 else "negative"
            rows += f"""<tr>
<td>{t.symbol}</td><td>{t.direction}</td>
<td>{t.entry_date}</td><td>{t.exit_date}</td>
<td>${t.entry_price:.2f}</td><td>${t.exit_price:.2f}</td>
<td>{t.quantity}</td>
<td class="{pnl_class}">${t.pnl:.2f}</td>
<td class="{pnl_class}">{t.pnl_pct*100:.2f}%</td>
<td>{t.entry_reason}</td><td>{t.exit_reason}</td>
</tr>"""
        return f"""<table>
<tr><th>Symbol</th><th>Direction</th><th>Entry</th><th>Exit</th>
<th>Entry$</th><th>Exit$</th><th>Qty</th><th>P&amp;L</th><th>P&amp;L%</th>
<th>Entry Reason</th><th>Exit Reason</th></tr>
{rows}</table>"""
