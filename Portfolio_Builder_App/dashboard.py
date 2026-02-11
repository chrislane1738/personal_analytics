import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Builder", layout="wide")

PORTFOLIO_VALUE = 100_000

# --- Data Fetching (cached) ---

@st.cache_data(ttl=3600)
def fetch_ticker_info(ticker):
    """Fetch live fundamental data for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName") or ticker,
            "price": info.get("regularMarketPrice") or info.get("previousClose"),
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "beta": info.get("beta"),
            "div_yield": info.get("dividendYield"),
            "debt_equity": info.get("debtToEquity"),
            "sector": info.get("sector") or "N/A",
            "country": info.get("country") or "Unknown",
            "market_cap": info.get("marketCap") or 0,
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_prices(tickers, start=None):
    """Fetch historical close prices for a list of tickers."""
    if start is None:
        start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data[["Close"]].rename(columns={"Close": tickers[0]})
    return data.ffill().dropna()


def categorize_market_cap(mc):
    if mc > 200e9:
        return "Large Cap"
    elif mc > 10e9:
        return "Mid Cap"
    return "Small Cap"


# --- Session State Initialization ---

if "holdings" not in st.session_state:
    st.session_state.holdings = [
        {"ticker": "AAPL", "weight": 25.0},
        {"ticker": "MSFT", "weight": 25.0},
        {"ticker": "NVDA", "weight": 25.0},
        {"ticker": "GOOGL", "weight": 25.0},
    ]
if "cash_weight" not in st.session_state:
    st.session_state.cash_weight = 0.0

CASH_INFO = {
    "ticker": "CASH",
    "name": "Cash",
    "price": 1.0,
    "pe": None,
    "forward_pe": None,
    "beta": 0.0,
    "div_yield": None,
    "debt_equity": None,
    "sector": "Cash",
    "country": "United States",
    "market_cap": 0,
}

# --- Title ---

st.markdown(
    f"""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <h1 style="margin:0;">Portfolio Builder</h1>
        <p style="color:gray; font-size:1.1rem;">Enter tickers and target weights to analyze your portfolio</p>
        <p style="font-size:1.3rem; font-weight:600;">Base Portfolio Value: ${PORTFOLIO_VALUE:,.0f}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# --- Portfolio Input ---

st.subheader("Build Your Portfolio")

# Cash allocation
cash_cols = st.columns([2, 3, 1])
with cash_cols[0]:
    st.markdown("**CASH**")
with cash_cols[1]:
    cash_w = st.slider(
        "Cash Weight",
        min_value=0,
        max_value=100,
        value=int(st.session_state.cash_weight),
        step=1,
        format="%d%%",
        key="cash_weight_slider",
        label_visibility="collapsed",
    )
    st.session_state.cash_weight = float(cash_w)

for i, holding in enumerate(st.session_state.holdings):
    cols = st.columns([2, 3, 1])
    with cols[0]:
        ticker = st.text_input(
            "Ticker",
            value=holding["ticker"],
            key=f"ticker_{i}",
            label_visibility="collapsed",
            placeholder="e.g. AAPL",
        )
        st.session_state.holdings[i]["ticker"] = ticker.upper().strip()
    with cols[1]:
        weight = st.slider(
            f"Weight {i}",
            min_value=0,
            max_value=100,
            value=int(holding["weight"]),
            step=1,
            format="%d%%",
            key=f"weight_{i}",
            label_visibility="collapsed",
        )
        st.session_state.holdings[i]["weight"] = float(weight)
    with cols[2]:
        if st.button("Remove", key=f"remove_{i}", use_container_width=True):
            st.session_state.holdings.pop(i)
            st.rerun()

col_add, col_spacer = st.columns([1, 4])
with col_add:
    if st.button("+ Add Holding", use_container_width=True):
        st.session_state.holdings.append({"ticker": "", "weight": 0.0})
        st.rerun()

# Weight validation
total_weight = sum(h["weight"] for h in st.session_state.holdings) + st.session_state.cash_weight
valid_tickers = [h for h in st.session_state.holdings if h["ticker"]]

if abs(total_weight - 100.0) < 0.01:
    st.success(f"Total allocation: {total_weight:.0f}%")
    weights_valid = True
else:
    st.warning(f"Total allocation: {total_weight:.0f}% — must equal 100%")
    weights_valid = False

st.divider()

# --- Fetch Data & Render Dashboard ---

if weights_valid and (valid_tickers or st.session_state.cash_weight > 0):
    # Fetch info for all tickers
    ticker_data = {}
    errors = []
    for h in valid_tickers:
        info = fetch_ticker_info(h["ticker"])
        if info and info["price"]:
            ticker_data[h["ticker"]] = info
        else:
            errors.append(h["ticker"])

    # Add cash if allocated
    if st.session_state.cash_weight > 0:
        ticker_data["CASH"] = CASH_INFO

    for err in errors:
        st.error(f"Could not fetch data for **{err}** — skipping.")

    if ticker_data:
        # Build weights map (only valid tickers)
        weights = {h["ticker"]: h["weight"] for h in valid_tickers if h["ticker"] in ticker_data}
        if st.session_state.cash_weight > 0:
            weights["CASH"] = st.session_state.cash_weight
        w_total = sum(weights.values())
        # List of actual stock tickers (exclude CASH)
        stock_tickers = [s for s in weights if s != "CASH"]

        # --- Weighted Metrics ---
        w_pe = 0
        pe_weight_sum = 0
        w_fpe = 0
        fpe_weight_sum = 0
        w_beta = 0
        beta_weight_sum = 0
        w_div = 0
        w_de = 0
        de_weight_sum = 0
        sectors = set()

        for sym, w in weights.items():
            d = ticker_data[sym]
            norm_w = w / w_total
            if d["pe"]:
                w_pe += d["pe"] * norm_w
                pe_weight_sum += norm_w
            if d["forward_pe"]:
                w_fpe += d["forward_pe"] * norm_w
                fpe_weight_sum += norm_w
            if d["beta"]:
                w_beta += d["beta"] * norm_w
                beta_weight_sum += norm_w
            if d["div_yield"]:
                w_div += d["div_yield"] * norm_w
            if d["debt_equity"] is not None:
                w_de += d["debt_equity"] * norm_w
                de_weight_sum += norm_w
            if d["sector"] != "N/A":
                sectors.add(d["sector"])

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.metric("Weighted P/E", f"{w_pe:.1f}" if pe_weight_sum else "N/A")
        with k2:
            st.metric("Weighted Fwd P/E", f"{w_fpe:.1f}" if fpe_weight_sum else "N/A")
        with k3:
            st.metric("Weighted Beta", f"{w_beta:.2f}" if beta_weight_sum else "N/A")
        with k4:
            st.metric("Weighted Div Yield", f"{w_div:.2f}%" if w_div else "0.00%")
        with k5:
            st.metric("Weighted D/E", f"{w_de:.1f}%" if de_weight_sum else "N/A")
        with k6:
            st.metric("Holdings / Sectors", f"{len(ticker_data)} / {len(sectors)}")

        st.divider()

        # --- Sector & Market Cap Distribution ---
        sec_col, cap_col = st.columns(2)

        # Sector allocation
        sector_weights = {}
        cap_dist = {}
        for sym, w in weights.items():
            d = ticker_data[sym]
            sector_weights[d["sector"]] = sector_weights.get(d["sector"], 0) + w
            if sym != "CASH":
                region = "Domestic" if d["country"] == "United States" else "International"
                label = f"{categorize_market_cap(d['market_cap'])} ({region})"
                cap_dist[label] = cap_dist.get(label, 0) + w

        with sec_col:
            st.subheader("Sector Allocation")
            sec_df = pd.DataFrame(
                {"Sector": list(sector_weights.keys()), "Weight %": list(sector_weights.values())}
            ).sort_values("Weight %", ascending=True)
            fig_sec = px.bar(
                sec_df, x="Weight %", y="Sector", orientation="h",
                color_discrete_sequence=["#636EFA"],
            )
            fig_sec.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_ticksuffix="%",
                yaxis_title="",
                xaxis_title="",
                dragmode=False,
            )
            st.plotly_chart(fig_sec, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

        with cap_col:
            st.subheader("Market Cap Distribution")
            cap_df = pd.DataFrame(
                {"Category": list(cap_dist.keys()), "Weight %": list(cap_dist.values())}
            )
            cap_df = cap_df[cap_df["Weight %"] > 0].sort_values("Weight %", ascending=True)
            fig_cap = px.bar(
                cap_df, x="Weight %", y="Category", orientation="h",
                color_discrete_sequence=["#EF553B"],
            )
            fig_cap.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_ticksuffix="%",
                yaxis_title="",
                xaxis_title="",
                dragmode=False,
            )
            st.plotly_chart(fig_cap, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

        # --- Charts ---
        chart1, chart2 = st.columns(2)

        # Fetch price data (reused for top 3 chart, portfolio perf, and correlation)
        price_data_available = False
        if stock_tickers:
            try:
                close_df = fetch_prices(stock_tickers)
                dates = close_df.index
                price_data_available = True
            except Exception as e:
                st.error(f"Could not fetch price history: {e}")

        with chart1:
            st.subheader("Asset Allocation")
            alloc_df = pd.DataFrame(
                {"Symbol": list(weights.keys()), "Weight": list(weights.values())}
            )
            fig_donut = px.pie(
                alloc_df, values="Weight", names="Symbol", hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_donut.update_traces(textinfo="label+percent", textposition="outside")
            fig_donut.update_layout(showlegend=False, margin=dict(t=60, b=60, l=60, r=60), height=450)
            st.plotly_chart(fig_donut, use_container_width=True)

        with chart2:
            st.subheader("Price Performance (Top 3 Holdings)")
            if price_data_available:
                sorted_syms = sorted([s for s in weights if s != "CASH"], key=lambda s: weights[s], reverse=True)
                top3 = sorted_syms[:3]

                lines = []
                for sym in top3:
                    if sym in close_df.columns:
                        prices = close_df[sym].values
                        change_pct = (prices / prices[0] - 1) * 100
                        lines.append(
                            pd.DataFrame({"Date": dates, "Change %": change_pct, "Symbol": sym})
                        )

                if lines:
                    perf_df = pd.concat(lines)
                    fig_line = px.line(
                        perf_df, x="Date", y="Change %", color="Symbol",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_line.update_layout(
                        yaxis_ticksuffix="%",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        margin=dict(t=40, b=20, l=20, r=20),
                        hovermode="x unified",
                        dragmode=False,
                    )
                    st.plotly_chart(fig_line, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
                else:
                    st.info("No price data available for top holdings.")

        # --- Portfolio Performance (full width) ---
        if price_data_available or st.session_state.cash_weight > 0:
            st.subheader("Portfolio Performance")
            period = st.radio(
                "Time Period",
                ["1 Month", "3 Months", "6 Months", "1 Year"],
                index=1,
                horizontal=True,
                label_visibility="collapsed",
            )
            period_days = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}[period]
            perf_start = (datetime.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")

            try:
                if stock_tickers:
                    perf_close = fetch_prices(stock_tickers, start=perf_start)
                    perf_returns = perf_close.pct_change().dropna()
                else:
                    # Cash-only portfolio: create a dummy returns series
                    import pandas as pd
                    date_range = pd.bdate_range(start=perf_start, end=datetime.today())
                    perf_returns = pd.DataFrame(index=date_range)
                    perf_close = perf_returns

                # Compute weighted portfolio daily return (cash contributes 0)
                port_daily = pd.Series(0.0, index=perf_returns.index)
                for sym, w in weights.items():
                    if sym != "CASH" and sym in perf_returns.columns:
                        port_daily += (w / w_total) * perf_returns[sym]

                # Convert to dollar value starting at $100k
                cum_return = (1 + port_daily).cumprod()
                port_value = PORTFOLIO_VALUE * cum_return

                # Prepend the starting value
                if len(perf_close) > 0:
                    start_row = pd.Series([PORTFOLIO_VALUE], index=[perf_close.index[0]])
                else:
                    start_row = pd.Series([PORTFOLIO_VALUE], index=[pd.Timestamp(perf_start)])
                port_value = pd.concat([start_row, port_value])

                port_perf_df = pd.DataFrame({"Date": port_value.index, "Portfolio Value": port_value.values})
                fig_port = px.line(
                    port_perf_df, x="Date", y="Portfolio Value",
                    color_discrete_sequence=["#636EFA"],
                )
                fig_port.update_layout(
                    yaxis_tickprefix="$",
                    yaxis_tickformat=",",
                    margin=dict(t=20, b=20, l=20, r=20),
                    hovermode="x unified",
                    dragmode=False,
                )
                fig_port.update_traces(
                    hovertemplate="$%{y:,.0f}<extra></extra>"
                )
                st.plotly_chart(fig_port, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

                # Show current portfolio value
                current_value = port_value.iloc[-1]
                gain = current_value - PORTFOLIO_VALUE
                gain_pct = (gain / PORTFOLIO_VALUE) * 100

                # Sharpe Ratio (annualized, assuming 252 trading days, risk-free rate ~4.5%)
                risk_free_daily = 0.045 / 252
                excess_returns = port_daily - risk_free_daily
                sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

                v1, v2, v3, v4 = st.columns(4)
                with v1:
                    st.metric("Current Value", f"${current_value:,.0f}")
                with v2:
                    st.metric("Gain/Loss", f"${gain:+,.0f}")
                with v3:
                    st.metric("Return", f"{gain_pct:+.2f}%")
                with v4:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            except Exception as e:
                st.error(f"Could not load performance data: {e}")

        # --- Holdings Table ---
        st.subheader("Holdings")
        rows = []
        for sym, w in weights.items():
            d = ticker_data[sym]
            rows.append({
                "Ticker": sym,
                "Name": d["name"],
                "Weight %": w,
                "Cost Basis": w / 100 * PORTFOLIO_VALUE,
                "Price": d["price"],
                "P/E": d["pe"] if d["pe"] else None,
                "Beta": d["beta"] if d["beta"] else None,
                "Div Yield %": round(d["div_yield"], 2) if d["div_yield"] else None,
                "D/E %": round(d["debt_equity"], 1) if d["debt_equity"] is not None else None,
                "Sector": d["sector"],
                "Market Cap": categorize_market_cap(d["market_cap"]),
            })

        table_df = pd.DataFrame(rows).sort_values("Weight %", ascending=False)
        st.dataframe(
            table_df.style.format({
                "Weight %": "{:.0f}%",
                "Cost Basis": "${:,.0f}",
                "Price": "${:,.2f}",
                "P/E": "{:.1f}",
                "Beta": "{:.2f}",
                "Div Yield %": "{:.2f}%",
                "D/E %": "{:.1f}%",
            }, na_rep="N/A"),
            use_container_width=True,
            hide_index=True,
        )

        # --- Correlation Matrix (1 year of data) ---
        if len(weights) > 1:
            st.subheader("Correlation Matrix")
            try:
                corr_start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

                if stock_tickers:
                    corr_close = fetch_prices(stock_tickers, start=corr_start)
                    corr_matrix = corr_close.pct_change().dropna().corr()
                else:
                    corr_matrix = pd.DataFrame()

                # Clamp negative correlations to 0
                corr_matrix = corr_matrix.clip(lower=0)

                # Add CASH row/column (0 correlation with everything, 1 with itself)
                if "CASH" in weights:
                    all_syms = list(corr_matrix.columns) + ["CASH"]
                    new_matrix = pd.DataFrame(0.0, index=all_syms, columns=all_syms)
                    # Fill in stock correlations
                    for s1 in corr_matrix.columns:
                        for s2 in corr_matrix.columns:
                            new_matrix.loc[s1, s2] = corr_matrix.loc[s1, s2]
                    # Cash diagonal = 1
                    new_matrix.loc["CASH", "CASH"] = 1.0
                    corr_matrix = new_matrix

                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
                    zmin=0, zmax=1,
                    aspect="equal",
                )
                fig_corr.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    dragmode=False,
                )
                st.plotly_chart(fig_corr, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
            except Exception as e:
                st.error(f"Could not load correlation data: {e}")

        # --- Portfolio Rebalancer (hidden) ---
        with st.expander("Portfolio Rebalancer", expanded=False):
            st.caption("Adjust target weights to see how a rebalanced portfolio would have performed.")

            try:
                symbols = stock_tickers
                if not symbols:
                    st.info("Add stock holdings to use the rebalancer.")
                    raise ValueError("No stocks")
                close_df_reb = fetch_prices(symbols)
                dates_reb = close_df_reb.index
                all_prices = {sym: close_df_reb[sym].values for sym in symbols if sym in close_df_reb.columns}

                actual_weights = {sym: weights[sym] for sym in symbols}

                slider_cols = st.columns(len(symbols))
                rebalanced_weights = {}
                for i, sym in enumerate(symbols):
                    with slider_cols[i]:
                        rebalanced_weights[sym] = st.slider(
                            sym, 0, 100, int(round(actual_weights.get(sym, 0))),
                            format="%d%%", key=f"reb_w_{sym}"
                        )

                reb_weight_sum = sum(rebalanced_weights.values())
                if reb_weight_sum == 100:
                    st.success(f"Total allocation: {reb_weight_sum}%")
                else:
                    st.warning(f"Total allocation: {reb_weight_sum}% — should be 100%")

                daily_returns_per_symbol = {}
                for sym, prices in all_prices.items():
                    daily_returns_per_symbol[sym] = np.diff(prices) / prices[:-1]

                def portfolio_cumulative_return(weights_dict):
                    weighted_daily = np.zeros(len(dates_reb) - 1)
                    total_w = sum(weights_dict.values())
                    if total_w == 0:
                        return np.zeros(len(dates_reb))
                    for sym in symbols:
                        if sym in daily_returns_per_symbol:
                            w = weights_dict.get(sym, 0) / total_w
                            weighted_daily += w * daily_returns_per_symbol[sym]
                    cum = np.concatenate([[0], np.cumsum(np.log1p(weighted_daily))])
                    return (np.exp(cum) - 1) * 100

                original_cum = portfolio_cumulative_return(actual_weights)
                rebalanced_cum = portfolio_cumulative_return(rebalanced_weights)

                chart_df = pd.DataFrame({
                    "Date": np.tile(dates_reb, 2),
                    "Cumulative Return %": np.concatenate([original_cum, rebalanced_cum]),
                    "Portfolio": ["Original"] * len(dates_reb) + ["Rebalanced"] * len(dates_reb),
                })

                fig_rebal = px.line(
                    chart_df, x="Date", y="Cumulative Return %", color="Portfolio",
                    color_discrete_map={"Original": "#636EFA", "Rebalanced": "#EF553B"},
                )
                fig_rebal.update_layout(
                    yaxis_ticksuffix="%",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(t=40, b=20, l=20, r=20),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_rebal, use_container_width=True)

                def compute_metrics(cum_return_pct):
                    final_return = cum_return_pct[-1]
                    cum_factor = 1 + cum_return_pct / 100
                    running_max = np.maximum.accumulate(cum_factor)
                    drawdowns = (cum_factor - running_max) / running_max * 100
                    max_dd = drawdowns.min()
                    return final_return, max_dd

                orig_ret, orig_dd = compute_metrics(original_cum)
                reb_ret, reb_dd = compute_metrics(rebalanced_cum)

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Total Return (Original)", f"{orig_ret:+.2f}%")
                    st.metric("Total Return (Rebalanced)", f"{reb_ret:+.2f}%", delta=f"{reb_ret - orig_ret:+.2f}%")
                with m2:
                    st.metric("Max Drawdown (Original)", f"{orig_dd:.2f}%")
                    st.metric("Max Drawdown (Rebalanced)", f"{reb_dd:.2f}%", delta=f"{reb_dd - orig_dd:+.2f}%", delta_color="inverse")

            except Exception as e:
                st.error(f"Rebalancer error: {e}")

else:
    st.info("Add tickers and/or cash, and set weights that sum to 100% to see your portfolio analysis.")
