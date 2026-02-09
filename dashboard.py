import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# Load data
df = pd.read_csv("mock_portfolio.csv")

# --- KPI ---
total_value = df["MarketValue"].sum()
total_cost = df["CostBasis"].sum()
total_gain = total_value - total_cost
gain_pct = (total_gain / total_cost) * 100

st.markdown(
    f"""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
        <h4 style="margin:0; color:gray;">Total Portfolio Value</h4>
        <h1 style="margin:0; font-size:3.5rem;">${total_value:,.2f}</h1>
        <p style="font-size:1.2rem; color:{'green' if total_gain >= 0 else 'red'};">
            {"+" if total_gain >= 0 else ""}{total_gain:,.2f}
            ({gain_pct:+.1f}%)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# --- Layout: donut chart left, line chart right ---
col1, col2 = st.columns(2)

# --- Donut Chart ---
with col1:
    st.subheader("Asset Allocation")
    fig_donut = px.pie(
        df,
        values="MarketValue",
        names="Symbol",
        hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_donut.update_traces(textinfo="label+percent", textposition="outside")
    fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_donut, use_container_width=True)

# --- Fetch real price data for all holdings ---
@st.cache_data(ttl=3600)
def fetch_prices(tickers):
    data = yf.download(tickers, start="2025-10-01", progress=False)
    return data["Close"]

symbols = df["Symbol"].tolist()
close_df = fetch_prices(symbols)
dates = close_df.index
all_prices = {sym: close_df[sym].values for sym in symbols}

# --- Price Performance Line Chart (Top 3 Holdings) ---
with col2:
    st.subheader("Price Performance (Top 3 Holdings)")
    top3 = df.nlargest(3, "MarketValue")

    lines = []
    for _, row in top3.iterrows():
        sym = row["Symbol"]
        lines.append(
            pd.DataFrame({"Date": dates, "Price": all_prices[sym], "Symbol": sym})
        )

    perf_df = pd.concat(lines)

    def normalize(group):
        group = group.copy()
        group["Change %"] = (group["Price"] / group["Price"].iloc[0] - 1) * 100
        return group

    perf_df = perf_df.groupby("Symbol", group_keys=False).apply(normalize)

    fig_line = px.line(
        perf_df,
        x="Date",
        y="Change %",
        color="Symbol",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_line.update_layout(
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=40, b=20, l=20, r=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True)

# --- Holdings Table ---
st.subheader("Holdings")
display_df = df.copy()
display_df["Gain/Loss"] = display_df["MarketValue"] - display_df["CostBasis"]
display_df["Return %"] = (display_df["Gain/Loss"] / display_df["CostBasis"]) * 100
st.dataframe(
    display_df.style.format(
        {
            "Price": "${:,.2f}",
            "MarketValue": "${:,.2f}",
            "CostBasis": "${:,.2f}",
            "Gain/Loss": "${:+,.2f}",
            "Return %": "{:+.1f}%",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# --- Portfolio Rebalancer ---
st.divider()
st.subheader("Portfolio Rebalancer")
st.caption("Adjust target weights to see how a rebalanced portfolio would have performed since October 2025.")

# Compute actual weights
actual_weights = {}
for _, row in df.iterrows():
    actual_weights[row["Symbol"]] = round(row["MarketValue"] / total_value * 100, 1)

# Sliders
slider_cols = st.columns(len(symbols))
rebalanced_weights = {}
for i, sym in enumerate(symbols):
    with slider_cols[i]:
        rebalanced_weights[sym] = st.slider(
            sym, 0, 100, int(round(actual_weights[sym])), format="%d%%", key=f"w_{sym}"
        )

weight_sum = sum(rebalanced_weights.values())
if weight_sum == 100:
    st.success(f"Total allocation: {weight_sum}%")
else:
    st.warning(f"Total allocation: {weight_sum}% — should be 100%")

# Compute daily returns per holding
daily_returns_per_symbol = {}
for sym, prices in all_prices.items():
    daily_returns_per_symbol[sym] = np.diff(prices) / prices[:-1]

# Build portfolio return series for a given weight dict
def portfolio_cumulative_return(weights_dict):
    weighted_daily = np.zeros(len(dates) - 1)
    total_w = sum(weights_dict.values())
    if total_w == 0:
        return np.zeros(len(dates))
    for sym in symbols:
        w = weights_dict[sym] / total_w
        weighted_daily += w * daily_returns_per_symbol[sym]
    cum = np.concatenate([[0], np.cumsum(np.log1p(weighted_daily))])
    return (np.exp(cum) - 1) * 100  # percent

original_cum = portfolio_cumulative_return(actual_weights)
rebalanced_cum = portfolio_cumulative_return(rebalanced_weights)

# Comparison line chart
chart_df = pd.DataFrame({
    "Date": np.tile(dates, 2),
    "Cumulative Return %": np.concatenate([original_cum, rebalanced_cum]),
    "Portfolio": ["Original"] * len(dates) + ["Rebalanced"] * len(dates),
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

# Summary metrics
def compute_metrics(cum_return_pct):
    final_return = cum_return_pct[-1]
    final_value = total_value * (1 + final_return / 100)
    # Max drawdown from cumulative return curve
    cum_factor = 1 + cum_return_pct / 100
    running_max = np.maximum.accumulate(cum_factor)
    drawdowns = (cum_factor - running_max) / running_max * 100
    max_dd = drawdowns.min()
    return final_value, final_return, max_dd

orig_val, orig_ret, orig_dd = compute_metrics(original_cum)
reb_val, reb_ret, reb_dd = compute_metrics(rebalanced_cum)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Final Value (Original)", f"${orig_val:,.2f}")
    st.metric("Final Value (Rebalanced)", f"${reb_val:,.2f}", delta=f"${reb_val - orig_val:+,.2f}")
with m2:
    st.metric("Total Return (Original)", f"{orig_ret:+.2f}%")
    st.metric("Total Return (Rebalanced)", f"{reb_ret:+.2f}%", delta=f"{reb_ret - orig_ret:+.2f}%")
with m3:
    st.metric("Max Drawdown (Original)", f"{orig_dd:.2f}%")
    st.metric("Max Drawdown (Rebalanced)", f"{reb_dd:.2f}%", delta=f"{reb_dd - orig_dd:+.2f}%", delta_color="inverse")
