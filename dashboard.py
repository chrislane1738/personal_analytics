import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

# --- Mock Price Performance Line Chart ---
with col2:
    st.subheader("Price Performance (Top 3 Holdings)")
    top3 = df.nlargest(3, "MarketValue")

    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=90)

    lines = []
    for _, row in top3.iterrows():
        end_price = row["Price"]
        # walk backward from current price with realistic daily returns
        daily_returns = np.random.normal(0.0008, 0.015, len(dates))
        prices = [end_price]
        for r in reversed(daily_returns[1:]):
            prices.insert(0, prices[0] / (1 + r))
        lines.append(
            pd.DataFrame({"Date": dates, "Price": prices, "Symbol": row["Symbol"]})
        )

    perf_df = pd.concat(lines)

    # Normalize to percentage change from day 1
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
