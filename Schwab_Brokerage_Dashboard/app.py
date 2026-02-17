import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Lane Enterprises Portfolio Terminal", layout="wide")

# ── Bloomberg Terminal CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0A0E17;
        color: #FF8C00;
    }

    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #0D1117 !important;
        color: #FF8C00 !important;
    }

    /* All text */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label,
    .stApp div, [data-testid="stMarkdownContainer"] {
        color: #FF8C00 !important;
    }

    /* Title styling */
    .stApp h1 {
        color: #FF8C00 !important;
        font-family: 'Consolas', 'Courier New', monospace !important;
        border-bottom: 2px solid #FF8C00;
        padding-bottom: 10px;
    }

    /* Subheaders */
    .stApp h2, .stApp h3 {
        color: #FFA500 !important;
        font-family: 'Consolas', 'Courier New', monospace !important;
        border-bottom: 1px solid #333D4D;
        padding-bottom: 6px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #111921;
        border: 1px solid #1E2D3D;
        border-radius: 4px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        color: #FF8C00 !important;
        font-family: 'Consolas', monospace !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8899AA !important;
        font-family: 'Consolas', monospace !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'Consolas', monospace !important;
    }

    /* Dataframes / tables */
    [data-testid="stDataFrame"], .stDataFrame {
        border: 1px solid #1E2D3D;
    }
    [data-testid="stDataFrame"] th {
        background-color: #111921 !important;
        color: #8899AA !important;
        font-family: 'Consolas', monospace !important;
    }
    [data-testid="stDataFrame"] td {
        background-color: #0A0E17 !important;
        color: #FF8C00 !important;
        font-family: 'Consolas', monospace !important;
    }

    /* Plotly charts */
    .js-plotly-plot .plotly .bg {
        fill: #0A0E17 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #111921;
        border-radius: 4px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0A0E17;
        color: #8899AA !important;
        border: 1px solid #1E2D3D;
        border-radius: 4px;
        font-family: 'Consolas', monospace !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E2D3D !important;
        color: #FF8C00 !important;
        border-color: #FF8C00 !important;
    }

    /* Select boxes and inputs */
    [data-testid="stSelectbox"], .stSelectbox,
    [data-testid="stDateInput"], .stDateInput,
    .stTextInput input {
        font-family: 'Consolas', monospace !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #1E2D3D;
        background-color: #111921;
    }

    /* Container / card borders — match expander style */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1a1c24 !important;
        border-color: rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #1a1c24 !important;
        border-color: rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div[style] {
        background-color: #1a1c24 !important;
        border-color: rgba(255,255,255,0.15) !important;
    }

    /* Horizontal rule */
    hr {
        border-color: #1E2D3D;
    }

    /* Links */
    a {
        color: #FF8C00 !important;
        text-decoration: none !important;
    }
    a:hover {
        color: #FFC04D !important;
        text-decoration: underline !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #FF8C00 !important;
    }

    /* Negative / positive styling */
    .negative { color: #FF4B4B !important; }
    .positive { color: #00C853 !important; }
</style>
""", unsafe_allow_html=True)


# ── Password Gate ─────────────────────────────────────────────────────────────
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("LANE ENTERPRISES, INC. PORTFOLIO TERMINAL")
    pwd = st.text_input("Enter dashboard password", type="password")
    if pwd:
        if pwd == DASHBOARD_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


# ── Schwab Client ─────────────────────────────────────────────────────────────
from schwab_auth import authenticate  # noqa: E402


@st.cache_resource(show_spinner="Connecting to Schwab...")
def get_schwab_client():
    return authenticate()


try:
    client = get_schwab_client()
except Exception as e:
    st.error(f"Schwab authentication failed: {e}")
    st.info("Delete token.json and re-run to start a fresh login flow.")
    st.stop()


# ── Fetch Account Data ────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner="Fetching account data...")
def fetch_account_data():
    resp = client.get_accounts(fields=[client.Account.Fields.POSITIONS])
    resp.raise_for_status()
    accounts = resp.json()

    all_positions = []
    total_cash = 0.0
    total_day_change = 0.0
    total_market_value = 0.0

    # Get account hashes from the dedicated endpoint
    acct_num_resp = client.get_account_numbers()
    acct_num_resp.raise_for_status()
    account_numbers = [a["hashValue"] for a in acct_num_resp.json()]

    for acct in accounts:
        info = acct["securitiesAccount"]
        balances = info.get("currentBalances", info.get("initialBalances", {}))
        total_cash += balances.get("cashBalance", 0)

        for pos in info.get("positions", []):
            symbol = pos.get("instrument", {}).get("symbol", "N/A")
            qty = pos.get("longQuantity", 0) - pos.get("shortQuantity", 0)
            market_value = pos.get("marketValue", 0)
            day_change = pos.get("currentDayProfitLoss", 0)
            day_change_pct = pos.get("currentDayProfitLossPercentage", 0)
            avg_price = pos.get("averagePrice", 0)
            total_day_change += day_change
            total_market_value += market_value
            all_positions.append({
                "symbol": symbol,
                "quantity": qty,
                "market_value": market_value,
                "day_change": day_change,
                "day_change_pct": day_change_pct,
                "avg_price": avg_price,
            })

    total_account_value = total_market_value + total_cash
    prev_total = total_account_value - total_day_change
    total_day_change_pct = (total_day_change / prev_total * 100) if prev_total else 0

    return total_account_value, total_cash, total_day_change, total_day_change_pct, all_positions, account_numbers


try:
    total_value, cash_balance, day_change, day_change_pct, positions, account_numbers = fetch_account_data()
except Exception as e:
    st.error(f"Failed to fetch account data: {e}")
    st.info("Your Schwab token may be expired. Delete token.json and re-run.")
    st.stop()


# ── Cached yfinance .info lookup (shared by fundamentals + market cap) ────────
@st.cache_data(ttl=3600, show_spinner=False)
def _yf_info_cache(sym):
    """Fetch yf.Ticker(sym).info once and cache for reuse."""
    try:
        return yf.Ticker(sym).info or {}
    except Exception:
        return {}


# ── Fetch Fundamental Data ────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner="Fetching fundamentals...")
def fetch_fundamentals(symbols):
    if not symbols:
        return {}
    fundamentals = {}

    # Try Schwab API first
    schwab_data = {}
    try:
        resp = client.get_quotes(symbols, fields=[client.Quote.Fields.FUNDAMENTAL])
        resp.raise_for_status()
        schwab_data = resp.json()
    except Exception:
        pass

    for sym in symbols:
        fund = schwab_data.get(sym, {}).get("fundamental", {})
        fundamentals[sym] = {
            "pe_ratio": fund.get("peRatio"),
            "forward_pe": fund.get("forwardPeRatio"),
            "ev_ebitda": fund.get("evToEbitda"),
            "price_sales": fund.get("priceToSales"),
            "price_fcf": fund.get("priceToFreeCashFlow") or fund.get("priceToCashFlow"),
            "debt_equity": fund.get("debtToEquity"),
        }

    # Fill gaps with yfinance (uses shared cache)
    missing_syms = [s for s in symbols if any(
        v is None for v in fundamentals[s].values()
    )]
    if missing_syms:
        for sym in missing_syms:
            try:
                info = _yf_info_cache(sym)
                f = fundamentals[sym]
                if f["pe_ratio"] is None:
                    f["pe_ratio"] = info.get("trailingPE")
                if f["forward_pe"] is None:
                    f["forward_pe"] = info.get("forwardPE")
                if f["ev_ebitda"] is None:
                    ev = info.get("enterpriseValue")
                    ebitda = info.get("ebitda")
                    if ev and ebitda and ebitda > 0:
                        f["ev_ebitda"] = ev / ebitda
                if f["price_sales"] is None:
                    f["price_sales"] = info.get("priceToSalesTrailing12Months")
                if f["price_fcf"] is None:
                    mcap = info.get("marketCap")
                    fcf = info.get("freeCashflow")
                    if mcap and fcf and fcf > 0:
                        f["price_fcf"] = mcap / fcf
                if f["debt_equity"] is None:
                    f["debt_equity"] = info.get("debtToEquity")
            except Exception:
                pass

    return fundamentals


# ── Fetch Transaction History ─────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner="Fetching transactions...")
def fetch_transactions(acct_hash, start_date, end_date):
    all_txns = []
    # Schwab API limits start_date to within 60 days of current date,
    # so we paginate in 60-day windows to cover the full range.
    window_end = end_date
    while window_end > start_date:
        window_start = max(start_date, window_end - datetime.timedelta(days=59))
        try:
            resp = client.get_transactions(
                acct_hash,
                start_date=datetime.datetime.combine(window_start, datetime.time.min),
                end_date=datetime.datetime.combine(window_end, datetime.time.max),
                transaction_types=[client.Transactions.TransactionType.TRADE],
            )
            resp.raise_for_status()
            all_txns.extend(resp.json())
        except Exception:
            pass
        window_end = window_start - datetime.timedelta(days=1)
    return all_txns


# ── Fetch Historical Prices for Correlation ───────────────────────────────────
def _download_single_close(sym, period=None, start=None, end=None):
    """Download daily close prices for a single ticker with retry logic."""
    import time
    kwargs = dict(auto_adjust=True, progress=False)
    if period:
        kwargs["period"] = period
    if start:
        kwargs["start"] = start
    if end:
        kwargs["end"] = end
    for attempt in range(3):
        try:
            data = yf.download(sym, **kwargs)
            if data.empty:
                if attempt < 2:
                    time.sleep(0.5)
                    continue
                return None
            if isinstance(data.columns, pd.MultiIndex):
                col = data["Close"]
            else:
                col = data[["Close"]]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            return col.rename(sym)
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
            continue
    return None


@st.cache_data(ttl=3600, show_spinner="Fetching price history...")
def fetch_price_history(symbols, period="1y"):
    """Download daily closing prices for each symbol individually to avoid
    yfinance batch-download failures, then merge into a single DataFrame."""
    if not symbols:
        return pd.DataFrame()
    frames = []
    for sym in symbols:
        col = _download_single_close(sym, period=period)
        if col is not None:
            frames.append(col)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


# ── Compute Sharpe Ratio ──────────────────────────────────────────────────────
def compute_sharpe_ratio(prices_df, weights, risk_free_rate=0.05):
    if prices_df.empty or len(weights) == 0:
        return None
    returns = prices_df.pct_change().dropna()
    if returns.empty or len(returns) < 10:
        return None
    aligned_symbols = [s for s in weights if s in returns.columns]
    if not aligned_symbols:
        return None
    w = np.array([weights[s] for s in aligned_symbols])
    w = w / w.sum()
    portfolio_returns = returns[aligned_symbols].dot(w)
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    if annual_vol == 0:
        return None
    return (annual_return - risk_free_rate) / annual_vol


# ── Fetch Market Cap & Region Data ────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching market cap data...")
def fetch_market_cap_data(symbols):
    """Returns list of dicts with symbol, market_cap, cap_category, region."""
    results = []
    for sym in symbols:
        try:
            info = _yf_info_cache(sym)
            mcap = info.get("marketCap", 0) or 0
            country = info.get("country", "United States") or "United States"
            # Classify cap size
            if mcap >= 200e9:
                cat = "Mega Cap"
            elif mcap >= 10e9:
                cat = "Large Cap"
            elif mcap >= 2e9:
                cat = "Mid Cap"
            elif mcap >= 300e6:
                cat = "Small Cap"
            elif mcap > 0:
                cat = "Micro Cap"
            else:
                cat = "Unknown"
            # Classify region
            us_territories = {"United States", "Puerto Rico", "Guam", "U.S. Virgin Islands"}
            region = "Domestic" if country in us_territories else "International"
            results.append({
                "symbol": sym, "market_cap": mcap,
                "cap_category": cat, "region": region, "country": country,
            })
        except Exception:
            results.append({
                "symbol": sym, "market_cap": 0,
                "cap_category": "Unknown", "region": "Unknown", "country": "Unknown",
            })
    return results


# ── Monte Carlo Simulation Engine ─────────────────────────────────────────────
def run_monte_carlo(prices_df, weights_dict, starting_value, n_sims=1000,
                    n_days=252):
    """Correlated GBM Monte Carlo using Cholesky decomposition."""
    returns = prices_df.pct_change().dropna()
    aligned = [s for s in weights_dict if s in returns.columns]
    if not aligned or returns.empty:
        return None, None
    ret = returns[aligned]
    w = np.array([weights_dict[s] for s in aligned])
    w = w / w.sum()

    mean_returns = ret.mean().values
    cov_matrix = ret.cov().values

    # Portfolio-level annualized drift
    daily_drift = float(mean_returns @ w)
    annual_drift = daily_drift * 252

    # Cholesky decomposition for correlated random draws
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback: use portfolio-level stats if cov isn't positive definite
        port_ret = ret.dot(w)
        mu = port_ret.mean()
        sigma = port_ret.std()
        Z = np.random.standard_normal((n_sims, n_days))
        daily_returns = mu + sigma * Z
        paths = starting_value * np.cumprod(1 + daily_returns, axis=1)
        paths = np.column_stack([np.full(n_sims, starting_value), paths])
        return paths, None

    all_paths = np.zeros((n_sims, n_days + 1))
    all_paths[:, 0] = starting_value

    for i in range(n_sims):
        Z = np.random.standard_normal((n_days, len(aligned)))
        correlated_Z = Z @ L.T
        daily_asset_returns = mean_returns + correlated_Z
        daily_portfolio_returns = daily_asset_returns @ w
        price_path = starting_value * np.cumprod(1 + daily_portfolio_returns)
        all_paths[i, 1:] = price_path

    # Compute stats from all simulations
    final_values = all_paths[:, -1]
    total_returns = (final_values / starting_value - 1) * 100
    win_threshold = starting_value * 1.10  # 10% gain
    stats = {
        "median_return": float(np.median(total_returns)),
        "p5": float(np.percentile(total_returns, 5)),
        "p95": float(np.percentile(total_returns, 95)),
        "p_loss": float(np.mean(total_returns < 0) * 100),
        "var_95": float(np.percentile(total_returns, 5)),  # VaR at 95% confidence
        "median_final": float(np.median(final_values)),
        "p5_final": float(np.percentile(final_values, 5)),
        "p95_final": float(np.percentile(final_values, 95)),
        "annual_drift": annual_drift * 100,  # as percentage
        "daily_drift": daily_drift * 100,
        "win_rate": float(np.mean(final_values >= win_threshold) * 100),
    }

    # Max drawdown (median path)
    median_idx = np.argsort(final_values)[len(final_values) // 2]
    median_path = all_paths[median_idx]
    running_max = np.maximum.accumulate(median_path)
    drawdowns = (median_path - running_max) / running_max * 100
    stats["max_drawdown"] = float(np.min(drawdowns))

    return all_paths, stats


# ── Historic Stress Test Engine ───────────────────────────────────────────────
STRESS_EVENTS = {
    "COVID Crash (2020)": ("2020-02-19", "2020-03-23"),
    "2022 Bear Market": ("2022-01-03", "2022-10-12"),
    "2008 Financial Crisis": ("2008-09-01", "2009-03-09"),
    "Dot-com Bust (2000-02)": ("2000-03-10", "2002-10-09"),
    "Black Monday (1987)": ("1987-10-01", "1987-12-31"),
}

# ── Hypothetical Future Scenario Definitions ─────────────────────────────────
HYPOTHETICAL_SCENARIOS = {
    "Taiwan Strait Crisis": {
        "description": "Military conflict over Taiwan disrupts global semiconductor supply chains. "
                       "Tech and chip-dependent sectors suffer massive losses while defense benefits.",
        "market_drop": -25,
        "sector_impacts": {
            "Technology": -40, "Communication Services": -25, "Consumer Discretionary": -20,
            "Industrials": -15, "Financials": -18, "Health Care": -8,
            "Consumer Staples": -5, "Utilities": -3, "Energy": -10,
            "Materials": -12, "Real Estate": -10,
        },
    },
    "US-China Trade War Escalation": {
        "description": "Broad tariff escalation and export bans between the US and China. "
                       "Trade-sensitive sectors hit hardest; domestic defensives hold up.",
        "market_drop": -20,
        "sector_impacts": {
            "Technology": -30, "Communication Services": -15, "Consumer Discretionary": -22,
            "Industrials": -18, "Financials": -12, "Health Care": -8,
            "Consumer Staples": -5, "Utilities": -3, "Energy": -10,
            "Materials": -20, "Real Estate": -6,
        },
    },
    "Global Supply Chain Rupture": {
        "description": "Major shipping lane disruption (e.g. Strait of Hormuz or Suez blockade). "
                       "Manufacturing and retail hit hard; energy producers benefit from scarcity.",
        "market_drop": -15,
        "sector_impacts": {
            "Technology": -18, "Communication Services": -8, "Consumer Discretionary": -22,
            "Industrials": -25, "Financials": -10, "Health Care": -6,
            "Consumer Staples": -12, "Utilities": -2, "Energy": 20,
            "Materials": -15, "Real Estate": -5,
        },
    },
    "AI Bubble Pop": {
        "description": "AI hype collapses as revenue expectations miss. Tech and AI-adjacent "
                       "sectors crater while defensive sectors are barely touched.",
        "market_drop": -22,
        "sector_impacts": {
            "Technology": -45, "Communication Services": -30, "Consumer Discretionary": -15,
            "Industrials": -8, "Financials": -10, "Health Care": -5,
            "Consumer Staples": -2, "Utilities": 0, "Energy": -3,
            "Materials": -5, "Real Estate": -4,
        },
    },
    "Sovereign Debt Crisis": {
        "description": "A major economy faces a debt spiral triggering contagion fears. "
                       "Financials and rate-sensitive sectors suffer the most.",
        "market_drop": -20,
        "sector_impacts": {
            "Technology": -12, "Communication Services": -10, "Consumer Discretionary": -15,
            "Industrials": -12, "Financials": -35, "Health Care": -5,
            "Consumer Staples": -4, "Utilities": -8, "Energy": -6,
            "Materials": -10, "Real Estate": -25,
        },
    },
    "Oil Supply Shock": {
        "description": "Major oil supply disruption sends crude above $150/bbl. Energy producers "
                       "surge while consumer-facing sectors absorb higher input costs.",
        "market_drop": -10,
        "sector_impacts": {
            "Technology": -8, "Communication Services": -5, "Consumer Discretionary": -18,
            "Industrials": -12, "Financials": -8, "Health Care": -4,
            "Consumer Staples": -10, "Utilities": -6, "Energy": 20,
            "Materials": -5, "Real Estate": -8,
        },
    },
    "Greenland Invasion": {
        "description": "US military seizure of Greenland triggers a severe rupture with NATO "
                       "and the EU. European retaliatory tariffs and sanctions disrupt transatlantic "
                       "trade; defense spending surges while export-heavy and EU-exposed sectors suffer.",
        "market_drop": -18,
        "sector_impacts": {
            "Technology": -20, "Communication Services": -10, "Consumer Discretionary": -22,
            "Industrials": -8, "Financials": -15, "Health Care": -6,
            "Consumer Staples": -14, "Utilities": -4, "Energy": -12,
            "Materials": -18, "Real Estate": -8,
        },
    },
}

# ── Tariff Sector Exposure Data ──────────────────────────────────────────────
# Estimated % of revenue exposed to each trade partner by GICS sector.
# These are reasonable sector-level defaults (per-company data not reliably available).
TARIFF_SECTOR_EXPOSURE = {
    "Technology":               {"China": 0.25, "EU": 0.18, "Mexico": 0.05, "Canada": 0.04, "ROW": 0.15},
    "Communication Services":   {"China": 0.10, "EU": 0.15, "Mexico": 0.04, "Canada": 0.05, "ROW": 0.10},
    "Consumer Discretionary":   {"China": 0.20, "EU": 0.12, "Mexico": 0.08, "Canada": 0.06, "ROW": 0.12},
    "Industrials":              {"China": 0.12, "EU": 0.14, "Mexico": 0.10, "Canada": 0.08, "ROW": 0.10},
    "Financials":               {"China": 0.06, "EU": 0.10, "Mexico": 0.03, "Canada": 0.05, "ROW": 0.06},
    "Health Care":              {"China": 0.08, "EU": 0.16, "Mexico": 0.04, "Canada": 0.06, "ROW": 0.10},
    "Consumer Staples":         {"China": 0.06, "EU": 0.10, "Mexico": 0.06, "Canada": 0.08, "ROW": 0.08},
    "Utilities":                {"China": 0.02, "EU": 0.03, "Mexico": 0.02, "Canada": 0.04, "ROW": 0.02},
    "Energy":                   {"China": 0.10, "EU": 0.12, "Mexico": 0.06, "Canada": 0.10, "ROW": 0.15},
    "Materials":                {"China": 0.15, "EU": 0.10, "Mexico": 0.06, "Canada": 0.08, "ROW": 0.12},
    "Real Estate":              {"China": 0.03, "EU": 0.04, "Mexico": 0.02, "Canada": 0.03, "ROW": 0.03},
}

# Average exposure used for unknown sectors
_AVG_EXPOSURE = {
    country: np.mean([TARIFF_SECTOR_EXPOSURE[s][country] for s in TARIFF_SECTOR_EXPOSURE])
    for country in ["China", "EU", "Mexico", "Canada", "ROW"]
}

# ── Tariff Scenario Definitions ──────────────────────────────────────────────
TARIFF_SCENARIOS = {
    "China 60% Tariff": {
        "description": "Blanket 60% tariff on all Chinese imports. Tech and consumer sectors "
                       "with heavy China supply-chain exposure take the biggest hit.",
        "tariffs": {"China": 60},
        "pass_through": 0.5,
    },
    "EU Auto & Ag Tariff": {
        "description": "25% tariff on EU goods targeting automotive and agricultural sectors. "
                       "Impacts EU-exposed industrials, consumer discretionary, and staples.",
        "tariffs": {"EU": 25},
        "pass_through": 0.5,
    },
    "USMCA Collapse": {
        "description": "Trade agreement collapses — 35% tariff on Mexico, 25% on Canada. "
                       "North American supply chains disrupted across all sectors.",
        "tariffs": {"Mexico": 35, "Canada": 25},
        "pass_through": 0.5,
    },
    "Global Trade War": {
        "description": "Simultaneous tariff escalation with all major partners. "
                       "Broad portfolio impact — no sector is spared.",
        "tariffs": {"China": 40, "EU": 25, "Mexico": 25, "Canada": 20, "ROW": 15},
        "pass_through": 0.5,
    },
    "Trade Deal Rally": {
        "description": "Comprehensive trade deals reduce tariffs across the board. "
                       "Negative tariff rates represent tariff reductions — positive portfolio impact.",
        "tariffs": {"China": -20, "EU": -10, "Mexico": -10, "Canada": -5, "ROW": -5},
        "pass_through": 0.5,
    },
}


def compute_tariff_impact(symbols, weights_dict, starting_value, sector_map,
                          tariffs, pass_through, sector_pass_through=None):
    """Compute per-holding tariff impact.
    impact = Σ(tariff_rate × country_exposure × pass_through) across countries.
    Returns sorted holdings list + portfolio totals."""
    holdings = []
    total_pnl = 0.0
    total_weighted_exposure = 0.0

    for sym in symbols:
        w = weights_dict.get(sym, 0)
        position_value = w * starting_value
        sector = sector_map.get(sym, "Unknown")
        exposure = TARIFF_SECTOR_EXPOSURE.get(sector, _AVG_EXPOSURE)

        impact_pct = 0.0
        country_details = {}
        for country, rate in tariffs.items():
            country_exp = exposure.get(country, 0.0)
            pt = pass_through
            if sector_pass_through and sector in sector_pass_through:
                pt = sector_pass_through[sector]
            eff_impact = -rate / 100 * country_exp * pt  # negative rate = tariff reduction = positive
            impact_pct += eff_impact * 100
            country_details[country] = {
                "exposure": country_exp,
                "rate": rate,
                "impact_pct": eff_impact * 100,
            }

        pnl = position_value * impact_pct / 100
        total_pnl += pnl
        total_weighted_exposure += w * sum(exposure.get(c, 0) for c in tariffs)

        holdings.append({
            "symbol": sym, "sector": sector, "weight": w,
            "impact_pct": impact_pct, "pnl": pnl,
            "position_value": position_value,
            "country_details": country_details,
        })

    holdings.sort(key=lambda h: h["pnl"])
    total_impact_pct = (total_pnl / starting_value * 100) if starting_value else 0
    avg_tariff = np.mean(list(tariffs.values())) if tariffs else 0
    sectors_losing = len({h["sector"] for h in holdings if h["impact_pct"] < -0.1})
    sectors_gaining = len({h["sector"] for h in holdings if h["impact_pct"] > 0.1})

    # Find top exposure country
    country_totals = {}
    for h in holdings:
        for c, det in h["country_details"].items():
            country_totals[c] = country_totals.get(c, 0) + abs(det["impact_pct"]) * h["weight"]
    top_country = max(country_totals, key=country_totals.get) if country_totals else "N/A"

    return {
        "holdings": holdings,
        "total_pnl": total_pnl,
        "total_impact_pct": total_impact_pct,
        "final_value": starting_value + total_pnl,
        "avg_tariff": avg_tariff,
        "weighted_exposure": total_weighted_exposure * 100,
        "top_country": top_country,
        "sectors_losing": sectors_losing,
        "sectors_gaining": sectors_gaining,
    }


def build_tariff_heatmap(holdings, tariffs, sector_map):
    """Build a Plotly heatmap: rows = sectors, columns = countries, cell = effective impact."""
    countries = list(tariffs.keys())
    # Collect unique sectors in portfolio
    sectors_in_portfolio = sorted({h["sector"] for h in holdings})

    z_data = []
    text_data = []
    for sector in sectors_in_portfolio:
        row = []
        text_row = []
        for country in countries:
            # Aggregate impact for this sector × country across holdings
            total_impact = 0.0
            for h in holdings:
                if h["sector"] == sector and country in h["country_details"]:
                    total_impact += h["country_details"][country]["impact_pct"] * h["weight"]
            row.append(total_impact)
            text_row.append(f"{total_impact:+.2f}%")
        z_data.append(row)
        text_data.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=countries,
        y=sectors_in_portfolio,
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=11, family="Consolas, monospace", color="#FFFFFF"),
        colorscale=[
            [0.0, "#B71C1C"],   # deep red (worst)
            [0.35, "#FF4B4B"],  # red
            [0.5, "#1E2D3D"],   # neutral dark
            [0.65, "#00C853"],  # green
            [1.0, "#00E676"],   # bright green (best)
        ],
        zmid=0,
        colorbar=dict(
            title=dict(text="Impact %", font=dict(color="#8899AA")),
            tickfont=dict(color="#8899AA"),
            ticksuffix="%",
        ),
    ))
    fig.update_layout(
        title=dict(
            text="SECTOR × COUNTRY TARIFF IMPACT",
            font=dict(color="#FFA500", size=14, family="Consolas, monospace"),
        ),
        xaxis=dict(tickfont=dict(color="#FF8C00", size=12), side="bottom", fixedrange=True),
        yaxis=dict(tickfont=dict(color="#FF8C00", size=11), autorange="reversed", fixedrange=True),
        height=max(300, len(sectors_in_portfolio) * 45 + 100),
        paper_bgcolor="#0A0E17", plot_bgcolor="#0A0E17",
        font=dict(family="Consolas, monospace", color="#FF8C00"),
        margin=dict(t=50, b=30, l=160, r=30),
    )
    return fig


@st.cache_data(ttl=3600, show_spinner="Fetching stress test data...")
def fetch_stress_test_data(symbols, start_str, end_str):
    """Download daily closing prices for the stress period, one ticker at a
    time to avoid yfinance batch failures."""
    frames = []
    for sym in symbols:
        col = _download_single_close(sym, start=start_str, end=end_str)
        if col is not None:
            frames.append(col)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def compute_stress_test(symbols, weights_dict, starting_value, start_str,
                        end_str):
    """Replay the portfolio through a historic stress period.
    Tickers that didn't exist during the period are proxied with SPY."""
    # Fetch data for portfolio symbols + SPY as fallback
    all_syms = list(set(symbols + ["SPY"]))
    close = fetch_stress_test_data(all_syms, start_str, end_str)
    if close.empty:
        return None

    # Drop columns that are entirely NaN (tickers that didn't exist)
    close = close.dropna(axis=1, how="all")
    if close.empty or "SPY" not in close.columns:
        return None

    # Build aligned weights — use SPY as proxy for missing tickers
    aligned_syms = []
    aligned_weights = []
    proxy_weight = 0.0
    for s in weights_dict:
        if s in close.columns and close[s].notna().sum() > 5:
            aligned_syms.append(s)
            aligned_weights.append(weights_dict[s])
        else:
            proxy_weight += weights_dict[s]

    if proxy_weight > 0:
        if "SPY" in aligned_syms:
            idx = aligned_syms.index("SPY")
            aligned_weights[idx] += proxy_weight
        else:
            aligned_syms.append("SPY")
            aligned_weights.append(proxy_weight)

    if not aligned_syms:
        return None

    w = np.array(aligned_weights)
    w = w / w.sum()

    # Forward-fill gaps, then back-fill for tickers that IPO'd mid-period
    price_data = close[aligned_syms].ffill().bfill().dropna(how="all")
    if len(price_data) < 5:
        return None

    returns = price_data.pct_change().dropna()
    port_returns = returns.dot(w)
    cum_returns = (1 + port_returns).cumprod()
    equity_curve = starting_value * cum_returns
    equity_curve = pd.concat([
        pd.Series([starting_value], index=[price_data.index[0]]),
        equity_curve,
    ])
    total_return = (equity_curve.iloc[-1] / starting_value - 1) * 100
    running_max = equity_curve.cummax()
    max_dd = ((equity_curve - running_max) / running_max * 100).min()
    n_days = len(returns)
    proxied = [s for s in weights_dict if s not in close.columns or close[s].isna().all()]
    return {
        "equity_curve": equity_curve,
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "n_days": n_days,
        "final_value": float(equity_curve.iloc[-1]),
        "proxied_tickers": proxied,
    }


# ── Compute Trailing Betas ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Computing betas...")
def compute_betas(symbols, period="2y"):
    """Compute each ticker's beta relative to SPY using trailing data."""
    all_syms = list(set(symbols + ["SPY"]))
    close = fetch_price_history(all_syms, period=period)
    if close.empty:
        return {}
    close = close.dropna(axis=1, how="all").ffill().dropna()
    if "SPY" not in close.columns:
        return {}
    returns = close.pct_change().dropna()
    spy_ret = returns["SPY"]
    spy_var = spy_ret.var()
    if spy_var == 0:
        return {}
    betas = {}
    for sym in symbols:
        if sym in returns.columns:
            cov = returns[sym].cov(spy_ret)
            betas[sym] = cov / spy_var
        else:
            betas[sym] = 1.0  # default to market beta
    return betas


def compute_beta_stress_test(symbols, weights_dict, betas, starting_value,
                             start_str, end_str):
    """Beta-adjusted stress test: apply each ticker's beta to SPY returns."""
    close = fetch_stress_test_data(["SPY"], start_str, end_str)
    if close.empty:
        return None
    if isinstance(close, pd.DataFrame) and "SPY" in close.columns:
        spy_prices = close["SPY"]
    elif isinstance(close, pd.Series):
        spy_prices = close
    else:
        return None
    spy_prices = spy_prices.ffill().dropna()
    if len(spy_prices) < 5:
        return None
    spy_returns = spy_prices.pct_change().dropna()

    # For each day, portfolio return = sum(weight_i * beta_i * spy_return)
    w_arr = np.array([weights_dict.get(s, 0) for s in symbols])
    b_arr = np.array([betas.get(s, 1.0) for s in symbols])
    w_arr = w_arr / w_arr.sum()
    weighted_beta = float(w_arr @ b_arr)

    port_returns = spy_returns * weighted_beta
    cum_returns = (1 + port_returns).cumprod()
    equity_curve = starting_value * cum_returns
    equity_curve = pd.concat([
        pd.Series([starting_value], index=[spy_prices.index[0]]),
        equity_curve,
    ])
    total_return = (equity_curve.iloc[-1] / starting_value - 1) * 100
    running_max = equity_curve.cummax()
    max_dd = ((equity_curve - running_max) / running_max * 100).min()
    return {
        "equity_curve": equity_curve,
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "n_days": len(spy_returns),
        "final_value": float(equity_curve.iloc[-1]),
        "portfolio_beta": weighted_beta,
    }


# ── Hypothetical Scenario Helpers ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_map(symbols):
    """Map each portfolio symbol to its GICS sector via cached yfinance info."""
    sector_map = {}
    for sym in symbols:
        info = _yf_info_cache(sym)
        sector_map[sym] = info.get("sector", "Unknown")
    return sector_map


def compute_sector_based_impact(symbols, weights_dict, starting_value, sector_map,
                                 sector_impacts, market_drop):
    """Compute per-holding P&L using sector-specific impact percentages."""
    holdings = []
    total_pnl = 0.0
    for sym in symbols:
        w = weights_dict.get(sym, 0)
        position_value = w * starting_value
        sector = sector_map.get(sym, "Unknown")
        impact_pct = sector_impacts.get(sector, market_drop)
        pnl = position_value * (impact_pct / 100)
        total_pnl += pnl
        holdings.append({
            "symbol": sym, "sector": sector, "weight": w,
            "impact_pct": impact_pct, "pnl": pnl,
            "position_value": position_value,
        })
    holdings.sort(key=lambda h: h["pnl"])
    total_impact_pct = (total_pnl / starting_value * 100) if starting_value else 0
    sectors_hit = len({h["sector"] for h in holdings if h["impact_pct"] < 0})
    return {
        "holdings": holdings,
        "total_pnl": total_pnl,
        "total_impact_pct": total_impact_pct,
        "final_value": starting_value + total_pnl,
        "sectors_hit": sectors_hit,
    }


def compute_beta_based_impact(symbols, weights_dict, betas, starting_value, market_drop):
    """Compute per-holding P&L using beta * overall market drop."""
    holdings = []
    total_pnl = 0.0
    weighted_beta = 0.0
    for sym in symbols:
        w = weights_dict.get(sym, 0)
        position_value = w * starting_value
        beta = betas.get(sym, 1.0)
        impact_pct = beta * market_drop
        pnl = position_value * (impact_pct / 100)
        total_pnl += pnl
        weighted_beta += w * beta
        holdings.append({
            "symbol": sym, "beta": beta, "weight": w,
            "impact_pct": impact_pct, "pnl": pnl,
            "position_value": position_value,
        })
    holdings.sort(key=lambda h: h["pnl"])
    total_impact_pct = (total_pnl / starting_value * 100) if starting_value else 0
    return {
        "holdings": holdings,
        "total_pnl": total_pnl,
        "total_impact_pct": total_impact_pct,
        "final_value": starting_value + total_pnl,
        "portfolio_beta": weighted_beta,
    }


def build_waterfall_chart(holdings, total_pnl, title):
    """Build a Plotly waterfall chart showing per-holding P&L contribution."""
    labels = [h["symbol"] for h in holdings] + ["NET"]
    values = [h["pnl"] for h in holdings] + [total_pnl]
    measures = ["relative"] * len(holdings) + ["total"]
    colors = ["#00C853" if v >= 0 else "#FF4B4B" for v in values[:-1]]
    colors.append("#FF8C00")  # net total bar

    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        connector=dict(line=dict(color="#1E2D3D", width=1)),
        increasing=dict(marker=dict(color="#00C853")),
        decreasing=dict(marker=dict(color="#FF4B4B")),
        totals=dict(marker=dict(color="#FF8C00")),
        textposition="outside",
        text=[f"${v:+,.0f}" for v in values],
        textfont=dict(color="#8899AA", size=9),
    ))
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color="#FFA500", size=14, family="Consolas, monospace"),
        ),
        xaxis=dict(color="#8899AA", tickfont=dict(color="#8899AA", size=9),
                   fixedrange=True),
        yaxis=dict(title="P&L ($)", color="#8899AA", gridcolor="#1E2D3D",
                   tickfont=dict(color="#8899AA"), tickformat="$,.0f",
                   fixedrange=True),
        height=450,
        paper_bgcolor="#0A0E17", plot_bgcolor="#0A0E17",
        font=dict(family="Consolas, monospace", color="#FF8C00"),
        margin=dict(t=50, b=80, l=70, r=30),
        showlegend=False,
    )
    return fig


# ── Helper: Yahoo Finance link ────────────────────────────────────────────────
def yahoo_link(symbol):
    return f"https://finance.yahoo.com/quote/{symbol}"


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("LANE ENTERPRISES, INC. PORTFOLIO TERMINAL")

# ── Header Metrics ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total Account Value", f"${total_value:,.2f}")
col2.metric("Cash Balance", f"${cash_balance:,.2f}")
day_change_display = f"-${abs(day_change):,.2f}" if day_change < 0 else f"${day_change:,.2f}"
col3.metric("Day Change", day_change_display,
            delta=f"{day_change_pct:+.2f}%",
            delta_color="normal")

if day_change < 0:
    st.markdown(
        """<style>
        [data-testid="stMetric"]:nth-child(3) [data-testid="stMetricValue"] {
            color: #FF4B4B !important;
        }
        [data-testid="stMetric"]:nth-child(3) [data-testid="stMetricDelta"] {
            color: #FF4B4B !important;
        }
        [data-testid="stMetric"]:nth-child(3) [data-testid="stMetricDelta"] svg {
            fill: #FF4B4B !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """<style>
        [data-testid="stMetric"]:nth-child(3) [data-testid="stMetricValue"] {
            color: #00C853 !important;
        }
        [data-testid="stMetric"]:nth-child(3) [data-testid="stMetricDelta"] {
            color: #00C853 !important;
        }
        [data-testid="stMetric"]:nth-child(3) [data-testid="stMetricDelta"] svg {
            fill: #00C853 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Prepare position data ────────────────────────────────────────────────────
if positions:
    df_pos = pd.DataFrame(positions)
    symbols = [p["symbol"] for p in positions if p["symbol"] != "N/A"]
    total_invested = df_pos["market_value"].sum()
    weights = {row["symbol"]: row["market_value"] / total_invested
               for _, row in df_pos.iterrows() if total_invested > 0}

    # ── Sharpe Ratio ──────────────────────────────────────────────────────
    prices_df = fetch_price_history(symbols)
    sharpe = compute_sharpe_ratio(prices_df, weights)

    # ── Portfolio Metrics Bar ─────────────────────────────────────────────
    fundamentals = fetch_fundamentals(symbols)
    metric_defs = {
        "pe_ratio": ("P/E", "x"),
        "forward_pe": ("Fwd P/E", "x"),
        "ev_ebitda": ("EV/EBITDA", "x"),
        "price_sales": ("P/S", "x"),
        "price_fcf": ("P/FCF", "x"),
        "debt_equity": ("D/E", "%"),
    }
    weighted_metrics = {}
    for key, (label, suffix) in metric_defs.items():
        weighted_sum = 0.0
        weight_sum = 0.0
        for sym, w in weights.items():
            val = fundamentals.get(sym, {}).get(key)
            if val is not None and val > 0:
                weighted_sum += val * w
                weight_sum += w
        weighted_metrics[label] = (weighted_sum / weight_sum if weight_sum > 0 else None, suffix)

    # Add Sharpe
    if sharpe is not None:
        sharpe_color = "#00C853" if sharpe > 1 else ("#FFD600" if sharpe > 0 else "#FF4B4B")
    else:
        sharpe_color = "#FF8C00"

    # Build single horizontal metrics strip
    cells_html = ""
    # Sharpe first
    sharpe_val = f"{sharpe:.2f}" if sharpe is not None else "N/A"
    cells_html += (
        f'<div style="flex:1; text-align:center; padding:10px 4px; min-width:0;">'
        f'<div style="color:#8899AA; font-size:0.65rem; text-transform:uppercase; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">Sharpe</div>'
        f'<div style="color:{sharpe_color}; font-size:1.4rem; font-weight:bold; '
        f'white-space:nowrap;">{sharpe_val}</div></div>'
    )
    # Fundamental metrics
    for label, (val, suffix) in weighted_metrics.items():
        if val is not None:
            display = f"{val:.2f}%" if suffix == "%" else f"{val:.1f}x"
        else:
            display = "N/A"
        cells_html += (
            f'<div style="flex:1; text-align:center; padding:10px 4px; min-width:0; '
            f'border-left:1px solid #1E2D3D;">'
            f'<div style="color:#8899AA; font-size:0.65rem; text-transform:uppercase; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{label}</div>'
            f'<div style="color:#FF8C00; font-size:1.4rem; font-weight:bold; '
            f'white-space:nowrap;">{display}</div></div>'
        )

    st.markdown(
        f'<div style="display:flex; background:#111921; border:1px solid #1E2D3D; '
        f'border-radius:4px; font-family:Consolas,monospace; overflow:hidden;">'
        f'{cells_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_positions, tab_allocation, tab_correlation, tab_trades, tab_sim, tab_tariffs = st.tabs([
        "POSITIONS", "ALLOCATION", "CORRELATION MATRIX", "TRADE LOG", "SIMULATION", "TARIFFS"
    ])

    # ── Positions Table ───────────────────────────────────────────────────
    with tab_positions:
        df_table = df_pos.copy()
        # Build clickable symbol links
        df_table["Symbol"] = df_table["symbol"].apply(
            lambda s: f'<a href="{yahoo_link(s)}" target="_blank">{s}</a>'
        )
        df_table["Weight"] = df_table["market_value"] / total_invested * 100
        df_table = df_table.rename(columns={
            "quantity": "Shares",
            "market_value": "Market Value",
            "day_change": "Day Change",
            "day_change_pct": "Day Chg %",
            "avg_price": "Avg Price",
        })
        df_table = df_table.sort_values("Market Value", ascending=False)

        display_cols = ["Symbol", "Shares", "Market Value", "Day Change", "Day Chg %", "Avg Price", "Weight"]
        df_display = df_table[display_cols].copy()

        # Format for display
        df_display["Shares"] = df_display["Shares"].apply(lambda x: f"{x:.2f}")
        df_display["Market Value"] = df_display["Market Value"].apply(lambda x: f"${x:,.2f}")
        def _color_cell(val, fmt):
            color = "#FF4B4B" if val < 0 else "#00C853"
            bg = "rgba(255,75,75,0.15)" if val < 0 else "rgba(0,200,83,0.15)"
            text = fmt(val)
            return (f'<span style="color:{color}; background:{bg}; '
                    f'padding:2px 6px; border-radius:3px;">{text}</span>')

        df_display["Day Change"] = df_display["Day Change"].apply(
            lambda x: _color_cell(x, lambda v: f"${v:,.2f}")
        )
        df_display["Day Chg %"] = df_display["Day Chg %"].apply(
            lambda x: _color_cell(x, lambda v: f"{v:+.2f}%")
        )
        df_display["Avg Price"] = df_display["Avg Price"].apply(lambda x: f"${x:,.2f}")
        df_display["Weight"] = df_display["Weight"].apply(lambda x: f"{x:.1f}%")

        html_table = df_display.to_html(escape=False, index=False)
        html_table = html_table.replace(
            '<table',
            '<table style="width:100%; border-collapse:collapse; font-family:Consolas,monospace; '
            'font-size:0.9rem;"'
        )
        html_table = html_table.replace(
            '<th>',
            '<th style="background:#111921; color:#8899AA; padding:8px 12px; '
            'text-align:right; border-bottom:2px solid #1E2D3D; text-transform:uppercase; font-size:0.75rem;">'
        )
        html_table = html_table.replace(
            '<td>',
            '<td style="padding:8px 12px; color:#FF8C00; border-bottom:1px solid #1E2D3D; text-align:right;">'
        )
        st.markdown(html_table, unsafe_allow_html=True)

    # ── Allocation Donut ──────────────────────────────────────────────────
    with tab_allocation:
        df_alloc = df_pos.copy()
        if cash_balance > 0:
            df_alloc = pd.concat([df_alloc, pd.DataFrame([{
                "symbol": "CASH",
                "quantity": 0,
                "market_value": cash_balance,
                "day_change": 0,
                "avg_price": 0,
            }])], ignore_index=True)

        fig_donut = px.pie(
            df_alloc, values="market_value", names="symbol", hole=0.45,
        )
        fig_donut.update_traces(
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(family="Consolas, monospace", size=13, color="#FF8C00"),
        )
        fig_donut.update_layout(
            showlegend=True,
            margin=dict(t=40, b=40, l=80, r=80),
            height=700,
            paper_bgcolor="#0A0E17",
            plot_bgcolor="#0A0E17",
            font=dict(family="Consolas, monospace", color="#FF8C00"),
            legend=dict(font=dict(color="#8899AA", size=12)),
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

        st.markdown("---")

        # ── Market Cap & Region Breakdown (combined sunburst) ─────────────
        mcap_data = fetch_market_cap_data(symbols)
        df_mcap = pd.DataFrame(mcap_data)
        df_mcap = df_mcap.merge(
            df_pos[["symbol", "market_value"]],
            on="symbol", how="left",
        )

        # Build sunburst: Region → Cap Category → Symbol
        fig_sun = px.sunburst(
            df_mcap[df_mcap["market_value"] > 0],
            path=["region", "cap_category", "symbol"],
            values="market_value",
            color="cap_category",
            color_discrete_map={
                "Mega Cap": "#B71C1C",
                "Large Cap": "#FF6D00",
                "Mid Cap": "#FFD600",
                "Small Cap": "#00C853",
                "Micro Cap": "#0D47A1",
                "Unknown": "#555555",
            },
        )
        fig_sun.update_traces(
            textinfo="label+percent parent",
            textfont=dict(family="Consolas, monospace", size=12),
            insidetextorientation="radial",
        )
        fig_sun.update_layout(
            title=dict(
                text="MARKET CAP & REGION BREAKDOWN",
                font=dict(color="#FFA500", size=14, family="Consolas, monospace"),
            ),
            margin=dict(t=50, b=20, l=20, r=20),
            height=650,
            paper_bgcolor="#0A0E17",
            plot_bgcolor="#0A0E17",
            font=dict(family="Consolas, monospace", color="#FF8C00"),
        )
        st.plotly_chart(fig_sun, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

    # ── Correlation Matrix Heatmap ────────────────────────────────────────
    with tab_correlation:
        if not prices_df.empty and len(prices_df.columns) > 1:
            corr_period = st.selectbox(
                "Correlation Period",
                ["3mo", "6mo", "1y", "2y", "5y"],
                index=2,
                key="corr_period",
            )
            if corr_period != "1y":
                corr_prices = fetch_price_history(symbols, period=corr_period)
            else:
                corr_prices = prices_df

            if not corr_prices.empty:
                returns = corr_prices.pct_change().dropna()
                corr_matrix = returns.corr()

                # Fixed colorscale: -1 (blue) → 0 (green) → 0.5 (amber) → 0.8 (red) → 1.0 (deep red)
                zmin = -1.0
                zmax = 1.0
                stops = [
                    [0.0,  "#0D47A1"],   # -1.0: deep blue
                    [0.5,  "#00C853"],   #  0.0: bright green
                    [0.75, "#FFAB00"],   #  0.5: amber/orange
                    [0.9,  "#D50000"],   #  0.8: red
                    [1.0,  "#B71C1C"],   #  1.0: deepest red
                ]

                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale=stops,
                    zmin=zmin,
                    zmax=zmax,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    textfont=dict(size=12, family="Consolas, monospace", color="#FFFFFF"),
                    colorbar=dict(
                        title=dict(text="Corr", font=dict(color="#8899AA")),
                        tickfont=dict(color="#8899AA"),
                    ),
                ))
                fig_corr.update_layout(
                    height=max(500, len(symbols) * 55),
                    paper_bgcolor="#0A0E17",
                    plot_bgcolor="#0A0E17",
                    font=dict(family="Consolas, monospace", color="#FF8C00"),
                    xaxis=dict(
                        tickfont=dict(color="#FF8C00", size=12),
                        side="bottom",
                        fixedrange=True,
                    ),
                    yaxis=dict(
                        tickfont=dict(color="#FF8C00", size=12),
                        autorange="reversed",
                        fixedrange=True,
                    ),
                    margin=dict(t=30, b=30, l=100, r=30),
                )
                st.plotly_chart(fig_corr, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
        else:
            st.info("Need at least 2 positions to compute correlation.")

    # ── Trade Log ─────────────────────────────────────────────────────────
    with tab_trades:
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.date.today() - datetime.timedelta(days=365),
                key="trade_start",
            )
        with tcol2:
            end_date = st.date_input(
                "End Date",
                value=datetime.date.today(),
                key="trade_end",
            )

        all_trades = []
        for acct_hash in account_numbers:
            txns = fetch_transactions(acct_hash, start_date, end_date)
            for t in txns:
                items = t.get("transferItems", [])
                # Filter to the actual security (skip fee items like CURRENCY)
                for item in items:
                    inst = item.get("instrument", {})
                    asset_type = inst.get("assetType", "")
                    if asset_type == "CURRENCY":
                        continue
                    symbol = inst.get("symbol", "N/A")
                    trade_date = t.get("tradeDate", t.get("time", ""))
                    if trade_date and len(trade_date) >= 10:
                        trade_date = trade_date[:10]
                    effect = item.get("positionEffect", "")
                    side = "BUY" if effect == "OPENING" else ("SELL" if effect == "CLOSING" else effect)
                    all_trades.append({
                        "Date": trade_date,
                        "Symbol": symbol,
                        "Side": side,
                        "Qty": abs(item.get("amount", 0)),
                        "Price": item.get("price", 0),
                        "Total": abs(item.get("cost", 0)),
                        "Type": inst.get("type", asset_type).replace("_", " ").title(),
                    })

        if all_trades:
            df_trades = pd.DataFrame(all_trades)
            df_trades = df_trades.sort_values("Date", ascending=False)

            # Build clickable symbol links
            df_trades["Symbol"] = df_trades["Symbol"].apply(
                lambda s: f'<a href="{yahoo_link(s)}" target="_blank">{s}</a>' if s != "N/A" else s
            )

            # Color-code buy/sell
            df_trades["Side"] = df_trades["Side"].apply(
                lambda s: f'<span style="color:#00C853;">{s}</span>' if s == "BUY"
                else f'<span style="color:#FF4B4B;">{s}</span>'
            )

            df_trades["Price"] = df_trades["Price"].apply(lambda x: f"${x:,.2f}" if x else "")
            df_trades["Total"] = df_trades["Total"].apply(lambda x: f"${x:,.2f}" if x else "")
            df_trades["Qty"] = df_trades["Qty"].apply(lambda x: f"{x:,.0f}" if x else "")

            html_trades = df_trades.to_html(escape=False, index=False)
            html_trades = html_trades.replace(
                '<table',
                '<table style="width:100%; border-collapse:collapse; font-family:Consolas,monospace; '
                'font-size:0.85rem;"'
            )
            html_trades = html_trades.replace(
                '<th>',
                '<th style="background:#111921; color:#8899AA; padding:8px 12px; '
                'text-align:right; border-bottom:2px solid #1E2D3D; text-transform:uppercase; font-size:0.75rem;">'
            )
            html_trades = html_trades.replace(
                '<td>',
                '<td style="padding:6px 12px; color:#FF8C00; border-bottom:1px solid #1E2D3D; text-align:right;">'
            )
            st.markdown(html_trades, unsafe_allow_html=True)
            st.caption(f"{len(all_trades)} trades found")
        else:
            st.info("No trades found for the selected period.")

    # ── Simulation Tab ─────────────────────────────────────────────────────
    with tab_sim:
        st.markdown(
            '<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.85rem; margin-bottom:0;">'
            'Correlated GBM Monte Carlo &mdash; 1,000 simulations, 252 trading days (1 year). '
            'Parameters derived from trailing 1-year daily returns.</p>',
            unsafe_allow_html=True,
        )

        # Recalculate button uses session state counter as seed differentiator
        if "mc_run_count" not in st.session_state:
            st.session_state.mc_run_count = 0

        if st.button("Recalculate Simulation", key="mc_recalc"):
            st.session_state.mc_run_count += 1

        # Run simulation (seed varies with run count so button gives new paths)
        np.random.seed(42 + st.session_state.mc_run_count)
        mc_paths, mc_stats = run_monte_carlo(
            prices_df, weights, total_value, n_sims=1000, n_days=252,
        )

        if mc_paths is not None and mc_stats is not None:
            # ── Stats strip ───────────────────────────────────────────────
            stat_items = [
                ("Drift (Ann.)", f"{mc_stats['annual_drift']:+.1f}%",
                 "#00C853" if mc_stats["annual_drift"] >= 0 else "#FF4B4B"),
                ("Median Return", f"{mc_stats['median_return']:+.1f}%",
                 "#00C853" if mc_stats["median_return"] >= 0 else "#FF4B4B"),
                ("5th Pctl", f"{mc_stats['p5']:+.1f}%",
                 "#FF4B4B"),
                ("95th Pctl", f"{mc_stats['p95']:+.1f}%",
                 "#00C853"),
                ("Win Rate (>10%)", f"{mc_stats['win_rate']:.1f}%",
                 "#00C853" if mc_stats["win_rate"] >= 50 else "#FF4B4B"),
                ("P(Loss)", f"{mc_stats['p_loss']:.1f}%",
                 "#FF4B4B" if mc_stats["p_loss"] > 50 else "#FFD600"),
                ("VaR (95%)", f"{mc_stats['var_95']:+.1f}%",
                 "#FF4B4B"),
                ("Max Drawdown", f"{mc_stats['max_drawdown']:.1f}%",
                 "#FF4B4B"),
            ]
            stat_cells = ""
            for j, (lbl, val_str, clr) in enumerate(stat_items):
                border = "border-left:1px solid #1E2D3D;" if j > 0 else ""
                stat_cells += (
                    f'<div style="flex:1; text-align:center; padding:10px 4px; '
                    f'min-width:0; {border}">'
                    f'<div style="color:#8899AA; font-size:0.65rem; text-transform:uppercase; '
                    f'white-space:nowrap;">{ lbl}</div>'
                    f'<div style="color:{clr}; font-size:1.3rem; font-weight:bold; '
                    f'white-space:nowrap;">{val_str}</div></div>'
                )
            st.markdown(
                f'<div style="display:flex; background:#111921; border:1px solid #1E2D3D; '
                f'border-radius:4px; font-family:Consolas,monospace; overflow:hidden; '
                f'margin:12px 0;">{stat_cells}</div>',
                unsafe_allow_html=True,
            )

            # ── Value projections ─────────────────────────────────────────
            val_cells = ""
            val_items = [
                ("Starting Value", f"${total_value:,.0f}", "#FF8C00"),
                ("Median (1Y)", f"${mc_stats['median_final']:,.0f}",
                 "#00C853" if mc_stats["median_final"] >= total_value else "#FF4B4B"),
                ("Bear Case (5th)", f"${mc_stats['p5_final']:,.0f}", "#FF4B4B"),
                ("Bull Case (95th)", f"${mc_stats['p95_final']:,.0f}", "#00C853"),
            ]
            for j, (lbl, val_str, clr) in enumerate(val_items):
                border = "border-left:1px solid #1E2D3D;" if j > 0 else ""
                val_cells += (
                    f'<div style="flex:1; text-align:center; padding:10px 4px; '
                    f'min-width:0; {border}">'
                    f'<div style="color:#8899AA; font-size:0.65rem; text-transform:uppercase; '
                    f'white-space:nowrap;">{lbl}</div>'
                    f'<div style="color:{clr}; font-size:1.3rem; font-weight:bold; '
                    f'white-space:nowrap;">{val_str}</div></div>'
                )
            st.markdown(
                f'<div style="display:flex; background:#111921; border:1px solid #1E2D3D; '
                f'border-radius:4px; font-family:Consolas,monospace; overflow:hidden; '
                f'margin-bottom:16px;">{val_cells}</div>',
                unsafe_allow_html=True,
            )

            # ── Equity paths chart ────────────────────────────────────────
            fig_mc = go.Figure()
            days = np.arange(mc_paths.shape[1])

            # Plot 100 sample paths with low opacity
            display_indices = np.linspace(0, mc_paths.shape[0] - 1, 100, dtype=int)
            for idx in display_indices:
                fig_mc.add_trace(go.Scatter(
                    x=days, y=mc_paths[idx],
                    mode="lines",
                    line=dict(width=0.6, color="rgba(255,140,0,0.12)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # Percentile bands
            p5_path = np.percentile(mc_paths, 5, axis=0)
            p50_path = np.percentile(mc_paths, 50, axis=0)
            p95_path = np.percentile(mc_paths, 95, axis=0)

            fig_mc.add_trace(go.Scatter(
                x=days, y=p95_path, mode="lines",
                line=dict(width=1.5, color="#00C853", dash="dash"),
                name="95th Pctl",
            ))
            fig_mc.add_trace(go.Scatter(
                x=days, y=p50_path, mode="lines",
                line=dict(width=2.5, color="#FF8C00"),
                name="Median",
            ))
            fig_mc.add_trace(go.Scatter(
                x=days, y=p5_path, mode="lines",
                line=dict(width=1.5, color="#FF4B4B", dash="dash"),
                name="5th Pctl",
            ))

            # Starting value reference line
            fig_mc.add_hline(
                y=total_value,
                line_dash="dot", line_color="#8899AA", line_width=1,
                annotation_text="Current",
                annotation_font=dict(color="#8899AA", size=10),
            )

            fig_mc.update_layout(
                title=dict(
                    text="MONTE CARLO EQUITY PATHS (1 YEAR)",
                    font=dict(color="#FFA500", size=14, family="Consolas, monospace"),
                ),
                xaxis=dict(
                    title="Trading Days",
                    color="#8899AA",
                    gridcolor="#1E2D3D",
                    tickfont=dict(color="#8899AA"),
                    fixedrange=True,
                ),
                yaxis=dict(
                    title="Portfolio Value ($)",
                    color="#8899AA",
                    gridcolor="#1E2D3D",
                    tickfont=dict(color="#8899AA"),
                    tickformat="$,.0f",
                    fixedrange=True,
                ),
                height=550,
                paper_bgcolor="#0A0E17",
                plot_bgcolor="#0A0E17",
                font=dict(family="Consolas, monospace", color="#FF8C00"),
                legend=dict(
                    font=dict(color="#8899AA", size=11),
                    bgcolor="rgba(17,25,33,0.8)",
                    bordercolor="#1E2D3D",
                ),
                margin=dict(t=50, b=50, l=70, r=30),
            )
            st.plotly_chart(fig_mc, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

        else:
            st.info("Insufficient data to run Monte Carlo simulation.")

        # ── Historic Stress Tests ─────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<p style="color:#FFA500; font-family:Consolas,monospace; font-size:1rem; '
            'font-weight:bold; text-transform:uppercase; border-bottom:1px solid #333D4D; '
            'padding-bottom:6px;">Historic Stress Tests</p>'
            '<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.8rem; '
            'margin-bottom:4px;">Replays your current portfolio weights through actual '
            'historical daily returns during each crisis period. For each trading day in the '
            'event window, the weighted daily return of your holdings is applied to compute a '
            'cumulative equity curve starting from your current portfolio value.</p>'
            '<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.75rem; '
            'margin-bottom:12px; font-style:italic;">Tickers that did not exist during a '
            'given period are proxied using SPY (S&amp;P 500 ETF) to maintain full portfolio '
            'weight coverage.</p>',
            unsafe_allow_html=True,
        )

        stress_results = {}
        for event_name, (ev_start, ev_end) in STRESS_EVENTS.items():
            result = compute_stress_test(
                symbols, weights, total_value, ev_start, ev_end,
            )
            if result is not None:
                stress_results[event_name] = result

        if stress_results:
            # Summary strip
            summary_cells = ""
            for j, (ev_name, res) in enumerate(stress_results.items()):
                border = "border-left:1px solid #1E2D3D;" if j > 0 else ""
                ret_color = "#00C853" if res["total_return"] >= 0 else "#FF4B4B"
                short_name = ev_name.split("(")[0].strip()
                summary_cells += (
                    f'<div style="flex:1; text-align:center; padding:10px 4px; '
                    f'min-width:0; {border}">'
                    f'<div style="color:#8899AA; font-size:0.6rem; text-transform:uppercase; '
                    f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
                    f'{short_name}</div>'
                    f'<div style="color:{ret_color}; font-size:1.2rem; font-weight:bold; '
                    f'white-space:nowrap;">{res["total_return"]:+.1f}%</div>'
                    f'<div style="color:#8899AA; font-size:0.55rem; white-space:nowrap;">'
                    f'DD: {res["max_drawdown"]:.1f}%</div></div>'
                )
            st.markdown(
                f'<div style="display:flex; background:#111921; border:1px solid #1E2D3D; '
                f'border-radius:4px; font-family:Consolas,monospace; overflow:hidden; '
                f'margin-bottom:16px;">{summary_cells}</div>',
                unsafe_allow_html=True,
            )

            # ── Chart 1: Actual Returns ───────────────────────────────────
            fig_stress = go.Figure()
            colors = {
                "COVID Crash (2020)": "#FF4B4B",
                "2022 Bear Market": "#FF6D00",
                "2008 Financial Crisis": "#FFD600",
                "Dot-com Bust (2000-02)": "#0D47A1",
                "Black Monday (1987)": "#AA00FF",
            }
            for ev_name, res in stress_results.items():
                ec = res["equity_curve"]
                x_days = list(range(len(ec)))
                fig_stress.add_trace(go.Scatter(
                    x=x_days, y=ec.values, mode="lines", name=ev_name,
                    line=dict(width=2.5, color=colors.get(ev_name, "#FF8C00")),
                    hovertemplate=(
                        f"<b>{ev_name}</b><br>"
                        "Day %{x}<br>"
                        "Value: $%{y:,.0f}<extra></extra>"
                    ),
                ))
            fig_stress.add_hline(
                y=total_value, line_dash="dot", line_color="#8899AA", line_width=1,
                annotation_text="Starting Value",
                annotation_font=dict(color="#8899AA", size=10),
            )
            fig_stress.update_layout(
                title=dict(
                    text="ACTUAL HISTORICAL RETURNS",
                    font=dict(color="#FFA500", size=14, family="Consolas, monospace"),
                ),
                xaxis=dict(title="Trading Days From Start", color="#8899AA",
                           gridcolor="#1E2D3D", tickfont=dict(color="#8899AA"),
                           fixedrange=True),
                yaxis=dict(title="Portfolio Value ($)", color="#8899AA",
                           gridcolor="#1E2D3D", tickfont=dict(color="#8899AA"),
                           tickformat="$,.0f", fixedrange=True),
                height=500,
                paper_bgcolor="#0A0E17", plot_bgcolor="#0A0E17",
                font=dict(family="Consolas, monospace", color="#FF8C00"),
                legend=dict(font=dict(color="#8899AA", size=11),
                            bgcolor="rgba(17,25,33,0.8)", bordercolor="#1E2D3D"),
                margin=dict(t=50, b=50, l=70, r=30),
            )
            st.plotly_chart(fig_stress, use_container_width=True,
                            config={"scrollZoom": False, "displayModeBar": False})

            st.markdown(
                '<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.75rem; '
                'margin-top:-8px; margin-bottom:20px; font-style:italic;">'
                'Uses each holding\'s actual daily returns during the event. '
                'Missing tickers are proxied with SPY at their portfolio weight.</p>',
                unsafe_allow_html=True,
            )

            # ── Chart 2: Beta-Adjusted Returns ───────────────────────────
            st.markdown("---")
            betas = compute_betas(symbols)
            portfolio_beta = sum(weights.get(s, 0) * betas.get(s, 1.0) for s in symbols)

            st.markdown(
                f'<p style="color:#FFA500; font-family:Consolas,monospace; font-size:1rem; '
                f'font-weight:bold; text-transform:uppercase; border-bottom:1px solid #333D4D; '
                f'padding-bottom:6px;">Beta-Adjusted Stress Tests</p>'
                f'<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.8rem; '
                f'margin-bottom:4px;">Applies your portfolio\'s weighted beta '
                f'(<span style="color:#FF8C00; font-weight:bold;">{portfolio_beta:.2f}</span>) '
                f'to the S&amp;P 500\'s actual daily returns during each crisis. '
                f'A beta &gt; 1 means the portfolio is more volatile than the market, so '
                f'drawdowns are amplified. A beta &lt; 1 means it is more defensive.</p>'
                f'<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.75rem; '
                f'margin-bottom:12px; font-style:italic;">Formula: daily portfolio return = '
                f'portfolio beta &times; SPY daily return. Beta is computed from trailing '
                f'2-year daily returns vs SPY.</p>',
                unsafe_allow_html=True,
            )

            beta_results = {}
            for event_name, (ev_start, ev_end) in STRESS_EVENTS.items():
                result = compute_beta_stress_test(
                    symbols, weights, betas, total_value, ev_start, ev_end,
                )
                if result is not None:
                    beta_results[event_name] = result

            if beta_results:
                # Summary strip
                beta_summary_cells = ""
                for j, (ev_name, res) in enumerate(beta_results.items()):
                    border = "border-left:1px solid #1E2D3D;" if j > 0 else ""
                    ret_color = "#00C853" if res["total_return"] >= 0 else "#FF4B4B"
                    short_name = ev_name.split("(")[0].strip()
                    beta_summary_cells += (
                        f'<div style="flex:1; text-align:center; padding:10px 4px; '
                        f'min-width:0; {border}">'
                        f'<div style="color:#8899AA; font-size:0.6rem; text-transform:uppercase; '
                        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
                        f'{short_name}</div>'
                        f'<div style="color:{ret_color}; font-size:1.2rem; font-weight:bold; '
                        f'white-space:nowrap;">{res["total_return"]:+.1f}%</div>'
                        f'<div style="color:#8899AA; font-size:0.55rem; white-space:nowrap;">'
                        f'DD: {res["max_drawdown"]:.1f}%</div></div>'
                    )
                st.markdown(
                    f'<div style="display:flex; background:#111921; border:1px solid #1E2D3D; '
                    f'border-radius:4px; font-family:Consolas,monospace; overflow:hidden; '
                    f'margin-bottom:16px;">{beta_summary_cells}</div>',
                    unsafe_allow_html=True,
                )

                # Beta-adjusted equity curves chart
                fig_beta = go.Figure()
                for ev_name, res in beta_results.items():
                    ec = res["equity_curve"]
                    x_days = list(range(len(ec)))
                    fig_beta.add_trace(go.Scatter(
                        x=x_days, y=ec.values, mode="lines", name=ev_name,
                        line=dict(width=2.5, color=colors.get(ev_name, "#FF8C00")),
                        hovertemplate=(
                            f"<b>{ev_name} (Beta-Adj)</b><br>"
                            "Day %{x}<br>"
                            "Value: $%{y:,.0f}<extra></extra>"
                        ),
                    ))
                fig_beta.add_hline(
                    y=total_value, line_dash="dot", line_color="#8899AA", line_width=1,
                    annotation_text="Starting Value",
                    annotation_font=dict(color="#8899AA", size=10),
                )
                fig_beta.update_layout(
                    title=dict(
                        text=f"BETA-ADJUSTED RETURNS (Portfolio \u03B2 = {portfolio_beta:.2f})",
                        font=dict(color="#FFA500", size=14, family="Consolas, monospace"),
                    ),
                    xaxis=dict(title="Trading Days From Start", color="#8899AA",
                               gridcolor="#1E2D3D", tickfont=dict(color="#8899AA"),
                               fixedrange=True),
                    yaxis=dict(title="Portfolio Value ($)", color="#8899AA",
                               gridcolor="#1E2D3D", tickfont=dict(color="#8899AA"),
                               tickformat="$,.0f", fixedrange=True),
                    height=500,
                    paper_bgcolor="#0A0E17", plot_bgcolor="#0A0E17",
                    font=dict(family="Consolas, monospace", color="#FF8C00"),
                    legend=dict(font=dict(color="#8899AA", size=11),
                                bgcolor="rgba(17,25,33,0.8)", bordercolor="#1E2D3D"),
                    margin=dict(t=50, b=50, l=70, r=30),
                )
                st.plotly_chart(fig_beta, use_container_width=True,
                                config={"scrollZoom": False, "displayModeBar": False})
            else:
                st.info("Could not compute beta-adjusted stress tests.")

        else:
            st.info("Could not fetch historical data for stress tests. "
                    "Some holdings may not have existed during these periods.")

        # ── Hypothetical Scenario Analysis ────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<p style="color:#FFA500; font-family:Consolas,monospace; font-size:1rem; '
            'font-weight:bold; text-transform:uppercase; border-bottom:1px solid #333D4D; '
            'padding-bottom:6px;">Hypothetical Scenario Analysis</p>'
            '<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.8rem; '
            'margin-bottom:4px;">Model the impact of hypothetical future events on your '
            'portfolio. Two models are shown side by side:</p>'
            '<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.75rem; '
            'margin-bottom:12px; font-style:italic;">'
            '<b style="color:#FF8C00;">Sector-Based</b> &mdash; applies per-sector impact '
            'estimates to each holding based on its GICS sector. '
            '<b style="color:#FF8C00;">Beta-Adjusted</b> &mdash; applies each holding\'s '
            'trailing beta &times; the overall market drop.</p>',
            unsafe_allow_html=True,
        )

        scenario_names = list(HYPOTHETICAL_SCENARIOS.keys()) + ["Custom Scenario"]
        selected_scenario = st.selectbox(
            "Select Scenario", scenario_names, key="hyp_scenario_select",
        )

        if selected_scenario == "Custom Scenario":
            custom_name = st.text_input("Scenario Name", value="My Custom Scenario",
                                        key="hyp_custom_name")
            hyp_market_drop = st.slider(
                "Overall Market Drop (%)", min_value=-60, max_value=10, value=-20,
                key="hyp_market_drop",
            )
            all_sectors = [
                "Technology", "Communication Services", "Consumer Discretionary",
                "Industrials", "Financials", "Health Care", "Consumer Staples",
                "Utilities", "Energy", "Materials", "Real Estate",
            ]
            st.markdown(
                '<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.75rem; '
                'margin-top:4px;">Adjust per-sector impacts:</p>',
                unsafe_allow_html=True,
            )
            custom_sector_impacts = {}
            cols = st.columns(3)
            for i, sector in enumerate(all_sectors):
                with cols[i % 3]:
                    custom_sector_impacts[sector] = st.slider(
                        sector, min_value=-60, max_value=30,
                        value=hyp_market_drop, key=f"hyp_sec_{sector}",
                    )
            scenario_data = {
                "description": f"Custom scenario: {custom_name}",
                "market_drop": hyp_market_drop,
                "sector_impacts": custom_sector_impacts,
            }
            scenario_label = custom_name
        else:
            scenario_data = HYPOTHETICAL_SCENARIOS[selected_scenario]
            scenario_label = selected_scenario

        st.markdown(
            f'<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.8rem; '
            f'margin-top:8px; margin-bottom:16px;">{scenario_data["description"]}</p>',
            unsafe_allow_html=True,
        )

        # Compute both models
        sector_map = fetch_sector_map(symbols)
        hyp_betas = compute_betas(symbols)
        sector_result = compute_sector_based_impact(
            symbols, weights, total_value, sector_map,
            scenario_data["sector_impacts"], scenario_data["market_drop"],
        )
        beta_result = compute_beta_based_impact(
            symbols, weights, hyp_betas, total_value, scenario_data["market_drop"],
        )

        # Summary cards
        def _hyp_card(label, impact_pct, pnl, final_val, extra_label, extra_val):
            ret_color = "#00C853" if impact_pct >= 0 else "#FF4B4B"
            return (
                f'<div style="flex:1; background:#111921; border:1px solid #1E2D3D; '
                f'border-radius:4px; padding:16px; margin:0 6px; font-family:Consolas,monospace;">'
                f'<div style="color:#FFA500; font-size:0.75rem; font-weight:bold; '
                f'text-transform:uppercase; margin-bottom:10px;">{label}</div>'
                f'<div style="color:{ret_color}; font-size:1.4rem; font-weight:bold;">'
                f'{impact_pct:+.1f}%</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Impact</div>'
                f'<div style="color:{ret_color}; font-size:1.1rem; margin-top:6px;">'
                f'${pnl:+,.0f}</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Estimated P&L</div>'
                f'<div style="color:#FF8C00; font-size:1rem; margin-top:6px;">'
                f'${final_val:,.0f}</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Projected Value</div>'
                f'<div style="color:#FF8C00; font-size:0.9rem; margin-top:6px;">'
                f'{extra_val}</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">{extra_label}</div>'
                f'</div>'
            )

        card_html = (
            '<div style="display:flex; margin-bottom:16px;">'
            + _hyp_card(
                "Sector-Based Model",
                sector_result["total_impact_pct"], sector_result["total_pnl"],
                sector_result["final_value"],
                "Sectors Impacted", str(sector_result["sectors_hit"]),
            )
            + _hyp_card(
                "Beta-Adjusted Model",
                beta_result["total_impact_pct"], beta_result["total_pnl"],
                beta_result["final_value"],
                "Portfolio Beta", f'{beta_result["portfolio_beta"]:.2f}',
            )
            + '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Waterfall charts side by side
        wf_col1, wf_col2 = st.columns(2)
        with wf_col1:
            fig_wf_sector = build_waterfall_chart(
                sector_result["holdings"], sector_result["total_pnl"],
                f"SECTOR-BASED P&L — {scenario_label.upper()}",
            )
            st.plotly_chart(fig_wf_sector, use_container_width=True,
                            config={"scrollZoom": False, "displayModeBar": False})
        with wf_col2:
            fig_wf_beta = build_waterfall_chart(
                beta_result["holdings"], beta_result["total_pnl"],
                f"BETA-ADJUSTED P&L — {scenario_label.upper()}",
            )
            st.plotly_chart(fig_wf_beta, use_container_width=True,
                            config={"scrollZoom": False, "displayModeBar": False})

        # Detailed holdings breakdown
        with st.expander("Detailed Holdings Breakdown"):
            # Merge sector and beta results into one table
            beta_map = {h["symbol"]: h for h in beta_result["holdings"]}
            rows_html = ""
            for h in sector_result["holdings"]:
                sym = h["symbol"]
                bh = beta_map.get(sym, {})
                beta_val = bh.get("beta", 1.0)
                beta_impact = bh.get("impact_pct", 0)
                beta_pnl = bh.get("pnl", 0)
                sec_color = "#00C853" if h["impact_pct"] >= 0 else "#FF4B4B"
                bet_color = "#00C853" if beta_impact >= 0 else "#FF4B4B"
                rows_html += (
                    f'<tr style="border-bottom:1px solid #1E2D3D;">'
                    f'<td style="padding:6px 8px; color:#FF8C00; font-weight:bold;">{sym}</td>'
                    f'<td style="padding:6px 8px; color:#8899AA;">{h["sector"]}</td>'
                    f'<td style="padding:6px 8px; color:#8899AA; text-align:right;">'
                    f'{h["weight"]*100:.1f}%</td>'
                    f'<td style="padding:6px 8px; color:#8899AA; text-align:right;">'
                    f'{beta_val:.2f}</td>'
                    f'<td style="padding:6px 8px; color:{sec_color}; text-align:right;">'
                    f'{h["impact_pct"]:+.1f}%</td>'
                    f'<td style="padding:6px 8px; color:{bet_color}; text-align:right;">'
                    f'{beta_impact:+.1f}%</td>'
                    f'<td style="padding:6px 8px; color:{sec_color}; text-align:right;">'
                    f'${h["pnl"]:+,.0f}</td>'
                    f'<td style="padding:6px 8px; color:{bet_color}; text-align:right;">'
                    f'${beta_pnl:+,.0f}</td>'
                    f'</tr>'
                )
            st.markdown(
                '<div style="overflow-x:auto;">'
                '<table style="width:100%; border-collapse:collapse; font-family:Consolas,monospace; '
                'font-size:0.75rem; background:#0A0E17;">'
                '<thead><tr style="border-bottom:2px solid #333D4D;">'
                '<th style="padding:8px; color:#FFA500; text-align:left;">Symbol</th>'
                '<th style="padding:8px; color:#FFA500; text-align:left;">Sector</th>'
                '<th style="padding:8px; color:#FFA500; text-align:right;">Weight</th>'
                '<th style="padding:8px; color:#FFA500; text-align:right;">Beta</th>'
                '<th style="padding:8px; color:#FFA500; text-align:right;">Sector Impact</th>'
                '<th style="padding:8px; color:#FFA500; text-align:right;">Beta Impact</th>'
                '<th style="padding:8px; color:#FFA500; text-align:right;">Sector P&L</th>'
                '<th style="padding:8px; color:#FFA500; text-align:right;">Beta P&L</th>'
                '</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                '</table></div>',
                unsafe_allow_html=True,
            )

    # ── Tariff Impact Simulator Tab ──────────────────────────────────────
    with tab_tariffs:
        st.markdown(
            '<p style="color:#FFA500; font-family:Consolas,monospace; font-size:1rem; '
            'font-weight:bold; text-transform:uppercase; border-bottom:1px solid #333D4D; '
            'padding-bottom:6px;">Tariff Impact Simulator</p>'
            '<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.8rem; '
            'margin-bottom:12px;">Model the impact of tariff hikes or reductions on your portfolio. '
            'Select a pre-built scenario or build a custom tariff schedule by country and sector.</p>',
            unsafe_allow_html=True,
        )

        # Scenario selector
        tariff_scenario_names = list(TARIFF_SCENARIOS.keys()) + ["Custom Tariff Schedule"]
        selected_tariff = st.selectbox(
            "Select Tariff Scenario", tariff_scenario_names, key="tariff_scenario_select",
        )

        all_countries = ["China", "EU", "Mexico", "Canada", "ROW"]

        if selected_tariff == "Custom Tariff Schedule":
            # Country selection and sliders
            active_countries = st.multiselect(
                "Trade Partners", all_countries, default=["China", "EU"],
                key="tariff_countries",
            )
            custom_tariffs = {}
            if active_countries:
                cols_t = st.columns(min(len(active_countries), 4))
                for i, country in enumerate(active_countries):
                    with cols_t[i % len(cols_t)]:
                        custom_tariffs[country] = st.slider(
                            f"{country} Tariff %", min_value=-30, max_value=100,
                            value=25, key=f"tariff_rate_{country}",
                        )

            custom_pass_through = st.slider(
                "Pass-Through Rate", min_value=0.0, max_value=1.0,
                value=0.5, step=0.05, key="tariff_pass_through",
                help="Fraction of tariff cost passed through to company earnings (0 = no impact, 1 = full impact)",
            )

            # Optional sector-level pass-through adjustments
            sector_pt_overrides = None
            with st.expander("Sector Pass-Through Adjustments (Optional)"):
                st.markdown(
                    '<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.75rem;">'
                    'Override the pass-through rate for specific sectors. Leave at the default to use '
                    'the global rate above.</p>',
                    unsafe_allow_html=True,
                )
                all_sectors_list = list(TARIFF_SECTOR_EXPOSURE.keys())
                sector_pt_overrides = {}
                sec_cols = st.columns(3)
                for i, sector in enumerate(all_sectors_list):
                    with sec_cols[i % 3]:
                        val = st.slider(
                            sector, min_value=0.0, max_value=1.0,
                            value=custom_pass_through, step=0.05,
                            key=f"tariff_spt_{sector}",
                        )
                        sector_pt_overrides[sector] = val

            tariff_rates = custom_tariffs
            tariff_pt = custom_pass_through
            tariff_scenario_desc = "Custom tariff schedule with user-defined rates per country."
        else:
            scenario_info = TARIFF_SCENARIOS[selected_tariff]
            tariff_rates = scenario_info["tariffs"]
            tariff_pt = scenario_info["pass_through"]
            tariff_scenario_desc = scenario_info["description"]
            sector_pt_overrides = None

        # Show scenario description
        st.markdown(
            f'<p style="color:#8899AA; font-family:Consolas,monospace; font-size:0.8rem; '
            f'margin-top:4px; margin-bottom:16px;">{tariff_scenario_desc}</p>',
            unsafe_allow_html=True,
        )

        if tariff_rates:
            # Compute tariff impact
            tariff_sector_map = fetch_sector_map(symbols)
            tariff_result = compute_tariff_impact(
                symbols, weights, total_value, tariff_sector_map,
                tariff_rates, tariff_pt, sector_pt_overrides,
            )

            # ── Summary Cards ────────────────────────────────────────────
            imp_color = "#00C853" if tariff_result["total_impact_pct"] >= 0 else "#FF4B4B"
            card1 = (
                f'<div style="flex:1; background:#111921; border:1px solid #1E2D3D; '
                f'border-radius:4px; padding:16px; margin:0 6px; font-family:Consolas,monospace;">'
                f'<div style="color:#FFA500; font-size:0.75rem; font-weight:bold; '
                f'text-transform:uppercase; margin-bottom:10px;">Portfolio Impact</div>'
                f'<div style="color:{imp_color}; font-size:1.5rem; font-weight:bold;">'
                f'{tariff_result["total_impact_pct"]:+.1f}%</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Estimated Impact</div>'
                f'<div style="color:{imp_color}; font-size:1.2rem; margin-top:6px;">'
                f'${tariff_result["total_pnl"]:+,.0f}</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Estimated P&L</div>'
                f'<div style="color:#FF8C00; font-size:1rem; margin-top:6px;">'
                f'${tariff_result["final_value"]:,.0f}</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Projected Value</div>'
                f'<div style="color:#FF8C00; font-size:0.9rem; margin-top:6px;">'
                f'{tariff_result["avg_tariff"]:.0f}%</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Avg Tariff Rate</div>'
                f'</div>'
            )
            card2 = (
                f'<div style="flex:1; background:#111921; border:1px solid #1E2D3D; '
                f'border-radius:4px; padding:16px; margin:0 6px; font-family:Consolas,monospace;">'
                f'<div style="color:#FFA500; font-size:0.75rem; font-weight:bold; '
                f'text-transform:uppercase; margin-bottom:10px;">Trade Exposure</div>'
                f'<div style="color:#FF8C00; font-size:1.5rem; font-weight:bold;">'
                f'{tariff_result["top_country"]}</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Top Exposure Country</div>'
                f'<div style="color:#FF8C00; font-size:1.2rem; margin-top:6px;">'
                f'{tariff_result["weighted_exposure"]:.1f}%</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Wtd Trade Exposure</div>'
                f'<div style="color:#FF4B4B; font-size:1rem; margin-top:6px;">'
                f'{tariff_result["sectors_losing"]}</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Sectors at Risk</div>'
                f'<div style="color:#00C853; font-size:0.9rem; margin-top:6px;">'
                f'{tariff_result["sectors_gaining"]}</div>'
                f'<div style="color:#8899AA; font-size:0.7rem;">Sectors Gaining</div>'
                f'</div>'
            )
            st.markdown(
                f'<div style="display:flex; margin-bottom:16px;">{card1}{card2}</div>',
                unsafe_allow_html=True,
            )

            # ── Waterfall Chart ──────────────────────────────────────────
            fig_tariff_wf = build_waterfall_chart(
                tariff_result["holdings"], tariff_result["total_pnl"],
                f"PER-HOLDING TARIFF P&L — {selected_tariff.upper()}",
            )
            st.plotly_chart(fig_tariff_wf, use_container_width=True,
                            config={"scrollZoom": False, "displayModeBar": False})

            # ── Heatmap ──────────────────────────────────────────────────
            fig_heatmap = build_tariff_heatmap(
                tariff_result["holdings"], tariff_rates, tariff_sector_map,
            )
            st.plotly_chart(fig_heatmap, use_container_width=True,
                            config={"scrollZoom": False, "displayModeBar": False})

            # ── Detailed Holdings Table ──────────────────────────────────
            with st.expander("Detailed Holdings Breakdown"):
                countries_in_tariff = list(tariff_rates.keys())
                # Build header
                country_headers = "".join(
                    f'<th style="padding:8px; color:#FFA500; text-align:right;">{c}</th>'
                    for c in countries_in_tariff
                )
                header_html = (
                    '<thead><tr style="border-bottom:2px solid #333D4D;">'
                    '<th style="padding:8px; color:#FFA500; text-align:left;">Symbol</th>'
                    '<th style="padding:8px; color:#FFA500; text-align:left;">Sector</th>'
                    '<th style="padding:8px; color:#FFA500; text-align:right;">Weight</th>'
                    '<th style="padding:8px; color:#FFA500; text-align:right;">Impact</th>'
                    '<th style="padding:8px; color:#FFA500; text-align:right;">P&L</th>'
                    f'{country_headers}'
                    '</tr></thead>'
                )
                # Build rows
                detail_rows = ""
                for h in tariff_result["holdings"]:
                    imp_c = "#00C853" if h["impact_pct"] >= 0 else "#FF4B4B"
                    country_cells = ""
                    for c in countries_in_tariff:
                        det = h["country_details"].get(c, {})
                        exp = det.get("exposure", 0) * 100
                        country_cells += (
                            f'<td style="padding:6px 8px; color:#8899AA; text-align:right;">'
                            f'{exp:.0f}%</td>'
                        )
                    detail_rows += (
                        f'<tr style="border-bottom:1px solid #1E2D3D;">'
                        f'<td style="padding:6px 8px; color:#FF8C00; font-weight:bold;">{h["symbol"]}</td>'
                        f'<td style="padding:6px 8px; color:#8899AA;">{h["sector"]}</td>'
                        f'<td style="padding:6px 8px; color:#8899AA; text-align:right;">'
                        f'{h["weight"]*100:.1f}%</td>'
                        f'<td style="padding:6px 8px; color:{imp_c}; text-align:right;">'
                        f'{h["impact_pct"]:+.1f}%</td>'
                        f'<td style="padding:6px 8px; color:{imp_c}; text-align:right;">'
                        f'${h["pnl"]:+,.0f}</td>'
                        f'{country_cells}'
                        f'</tr>'
                    )
                st.markdown(
                    '<div style="overflow-x:auto;">'
                    '<table style="width:100%; border-collapse:collapse; font-family:Consolas,monospace; '
                    'font-size:0.75rem; background:#0A0E17;">'
                    f'{header_html}'
                    f'<tbody>{detail_rows}</tbody>'
                    '</table></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Select at least one trade partner to model tariff impact.")

else:
    st.info("No positions found.")
