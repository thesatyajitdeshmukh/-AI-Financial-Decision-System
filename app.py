import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Decision System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp { background-color: #0a0e1a; color: #e0e6f0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1628 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2d4a;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #0f1e36 0%, #0a1525 100%);
    border: 1px solid #1a3050;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.metric-card h3 {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #5a8fbf;
    margin: 0 0 6px 0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #00e5a0;
    line-height: 1;
}
.metric-card .sub {
    font-size: 12px;
    color: #4a6a8a;
    margin-top: 4px;
}

/* Signal badges */
.signal-buy   { background:#0a2e1a; color:#00e5a0; border:1px solid #00e5a0; padding:4px 14px; border-radius:20px; font-family:'Space Mono',monospace; font-size:12px; font-weight:700; }
.signal-sell  { background:#2e0a0a; color:#ff4d6d; border:1px solid #ff4d6d; padding:4px 14px; border-radius:20px; font-family:'Space Mono',monospace; font-size:12px; font-weight:700; }
.signal-hold  { background:#1a1a0a; color:#ffd166; border:1px solid #ffd166; padding:4px 14px; border-radius:20px; font-family:'Space Mono',monospace; font-size:12px; font-weight:700; }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #3a6a9a;
    text-transform: uppercase;
    letter-spacing: 3px;
    padding: 8px 0 6px 0;
    border-bottom: 1px solid #1a2e44;
    margin-bottom: 16px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1628;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #4a7aa0;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: #1a3050 !important;
    color: #00e5a0 !important;
}

/* Selectbox / inputs */
.stSelectbox label, .stSlider label, .stNumberInput label { color: #5a8fbf; font-size: 13px; }
div[data-baseweb="select"] > div { background:#0f1628; border-color:#1a3050; color:#e0e6f0; }

/* Dataframe */
.stDataFrame { border-radius: 10px; }

/* Divider */
hr { border-color: #1a2e44; }

/* Warning / info boxes */
.stAlert { border-radius: 8px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1a3050; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
plt.style.use("dark_background")
CHART_BG  = "#0a0e1a"
CHART_AX  = "#0f1628"
COLOR_1   = "#00e5a0"
COLOR_2   = "#4a9eff"
COLOR_3   = "#ff4d6d"
COLOR_4   = "#ffd166"

def dark_fig(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_AX)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a3050")
    ax.tick_params(colors="#4a7aa0", labelsize=9)
    ax.xaxis.label.set_color("#4a7aa0")
    ax.yaxis.label.set_color("#4a7aa0")
    ax.title.set_color("#a0c4e0")
    return fig, ax

@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    import yfinance as yf
    frames = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df["Ticker"] = ticker
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    return data

@st.cache_data(show_spinner=False)
def compute_features(data):
    from ta.momentum  import RSIIndicator, StochasticOscillator, ROCIndicator
    from ta.trend     import MACD, EMAIndicator, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from sklearn.mixture import GaussianMixture

    df = data.copy().sort_values(["Ticker","Date"])

    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns and col.lower() in df.columns:
            df[col] = df[col.lower()]

    df["RSI"]        = df.groupby("Ticker")["Close"].transform(lambda x: RSIIndicator(x,14).rsi())
    df["MACD"]       = df.groupby("Ticker")["Close"].transform(lambda x: MACD(x).macd())
    df["EMA_12"]     = df.groupby("Ticker")["Close"].transform(lambda x: EMAIndicator(x,12).ema_indicator())
    df["EMA_26"]     = df.groupby("Ticker")["Close"].transform(lambda x: EMAIndicator(x,26).ema_indicator())
    df["Returns"]    = df.groupby("Ticker")["Close"].pct_change()
    df["Log_Returns"]= df.groupby("Ticker")["Close"].transform(lambda x: np.log(x/x.shift(1)))
    df["Volatility_20"]   = df.groupby("Ticker")["Returns"].transform(lambda x: x.rolling(20).std())
    df["Rolling_Mean_20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
    df["Skewness_20"]     = df.groupby("Ticker")["Returns"].transform(lambda x: x.rolling(20).skew())
    df["Kurtosis_20"]     = df.groupby("Ticker")["Returns"].transform(lambda x: x.rolling(20).kurt())
    df["Close_Lag1"]      = df.groupby("Ticker")["Close"].shift(1)
    df["Close_Lag5"]      = df.groupby("Ticker")["Close"].shift(5)

    # ATR (needs High/Low/Close)
    try:
        df["ATR"] = df.groupby("Ticker").apply(
            lambda g: AverageTrueRange(g["High"], g["Low"], g["Close"]).average_true_range()
        ).reset_index(level=0, drop=True)
    except Exception:
        df["ATR"] = np.nan

    # Market regime
    try:
        returns_arr = df["Returns"].dropna().values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=3, random_state=42)
        regimes = gmm.fit_predict(returns_arr)
        df.loc[df["Returns"].notna(), "market_regime"] = regimes
    except Exception:
        df["market_regime"] = 0

    # Date features
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"]       = df["Date"].dt.month

    df = df.dropna(subset=["RSI","MACD","Returns"])
    return df

def generate_signals(df, ticker):
    t = df[df["Ticker"]==ticker].copy().sort_values("Date")
    signals = []
    for _, row in t.iterrows():
        rsi  = row.get("RSI", 50)
        macd = row.get("MACD", 0)
        if   rsi < 35 and macd > 0: signals.append("BUY")
        elif rsi > 65 and macd < 0: signals.append("SELL")
        else: signals.append("HOLD")
    t["Signal"] = signals
    return t

def simulate_portfolio(signal_df, initial_capital=10000):
    capital = initial_capital
    shares  = 0
    portfolio = []
    prices = signal_df["Close"].values
    sigs   = signal_df["Signal"].values
    for i in range(len(sigs)):
        price = prices[i]
        if sigs[i]=="BUY"  and capital > price:  shares += 1;  capital -= price
        elif sigs[i]=="SELL" and shares > 0:      shares -= 1;  capital += price
        portfolio.append(capital + shares * price)
    return portfolio

def sharpe_ratio(portfolio):
    r = pd.Series(portfolio).pct_change().dropna()
    return (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0

def max_drawdown(portfolio):
    s = pd.Series(portfolio)
    dd = (s - s.cummax()) / s.cummax()
    return dd.min()

def monte_carlo(S0, mu, sigma, T=252, n=500):
    dt = 1/T
    paths = np.zeros((T, n))
    paths[0] = S0
    for t in range(1, T):
        z = np.random.standard_normal(n)
        paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return paths

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

    TICKERS = st.multiselect(
        "Select Tickers",
        ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","SPY"],
        default=["AAPL","MSFT","TSLA"],
    )
    if not TICKERS:
        TICKERS = ["AAPL"]

    col_s, col_e = st.columns(2)
    with col_s:
        START = st.date_input("Start", value=pd.to_datetime("2020-01-01"))
    with col_e:
        END   = st.date_input("End",   value=pd.to_datetime("2024-01-01"))

    st.markdown('<div class="section-header">Portfolio</div>', unsafe_allow_html=True)
    CAPITAL   = st.number_input("Initial Capital ($)", 1000, 1_000_000, 10_000, step=1000)
    SEL_TICKER = st.selectbox("Primary Ticker for Strategy", TICKERS)

    st.markdown('<div class="section-header">Monte Carlo</div>', unsafe_allow_html=True)
    MC_MU    = st.slider("Expected Return (μ)", 0.0, 0.3, 0.08, 0.01)
    MC_SIGMA = st.slider("Volatility (σ)",      0.05, 0.6, 0.2,  0.01)
    MC_SIMS  = st.slider("Simulations",          100, 2000, 500,  100)

    st.markdown('<div class="section-header">Sentiment</div>', unsafe_allow_html=True)
    NEWS_TEXT = st.text_area(
        "News Headline(s) (one per line)",
        "Apple reports record earnings\nFed raises interest rates again\nAI boom drives tech stocks higher",
        height=100,
    )

    load_btn = st.button("🚀  Run Analysis", use_container_width=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding:20px 0 10px 0;'>
  <span style='font-family:"Space Mono",monospace;font-size:22px;color:#00e5a0;font-weight:700;'>
    AI Financial Decision System
  </span>
  <span style='font-family:"DM Sans",sans-serif;font-size:14px;color:#3a6a9a;margin-left:12px;'>
    TFT · FinBERT · RL · GNN · Monte Carlo
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
if not load_btn:
    st.info("👈  Configure settings in the sidebar and click **Run Analysis** to start.")
    st.stop()

# ── LOAD DATA ──
with st.spinner("Fetching market data…"):
    try:
        raw = load_data(TICKERS, str(START), str(END))
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

with st.spinner("Engineering features…"):
    df = compute_features(raw)

# ── TABS ──
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Market Overview",
    "⚙️ Feature Engineering",
    "🤖 Trading Signals & Portfolio",
    "🎲 Monte Carlo Risk",
    "🧠 Model Insights",
    "💬 Sentiment Analysis",
])

# ════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Market Data Summary</div>', unsafe_allow_html=True)

    # KPI row
    ticker_latest = {}
    for t in TICKERS:
        sub = df[df["Ticker"]==t]
        if not sub.empty:
            latest = sub.iloc[-1]
            first  = sub.iloc[0]
            pct    = (latest["Close"] - first["Close"]) / first["Close"] * 100
            ticker_latest[t] = {"price": latest["Close"], "pct": pct, "vol": latest.get("Volatility_20", np.nan)}

    cols = st.columns(len(TICKERS))
    for i, t in enumerate(TICKERS):
        info = ticker_latest.get(t, {})
        price = info.get("price", 0)
        pct   = info.get("pct",   0)
        arrow = "▲" if pct >= 0 else "▼"
        color = "#00e5a0" if pct >= 0 else "#ff4d6d"
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
              <h3>{t}</h3>
              <div class="value">${price:.2f}</div>
              <div class="sub" style="color:{color}">{arrow} {abs(pct):.1f}% total period</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Price chart — all tickers
    fig, ax = dark_fig((13, 4))
    colors_list = [COLOR_1, COLOR_2, COLOR_3, COLOR_4, "#c77dff", "#ff9f1c", "#2ec4b6", "#e76f51"]
    for i, t in enumerate(TICKERS):
        sub = df[df["Ticker"]==t].sort_values("Date")
        if not sub.empty:
            ax.plot(sub["Date"], sub["Close"], label=t, color=colors_list[i % len(colors_list)], linewidth=1.5, alpha=0.9)
    ax.set_title("Closing Prices — All Tickers", fontsize=12, pad=10)
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")
    ax.grid(True, alpha=0.1, color="#1a3050")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Volume chart
    if "Volume" in df.columns:
        fig2, ax2 = dark_fig((13, 3))
        for i, t in enumerate(TICKERS):
            sub = df[df["Ticker"]==t].sort_values("Date")
            if not sub.empty:
                ax2.bar(sub["Date"], sub["Volume"], label=t, alpha=0.5, color=colors_list[i % len(colors_list)], width=1)
        ax2.set_title("Trading Volume", fontsize=12, pad=10)
        ax2.set_ylabel("Volume")
        ax2.legend(fontsize=8, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")
        ax2.grid(True, alpha=0.1, color="#1a3050")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Raw data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            df[["Date","Ticker","Open","High","Low","Close","Volume"]].tail(200).reset_index(drop=True),
            use_container_width=True,
        )

# ════════════════════════════════════════════
# TAB 2 — FEATURE ENGINEERING
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Technical Indicators</div>', unsafe_allow_html=True)

    ft_ticker = st.selectbox("Ticker", TICKERS, key="ft_ticker")
    sub = df[df["Ticker"]==ft_ticker].sort_values("Date").tail(252)

    # RSI
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.patch.set_facecolor(CHART_BG)

    ax_p, ax_r, ax_m = axes
    for ax in axes:
        ax.set_facecolor(CHART_AX)
        for spine in ax.spines.values(): spine.set_edgecolor("#1a3050")
        ax.tick_params(colors="#4a7aa0", labelsize=9)
        ax.yaxis.label.set_color("#4a7aa0")
        ax.grid(True, alpha=0.08, color="#1a3050")

    ax_p.plot(sub["Date"], sub["Close"],   color=COLOR_2, lw=1.5, label="Close")
    ax_p.plot(sub["Date"], sub["EMA_12"],  color=COLOR_1, lw=1,   label="EMA 12", alpha=0.7)
    ax_p.plot(sub["Date"], sub["EMA_26"],  color=COLOR_4, lw=1,   label="EMA 26", alpha=0.7)
    if "Rolling_Mean_20" in sub.columns:
        ax_p.plot(sub["Date"], sub["Rolling_Mean_20"], color="#c77dff", lw=1, label="MA 20", alpha=0.7)
    ax_p.set_title(f"{ft_ticker} — Price & Moving Averages", color="#a0c4e0", fontsize=11)
    ax_p.legend(fontsize=8, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")

    ax_r.plot(sub["Date"], sub["RSI"], color=COLOR_3, lw=1.5)
    ax_r.axhline(70, color=COLOR_3, ls="--", lw=0.8, alpha=0.5)
    ax_r.axhline(30, color=COLOR_1, ls="--", lw=0.8, alpha=0.5)
    ax_r.set_ylim(0, 100)
    ax_r.set_title("RSI (14)", color="#a0c4e0", fontsize=10)

    ax_m.plot(sub["Date"], sub["MACD"], color=COLOR_2, lw=1.5)
    ax_m.axhline(0, color="#1a3050", lw=1)
    ax_m.set_title("MACD", color="#a0c4e0", fontsize=10)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Volatility & Returns
    st.markdown('<div class="section-header">Statistical Features</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = dark_fig((7, 3))
        ax.fill_between(sub["Date"], sub["Volatility_20"], color=COLOR_4, alpha=0.6)
        ax.plot(sub["Date"], sub["Volatility_20"], color=COLOR_4, lw=1)
        ax.set_title("Rolling 20-Day Volatility", fontsize=11, pad=8)
        ax.grid(True, alpha=0.1, color="#1a3050")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        if "Returns" in sub.columns:
            fig, ax = dark_fig((7, 3))
            clean_ret = sub["Returns"].dropna()
            ax.hist(clean_ret, bins=50, color=COLOR_2, alpha=0.8, edgecolor="#0a0e1a")
            ax.axvline(clean_ret.mean(), color=COLOR_1, lw=1.5, ls="--", label=f"Mean {clean_ret.mean():.4f}")
            ax.set_title("Return Distribution", fontsize=11, pad=8)
            ax.legend(fontsize=8, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")
            ax.grid(True, alpha=0.1, color="#1a3050")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Market Regime
    if "market_regime" in df.columns:
        st.markdown('<div class="section-header">Market Regime Detection (GMM)</div>', unsafe_allow_html=True)
        fig, ax = dark_fig((13, 3))
        regime_colors = {0: COLOR_1, 1: COLOR_4, 2: COLOR_3}
        sub2 = df[df["Ticker"]==ft_ticker].sort_values("Date")
        for regime in [0, 1, 2]:
            mask = sub2["market_regime"] == regime
            labels = {0: "Bull", 1: "Neutral", 2: "Bear"}
            ax.scatter(sub2.loc[mask, "Date"], sub2.loc[mask, "Returns"],
                       c=regime_colors.get(regime, "#888"), s=4, alpha=0.6, label=labels.get(regime, str(regime)))
        ax.axhline(0, color="#1a3050", lw=1)
        ax.set_title("Market Regime Detection", fontsize=11)
        ax.legend(fontsize=9, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")
        ax.grid(True, alpha=0.08)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Feature table
    with st.expander("📋 Feature Data Table"):
        feat_cols = ["Date","Ticker","Close","RSI","MACD","EMA_12","EMA_26","Volatility_20","Returns","Log_Returns"]
        avail = [c for c in feat_cols if c in df.columns]
        st.dataframe(df[df["Ticker"]==ft_ticker][avail].tail(100).reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════
# TAB 3 — TRADING SIGNALS & PORTFOLIO
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">RSI + MACD Signal Strategy</div>', unsafe_allow_html=True)

    signal_df = generate_signals(df, SEL_TICKER)
    portfolio  = simulate_portfolio(signal_df, CAPITAL)

    # KPIs
    final_val  = portfolio[-1] if portfolio else CAPITAL
    total_ret  = (final_val - CAPITAL) / CAPITAL * 100
    sr         = sharpe_ratio(portfolio)
    mdd        = max_drawdown(portfolio) * 100
    n_buy  = (signal_df["Signal"]=="BUY").sum()
    n_sell = (signal_df["Signal"]=="SELL").sum()

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, sub_t in [
        (k1, "Final Portfolio",  f"${final_val:,.0f}",   f"{total_ret:+.1f}% return"),
        (k2, "Sharpe Ratio",     f"{sr:.3f}",             "annualised"),
        (k3, "Max Drawdown",     f"{mdd:.1f}%",           "peak to trough"),
        (k4, "Trades",           f"{n_buy+n_sell}",       f"↑{n_buy} buys  ↓{n_sell} sells"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <h3>{label}</h3>
          <div class="value">{val}</div>
          <div class="sub">{sub_t}</div>
        </div>""", unsafe_allow_html=True)

    # Portfolio chart
    fig, ax = dark_fig((13, 4))
    ax.fill_between(range(len(portfolio)), portfolio, CAPITAL, where=[p > CAPITAL for p in portfolio], color=COLOR_1, alpha=0.2)
    ax.fill_between(range(len(portfolio)), portfolio, CAPITAL, where=[p <= CAPITAL for p in portfolio], color=COLOR_3, alpha=0.2)
    ax.plot(portfolio, color=COLOR_2, lw=2, label="Portfolio Value")
    ax.axhline(CAPITAL, color="#4a7aa0", ls="--", lw=1, label=f"Initial ${CAPITAL:,}")
    ax.set_title(f"{SEL_TICKER} — Portfolio Value Over Time", fontsize=11)
    ax.set_xlabel("Trading Days"); ax.set_ylabel("Value (USD)")
    ax.legend(fontsize=9, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")
    ax.grid(True, alpha=0.1)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Drawdown chart
        # Drawdown chart
    pf_series = pd.Series(portfolio)

    cummax = pf_series.cummax()
    cummax = cummax.replace(0, np.nan)

    drawdown = (pf_series - cummax) / cummax * 100

    # 🔥 CRITICAL FIX
    drawdown = drawdown.astype(float).replace([np.inf, -np.inf], 0).fillna(0)

    fig2, ax2 = dark_fig((13, 3))

    x = np.arange(len(drawdown))
    y = drawdown.to_numpy()   # ✅ convert to clean numpy array

    ax2.fill_between(x, y, 0, color=COLOR_3, alpha=0.5)
    ax2.plot(x, y, color=COLOR_3, lw=1)

    ax2.set_title("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("%")
    ax2.grid(True, alpha=0.1)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Signal chart
    st.markdown('<div class="section-header">Signal Overlay on Price</div>', unsafe_allow_html=True)
    fig3, ax3 = dark_fig((13, 4))
    ax3.plot(signal_df["Date"], signal_df["Close"], color=COLOR_2, lw=1.5, zorder=1)
    buys  = signal_df[signal_df["Signal"]=="BUY"]
    sells = signal_df[signal_df["Signal"]=="SELL"]
    ax3.scatter(buys["Date"],  buys["Close"],  marker="^", color=COLOR_1, s=40, zorder=3, label="BUY")
    ax3.scatter(sells["Date"], sells["Close"], marker="v", color=COLOR_3, s=40, zorder=3, label="SELL")
    ax3.set_title(f"{SEL_TICKER} — Buy/Sell Signals", fontsize=11)
    ax3.legend(fontsize=9, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")
    ax3.grid(True, alpha=0.1)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Signal table
    with st.expander("📋 Full Signal Log"):
        show_sig = signal_df[["Date","Close","RSI","MACD","Signal"]].tail(100).reset_index(drop=True)
        st.dataframe(show_sig, use_container_width=True)

# ════════════════════════════════════════════
# TAB 4 — MONTE CARLO RISK
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Monte Carlo Simulation</div>', unsafe_allow_html=True)

    sub_mc = df[df["Ticker"]==SEL_TICKER].sort_values("Date")
    S0_val = float(sub_mc["Close"].iloc[-1]) if not sub_mc.empty else 100.0

    with st.spinner("Running simulations…"):
        paths = monte_carlo(S0_val, MC_MU, MC_SIGMA, T=252, n=MC_SIMS)

    final_p = paths[-1]
    var_95  = np.percentile(final_p, 5)
    cvar_95 = final_p[final_p <= var_95].mean()
    exp_p   = final_p.mean()
    p5, p95 = np.percentile(final_p, 5), np.percentile(final_p, 95)

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, val, sub_t in [
        (mc1, "Expected Price",  f"${exp_p:.2f}",  f"from ${S0_val:.2f}"),
        (mc2, "VaR 95%",        f"${var_95:.2f}",  "5th percentile"),
        (mc3, "CVaR 95%",       f"${cvar_95:.2f}", "expected shortfall"),
        (mc4, "90% CI",         f"${p5:.0f}–${p95:.0f}", "price range"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <h3>{label}</h3>
          <div class="value">{val}</div>
          <div class="sub">{sub_t}</div>
        </div>""", unsafe_allow_html=True)

    # Paths chart
    fig, ax = dark_fig((13, 5))
    for i in range(min(150, MC_SIMS)):
        ax.plot(paths[:, i], alpha=0.07, color=COLOR_2, lw=0.8)
    ax.plot(paths.mean(axis=1), color=COLOR_1,  lw=2.5, label="Mean Path", zorder=5)
    ax.plot(np.percentile(paths, 5,  axis=1), color=COLOR_3, lw=1.5, ls="--", label="5th pct")
    ax.plot(np.percentile(paths, 95, axis=1), color=COLOR_4, lw=1.5, ls="--", label="95th pct")
    ax.axhline(S0_val, color="#4a7aa0", ls=":", lw=1)
    ax.set_title(f"Monte Carlo — {SEL_TICKER} ({MC_SIMS:,} simulations, 252 days)", fontsize=11)
    ax.set_xlabel("Trading Days"); ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=9, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")
    ax.grid(True, alpha=0.08)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Final distribution
    fig2, ax2 = dark_fig((13, 3))
    ax2.hist(final_p, bins=80, color=COLOR_2, alpha=0.75, edgecolor="#0a0e1a")
    ax2.axvline(var_95,  color=COLOR_3, lw=2, ls="--", label=f"VaR 95% ${var_95:.2f}")
    ax2.axvline(exp_p,   color=COLOR_1, lw=2, ls="--", label=f"Expected ${exp_p:.2f}")
    ax2.axvline(S0_val,  color=COLOR_4, lw=2, ls=":",  label=f"Current  ${S0_val:.2f}")
    ax2.set_title("Final Price Distribution", fontsize=11)
    ax2.set_xlabel("Price (USD)")
    ax2.legend(fontsize=9, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")
    ax2.grid(True, alpha=0.1)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ════════════════════════════════════════════
# TAB 5 — MODEL INSIGHTS
# ════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Portfolio Optimization (Mean-Variance)</div>', unsafe_allow_html=True)

    if len(TICKERS) >= 2:
        price_pivot = df.pivot_table(index="Date", columns="Ticker", values="Close")
        price_pivot = price_pivot.dropna(how="any")
        returns_mat  = price_pivot.pct_change().dropna()
        cov_matrix   = returns_mat.cov() * 252
        mean_returns = returns_mat.mean() * 252

        # Random portfolios
        np.random.seed(42)
        n_ports = 3000
        port_returns = []
        port_vols    = []
        port_sharpes = []
        all_weights  = []
        n_assets     = len(TICKERS)
        for _ in range(n_ports):
            w = np.random.dirichlet(np.ones(n_assets))
            all_weights.append(w)
            pr = np.dot(w, mean_returns.values)
            pv = np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))
            port_returns.append(pr)
            port_vols.append(pv)
            port_sharpes.append(pr / pv if pv > 0 else 0)

        port_returns  = np.array(port_returns)
        port_vols     = np.array(port_vols)
        port_sharpes  = np.array(port_sharpes)
        best_idx      = np.argmax(port_sharpes)
        best_weights  = all_weights[best_idx]

        # Efficient frontier chart
        fig, ax = dark_fig((10, 5))
        sc = ax.scatter(port_vols, port_returns, c=port_sharpes, cmap="plasma", alpha=0.5, s=6)
        ax.scatter(port_vols[best_idx], port_returns[best_idx], marker="*", color=COLOR_1, s=300, zorder=5, label="Max Sharpe")
        plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
        ax.set_title("Efficient Frontier — Random Portfolios", fontsize=11)
        ax.set_xlabel("Annualised Volatility"); ax.set_ylabel("Annualised Return")
        ax.legend(fontsize=9, facecolor="#0f1628", edgecolor="#1a3050", labelcolor="#a0c4e0")
        ax.grid(True, alpha=0.08)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Optimal weights bar chart
        fig2, ax2 = dark_fig((8, 3))
        bars = ax2.bar(TICKERS, best_weights * 100, color=[colors_list[i % len(colors_list)] for i in range(len(TICKERS))], edgecolor="#0a0e1a")
        ax2.set_title("Optimal Portfolio Weights (Max Sharpe)", fontsize=11)
        ax2.set_ylabel("Weight (%)")
        for bar, val in zip(bars, best_weights):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val*100:.1f}%", ha="center", va="bottom", color="#a0c4e0", fontsize=9)
        ax2.grid(True, alpha=0.1, axis="y")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # Correlation heatmap
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        corr = returns_mat.corr()
        fig3, ax3 = dark_fig((7, 5))
        im = ax3.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
        ax3.set_xticks(range(len(corr.columns))); ax3.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9, color="#a0c4e0")
        ax3.set_yticks(range(len(corr.index)));   ax3.set_yticklabels(corr.index, fontsize=9, color="#a0c4e0")
        plt.colorbar(im, ax=ax3)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax3.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white")
        ax3.set_title("Return Correlation Heatmap", fontsize=11)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("Select **2 or more tickers** in the sidebar to see portfolio optimization.")

    # TFT model info
    st.markdown('<div class="section-header">Temporal Fusion Transformer — Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0f1e36;border:1px solid #1a3050;border-radius:10px;padding:16px;font-family:'Space Mono',monospace;font-size:12px;color:#5a8fbf;line-height:1.8;">
    <b style="color:#00e5a0">TFT Model Architecture</b><br>
    ├─ Encoder Length: <span style="color:#ffd166">30 timesteps</span><br>
    ├─ Prediction Length: <span style="color:#ffd166">1 timestep</span><br>
    ├─ Input Features: RSI, MACD, Volatility_20, Rolling_Mean_20, market_regime<br>
    ├─ Group IDs: Ticker (multi-asset)<br>
    ├─ Loss: QuantileLoss<br>
    ├─ Optimizer: Adam  (lr=0.03)<br>
    └─ Training: 5 epochs (extensible)<br><br>
    <b style="color:#00e5a0">Saved Model</b>: <span style="color:#4a9eff">models/tft_stock_model.ckpt</span><br>
    <b style="color:#00e5a0">Predictions</b>: <span style="color:#4a9eff">results/tft_predictions.csv</span>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 6 — SENTIMENT ANALYSIS
# ════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">FinBERT Financial Sentiment Analysis</div>', unsafe_allow_html=True)
    st.info("ℹ️ FinBERT (ProsusAI/finbert) requires a Hugging Face connection. Results below use a rule-based approximation for fast demo. Enable the real model in `app.py` if deploying with GPU/HF access.", icon="💡")

    headlines = [h.strip() for h in NEWS_TEXT.strip().split("\n") if h.strip()]

    # Rule-based sentiment fallback (fast, no model download)
    POSITIVE_WORDS = {"strong","record","growth","boost","rise","gain","profit","up","bull","innovation","beat","exceed","surpass","high","positive","increase","expand"}
    NEGATIVE_WORDS = {"fall","drop","loss","inflation","pressure","fear","crash","down","bear","recession","decline","cut","risk","weak","negative","decrease","concern","raise"}

    def rule_sentiment(text):
        words = text.lower().split()
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        if pos > neg:  return "positive", pos/(pos+neg+1)
        if neg > pos:  return "negative", neg/(pos+neg+1)
        return "neutral", 0.5

    results = [rule_sentiment(h) for h in headlines]

    if headlines:
        for headline, (label, score) in zip(headlines, results):
            badge_class = {"positive": "signal-buy", "negative": "signal-sell"}.get(label, "signal-hold")
            score_pct   = score * 100
            bar_color   = {"positive": COLOR_1, "negative": COLOR_3}.get(label, COLOR_4)
            st.markdown(f"""
            <div style="background:#0f1e36;border:1px solid #1a3050;border-radius:10px;padding:14px 16px;margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="color:#c8d8e8;font-size:13px;">"{headline}"</span>
                <span class="{badge_class}">{label.upper()}</span>
              </div>
              <div style="background:#0a0e1a;border-radius:4px;height:6px;overflow:hidden;">
                <div style="width:{score_pct:.0f}%;height:100%;background:{bar_color};border-radius:4px;transition:width 0.4s;"></div>
              </div>
              <span style="font-size:11px;color:#3a6a9a;">Confidence {score_pct:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Aggregate score
        label_to_num = {"positive": 1, "negative": -1, "neutral": 0}
        agg = np.mean([label_to_num[r[0]] for r in results])
        agg_label = "BULLISH 📈" if agg > 0.1 else "BEARISH 📉" if agg < -0.1 else "NEUTRAL ➡️"
        agg_color = COLOR_1 if agg > 0.1 else COLOR_3 if agg < -0.1 else COLOR_4
        st.markdown(f"""
        <div style="background:#0f1e36;border:2px solid {agg_color};border-radius:12px;padding:16px;text-align:center;margin-top:10px;">
          <div style="font-family:'Space Mono',monospace;font-size:11px;color:#3a6a9a;letter-spacing:2px;margin-bottom:6px;">AGGREGATE SENTIMENT</div>
          <div style="font-family:'Space Mono',monospace;font-size:24px;font-weight:700;color:{agg_color};">{agg_label}</div>
          <div style="font-size:12px;color:#4a6a8a;margin-top:4px;">Score: {agg:+.2f} (range −1 to +1)</div>
        </div>
        """, unsafe_allow_html=True)

        # Sentiment distribution pie
        from collections import Counter
        cnt = Counter(r[0] for r in results)
        labels_pie = list(cnt.keys())
        sizes_pie  = list(cnt.values())
        palette = {"positive": COLOR_1, "negative": COLOR_3, "neutral": COLOR_4}
        pie_colors = [palette.get(l, COLOR_2) for l in labels_pie]
        fig, ax = dark_fig((5, 4))
        wedges, texts, autotexts = ax.pie(sizes_pie, labels=labels_pie, colors=pie_colors,
                                           autopct="%1.0f%%", startangle=90,
                                           textprops={"color":"#a0c4e0","fontsize":10},
                                           pctdistance=0.75)
        for at in autotexts: at.set_color("#0a0e1a"); at.set_fontweight("bold")
        ax.set_title("Sentiment Distribution", color="#a0c4e0", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#2a4a6a;font-size:11px;font-family:'Space Mono',monospace;padding:8px 0;">
  AI Financial Decision System · TFT · FinBERT · RL (PPO) · GNN · Monte Carlo · Streamlit
</div>
""", unsafe_allow_html=True)
