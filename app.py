"""
台指期貨 & 台灣加權指數 費氏數列分析看板
Fibonacci Analysis Dashboard for Taiwan Futures & TAIEX
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📊 台指期 費氏數列看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: "Microsoft JhengHei", "PingFang TC", sans-serif; }
  #MainMenu, footer { visibility: hidden; }
  .card {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    border-left: 4px solid #00d4aa;
  }
  .card-red  { border-left-color: #ef5350; }
  .card-green{ border-left-color: #26a69a; }
  .card-gold { border-left-color: #ffd700; }
  .card-blue { border-left-color: #2196f3; }
  .card h4   { margin: 0 0 4px 0; font-size: 13px; color: #aaa; }
  .card .val { font-size: 22px; font-weight: bold; color: #fff; }
  .card .sub { font-size: 12px; color: #888; margin-top: 4px; }
  .bull { color: #26a69a; }
  .bear { color: #ef5350; }
  .neut { color: #ffd700; }
  .fib-table td, .fib-table th { padding: 5px 10px; font-size: 13px; }
  .fib-table th { color: #aaa; }
  .fib-highlight { background: rgba(255,215,0,0.15); font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TW_TZ = pytz.timezone("Asia/Taipei")

TICKERS = {
    "台灣加權指數  ^TWII":  "^TWII",
    "元大台灣50   0050": "0050.TW",
    "台灣中型100  0051": "0051.TW",
    "台積電       2330": "2330.TW",
}

FIB_RET = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
FIB_EXT = [1.272, 1.414, 1.618, 2.000, 2.618]

FIB_COLORS = {
    0.0:   "#FF4444",
    0.236: "#FF8C00",
    0.382: "#FFD700",
    0.500: "#00DD88",
    0.618: "#2196F3",
    0.786: "#9C27B0",
    1.0:   "#FF4444",
    1.272: "#FF69B4",
    1.414: "#00BCD4",
    1.618: "#4CAF50",
    2.000: "#FF5722",
    2.618: "#673AB7",
}

FIB_LABELS = {
    0.0:   "0.000  起點",
    0.236: "0.236",
    0.382: "0.382  ★",
    0.500: "0.500  中線",
    0.618: "0.618  ★",
    0.786: "0.786",
    1.0:   "1.000  終點",
    1.272: "1.272  延伸",
    1.414: "1.414  延伸",
    1.618: "1.618  延伸★",
    2.000: "2.000  延伸",
    2.618: "2.618  延伸",
}

INTERVAL_PERIODS = {
    "1m":  ["1d", "5d"],
    "5m":  ["1d", "5d", "1mo"],
    "15m": ["5d", "1mo"],
    "30m": ["1mo", "3mo"],
    "1h":  ["1mo", "3mo", "6mo"],
    "1d":  ["3mo", "6mo", "1y", "2y", "5y"],
    "1wk": ["6mo", "1y", "2y", "5y"],
    "1mo": ["1y", "2y", "5y"],
}

PERIOD_LABELS = {
    "1d": "1天", "5d": "5天", "1mo": "1個月", "3mo": "3個月",
    "6mo": "6個月", "1y": "1年", "2y": "2年", "5y": "5年",
}

INTERVAL_LABELS = {
    "1m": "1分鐘", "5m": "5分鐘", "15m": "15分鐘", "30m": "30分鐘",
    "1h": "1小時", "1d": "日線", "1wk": "週線", "1mo": "月線",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=180, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            interval=interval,
            period=period,
            auto_adjust=True,
            progress=False,
            actions=False,
        )
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as exc:
        st.error(f"資料載入失敗 ({ticker}): {exc}")
        return pd.DataFrame()


def taiwan_market_open() -> bool:
    now = datetime.now(TW_TZ)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    market_close = now.replace(hour=13, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ─────────────────────────────────────────────────────────────────────────────
# Fibonacci core
# ─────────────────────────────────────────────────────────────────────────────

def detect_swings(df: pd.DataFrame, window: int = 10):
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(window, n - window):
        slice_hi = df["High"].iloc[i - window: i + window + 1]
        slice_lo = df["Low"].iloc[i - window: i + window + 1]
        if df["High"].iloc[i] == slice_hi.max():
            swing_highs.append(i)
        if df["Low"].iloc[i] == slice_lo.min():
            swing_lows.append(i)
    return swing_highs, swing_lows


def dominant_swing(df: pd.DataFrame, window: int = 10):
    hi_idx, lo_idx = detect_swings(df, window)
    if not hi_idx:
        hi_idx = [int(df["High"].values.argmax())]
    if not lo_idx:
        lo_idx = [int(df["Low"].values.argmin())]
    cutoff = max(0, len(df) - int(len(df) * 0.40))

    def best_high(idxs):
        recent = [i for i in idxs if i >= cutoff] or idxs
        return max(recent, key=lambda i: df["High"].iloc[i])

    def best_low(idxs):
        recent = [i for i in idxs if i >= cutoff] or idxs
        return min(recent, key=lambda i: df["Low"].iloc[i])

    h_i = best_high(hi_idx)
    l_i = best_low(lo_idx)
    return df["High"].iloc[h_i], h_i, df["Low"].iloc[l_i], l_i


def fib_retracement(high: float, low: float) -> dict:
    diff = high - low
    return {lvl: high - diff * lvl for lvl in FIB_RET}


def fib_extension(high: float, low: float, trend: str = "up") -> dict:
    diff = high - low
    if trend == "up":
        return {lvl: high + diff * (lvl - 1.0) for lvl in FIB_EXT}
    else:
        return {lvl: low - diff * (lvl - 1.0) for lvl in FIB_EXT}


def current_fib_position(price: float, levels: dict):
    sorted_lvl = sorted(levels.items(), key=lambda x: x[1])
    below = above = None
    for lvl, p in sorted_lvl:
        if p <= price:
            below = (lvl, p)
        elif above is None:
            above = (lvl, p)
    if below is not None and above is not None:
        span = above[1] - below[1]
        pct  = (price - below[1]) / span if span else 0.5
        return below[0], below[1], above[0], above[1], pct
    if below is not None:
        return below[0], below[1], None, None, 1.0
    if above is not None:
        return None, None, above[0], above[1], 0.0
    return None, None, None, None, 0.5


def trend_from_swings(hi_idx: int, lo_idx: int) -> str:
    return "up" if lo_idx > hi_idx else "down"


# ─────────────────────────────────────────────────────────────────────────────
# Bull / Bear signal
# ─────────────────────────────────────────────────────────────────────────────

def bull_bear_signal(price: float, ret_levels: dict, trend: str) -> dict:
    h    = ret_levels[0.0]
    l    = ret_levels[1.0]
    r382 = ret_levels[0.382]
    r500 = ret_levels[0.500]
    r618 = ret_levels[0.618]

    score = 50
    tags  = []

    if trend == "up":
        score += 10; tags.append("上升趨勢")
    else:
        score -= 10; tags.append("下降趨勢")

    if price > r382:
        score += 15; tags.append("站上0.382")
    elif price < r618:
        score -= 15; tags.append("跌破0.618")

    if price > r500:
        score += 10; tags.append("站上0.5中線")
    else:
        score -= 5

    if price > h * 0.99:
        score += 15; tags.append("逼近高點")
    if price < l * 1.01:
        score -= 15; tags.append("逼近低點")

    score = max(0, min(100, score))

    if score >= 70:   label, css = "強勢多頭 ▲▲", "bull"
    elif score >= 55: label, css = "偏多 ▲",       "bull"
    elif score >= 45: label, css = "中性觀望 →",   "neut"
    elif score >= 30: label, css = "偏空 ▼",       "bear"
    else:             label, css = "強勢空頭 ▼▼",  "bear"

    return {"label": label, "css": css, "score": score, "tags": tags}


# ─────────────────────────────────────────────────────────────────────────────
# Tomorrow target
# ─────────────────────────────────────────────────────────────────────────────

def tomorrow_targets(price: float, ret_levels: dict, ext_levels: dict, trend: str) -> dict:
    all_levels = {**ret_levels, **ext_levels}
    sorted_asc = sorted(all_levels.items(), key=lambda x: x[1])

    resistance_list = [(lv, p) for lv, p in sorted_asc if p > price]
    support_list    = [(lv, p) for lv, p in sorted_asc if p < price]

    r1 = resistance_list[0] if resistance_list else None
    s1 = support_list[-1]   if support_list    else None

    if r1 and s1:
        up_tgt, dn_tgt = r1[1], s1[1]
        up_lbl = FIB_LABELS.get(r1[0], f"{r1[0]:.3f}")
        dn_lbl = FIB_LABELS.get(s1[0], f"{s1[0]:.3f}")
        bias   = "偏多，關注壓力" if trend == "up" else "偏空，關注支撐"
        desc   = f"趨勢{bias}｜上方目標 {up_tgt:,.0f}（{up_lbl}）｜下方支撐 {dn_tgt:,.0f}（{dn_lbl}）"
    elif r1:
        up_tgt, dn_tgt = r1[1], price * 0.99
        desc = f"上方目標 {r1[1]:,.0f}（{FIB_LABELS.get(r1[0], '')}）"
    elif s1:
        up_tgt, dn_tgt = price * 1.01, s1[1]
        desc = f"下方支撐 {s1[1]:,.0f}（{FIB_LABELS.get(s1[0], '')}）"
    else:
        up_tgt, dn_tgt = price * 1.01, price * 0.99
        desc = "無明確費氏目標"

    return {"up_target": up_tgt, "dn_target": dn_tgt, "pivot": price, "description": desc}


# ─────────────────────────────────────────────────────────────────────────────
# Chart builder
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(df, name, ret_levels, ext_levels,
                swing_hi, swing_hi_idx, swing_lo, swing_lo_idx,
                trend, show_ext, tmr):

    current = float(df["Close"].iloc[-1])
    x_axis  = df.index.tolist()
    x0, x1  = x_axis[0], x_axis[-1]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.20, 0.15], vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=x_axis, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="K線",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        line_width=1,
    ), row=1, col=1)

    for lvl, price in ret_levels.items():
        color = FIB_COLORS.get(lvl, "#888")
        lw    = 2 if lvl in (0.382, 0.618) else 1
        dash  = "solid" if lvl in (0.382, 0.618) else "dash"
        fig.add_shape(type="line", x0=x0, x1=x1, y0=price, y1=price,
                      line=dict(color=color, width=lw, dash=dash), row=1, col=1)
        fig.add_annotation(x=x1, y=price, xanchor="left", showarrow=False,
                           text=f"  {FIB_LABELS.get(lvl, f'{lvl:.3f}')} : {price:,.0f}",
                           font=dict(size=10, color=color), row=1, col=1)

    if show_ext:
        for lvl, price in ext_levels.items():
            color = FIB_COLORS.get(lvl, "#888")
            fig.add_shape(type="line", x0=x0, x1=x1, y0=price, y1=price,
                          line=dict(color=color, width=1, dash="dot"), row=1, col=1)
            fig.add_annotation(x=x1, y=price, xanchor="left", showarrow=False,
                               text=f"  {FIB_LABELS.get(lvl, f'{lvl:.3f}')} : {price:,.0f}",
                               font=dict(size=9, color=color), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[x_axis[swing_hi_idx]], y=[swing_hi], mode="markers+text",
        marker=dict(symbol="triangle-down", size=14, color="#ef5350"),
        text=["▼波段高"], textposition="top center",
        textfont=dict(size=11, color="#ef5350"), showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[x_axis[swing_lo_idx]], y=[swing_lo], mode="markers+text",
        marker=dict(symbol="triangle-up", size=14, color="#26a69a"),
        text=["▲波段低"], textposition="bottom center",
        textfont=dict(size=11, color="#26a69a"), showlegend=False,
    ), row=1, col=1)

    fig.add_shape(type="line", x0=x0, x1=x1, y0=current, y1=current,
                  line=dict(color="white", width=1.5, dash="dot"), row=1, col=1)
    fig.add_annotation(x=x0, y=current, text=f"  ▶ 現價 {current:,.0f}",
                       xanchor="right", showarrow=False,
                       font=dict(size=11, color="white"), row=1, col=1)

    for tgt_price, tgt_color, tgt_lbl in [
        (tmr["up_target"], "#26a69a", f"↑明日目標 {tmr['up_target']:,.0f}"),
        (tmr["dn_target"], "#ef5350", f"↓明日支撐 {tmr['dn_target']:,.0f}"),
    ]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=tgt_price, y1=tgt_price,
                      line=dict(color=tgt_color, width=2, dash="dashdot"), row=1, col=1)
        fig.add_annotation(x=x0, y=tgt_price, text=f"  {tgt_lbl}",
                           xanchor="right", showarrow=False,
                           font=dict(size=10, color=tgt_color), row=1, col=1)

    vol_colors = [
        "#26a69a" if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i])
        else "#ef5350" for i in range(len(df))
    ]
    fig.add_trace(go.Bar(x=x_axis, y=df["Volume"].tolist(),
                         name="成交量", marker_color=vol_colors, opacity=0.8), row=2, col=1)

    diff = swing_hi - swing_lo if swing_hi != swing_lo else 1.0
    fib_osc = [max(0.0, min(1.0, (swing_hi - float(p)) / diff)) for p in df["Close"]]
    osc_colors = [
        "#ef5350" if v > 0.618 else ("#ffd700" if v > 0.382 else "#26a69a")
        for v in fib_osc
    ]
    fig.add_trace(go.Bar(x=x_axis, y=fib_osc, name="費氏位階",
                         marker_color=osc_colors, opacity=0.9), row=3, col=1)
    for ref in [0.382, 0.618]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=ref, y1=ref,
                      line=dict(color="white", width=1, dash="dash"), row=3, col=1)

    trend_arrow = "↑ 多頭" if trend == "up" else "↓ 空頭"
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=820, showlegend=False, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=160, t=55, b=10), font=dict(size=11, color="#ccc"),
        title=dict(
            text=(f"<b>{name}</b>　｜　費氏數列分析　｜　"
                  f"現價 <b>{current:,.0f}</b>　｜　趨勢 <b>{trend_arrow}</b>"),
            x=0.02, font=dict(size=15, color="white"),
        ),
    )
    for row in [1, 2, 3]:
        fig.update_xaxes(gridcolor="#1e2130", zeroline=False,
                         showticklabels=(row == 3), row=row, col=1)
        fig.update_yaxes(gridcolor="#1e2130", zeroline=False, row=row, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar() -> dict:
    st.sidebar.markdown("## 📊 台指期 費氏看板")
    st.sidebar.markdown("---")

    ticker_label = st.sidebar.selectbox("商品選擇", list(TICKERS.keys()), index=0)
    ticker = TICKERS[ticker_label]
    st.sidebar.markdown("---")

    interval_label = st.sidebar.selectbox("時間週期", list(INTERVAL_LABELS.values()), index=5)
    interval = [k for k, v in INTERVAL_LABELS.items() if v == interval_label][0]

    valid_period_keys   = INTERVAL_PERIODS.get(interval, ["1y"])
    valid_period_labels = [PERIOD_LABELS[k] for k in valid_period_keys]
    period_label = st.sidebar.selectbox(
        "資料期間", valid_period_labels, index=min(2, len(valid_period_labels) - 1)
    )
    period = [k for k, v in PERIOD_LABELS.items() if v == period_label][0]
    st.sidebar.markdown("---")

    swing_window = st.sidebar.slider(
        "波段偵測靈敏度（K棒數）", min_value=3, max_value=30, value=10, step=1,
        help="數值越大偵測到的波段越大，越小波段越精細"
    )
    show_ext     = st.sidebar.checkbox("顯示費氏延伸線", value=True)
    st.sidebar.markdown("---")

    auto_refresh = st.sidebar.checkbox("自動更新（3分鐘）", value=False)
    if auto_refresh:
        st.sidebar.info("每3分鐘自動刷新資料")

    if st.sidebar.button("🔄 立即更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    status = "🟢 開盤中" if taiwan_market_open() else "🔴 收盤中"
    now_tw = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
    st.sidebar.markdown(f"**市場狀態**: {status}")
    st.sidebar.markdown(f"台灣時間: `{now_tw}`")

    return {
        "ticker": ticker, "ticker_label": ticker_label.split()[0],
        "interval": interval, "period": period,
        "swing_window": swing_window, "show_ext": show_ext, "auto_refresh": auto_refresh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def card(title: str, value: str, sub: str = "", css_class: str = "") -> str:
    return (f'<div class="card {css_class}"><h4>{title}</h4>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def fib_table_html(ret_levels: dict, current_price: float) -> str:
    rows = ""
    for lvl, price in sorted(ret_levels.items(), key=lambda x: x[1], reverse=True):
        diff  = current_price - price
        pct   = diff / price * 100
        arrow = "▲" if diff >= 0 else "▼"
        color = "#26a69a" if diff >= 0 else "#ef5350"
        hl    = 'class="fib-highlight"' if lvl in (0.382, 0.618) else ""
        rows += (f'<tr {hl}>'
                 f'<td>{FIB_LABELS.get(lvl, f"{lvl:.3f}")}</td>'
                 f'<td style="color:{FIB_COLORS.get(lvl, "#888")}"><b>{price:,.0f}</b></td>'
                 f'<td style="color:{color}">{arrow} {abs(diff):,.0f} ({abs(pct):.2f}%)</td>'
                 f'</tr>')
    return (f'<table class="fib-table" style="width:100%;border-collapse:collapse">'
            f'<thead><tr><th>費氏層級</th><th>價格</th><th>距現價</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = sidebar()

    if cfg["auto_refresh"]:
        st.markdown('<meta http-equiv="refresh" content="180">', unsafe_allow_html=True)

    with st.spinner(f"載入 {cfg['ticker_label']} 資料中…"):
        df = fetch_ohlcv(cfg["ticker"], cfg["interval"], cfg["period"])

    if df.empty:
        st.error("無法取得資料，請確認商品代碼或稍後再試。")
        st.stop()

    swing_hi, sh_idx, swing_lo, sl_idx = dominant_swing(df, cfg["swing_window"])
    trend = trend_from_swings(sh_idx, sl_idx)
    ret   = fib_retracement(swing_hi, swing_lo)
    ext   = fib_extension(swing_hi, swing_lo, trend)

    current_price = float(df["Close"].iloc[-1])
    prev_price    = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
    change        = current_price - prev_price
    change_pct    = change / prev_price * 100 if prev_price else 0

    signal = bull_bear_signal(current_price, ret, trend)
    tmr    = tomorrow_targets(current_price, ret, ext, trend)
    lb, lb_p, ab, ab_p, pct_in = current_fib_position(current_price, ret)

    # ── Header metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(card("現價", f"{current_price:,.0f}",
                         f"{'▲' if change>=0 else '▼'} {change:+,.0f} ({change_pct:+.2f}%)",
                         "card-green" if change >= 0 else "card-red"), unsafe_allow_html=True)
    with col2:
        st.markdown(card("波段高點", f"{swing_hi:,.0f}", "起算點（Fib 0.0）", "card-red"),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(card("波段低點", f"{swing_lo:,.0f}", "起算點（Fib 1.0）", "card-green"),
                    unsafe_allow_html=True)
    with col4:
        sig_css = "card-green" if signal["css"] == "bull" else (
            "card-red" if signal["css"] == "bear" else "card-gold")
        st.markdown(card("多空訊號", signal["label"], f"強度指數 {signal['score']}/100", sig_css),
                    unsafe_allow_html=True)
    with col5:
        t_css = "card-green" if trend == "up" else "card-red"
        st.markdown(card("趨勢方向", "↑ 多頭" if trend == "up" else "↓ 空頭",
                         f"波段漲幅 {abs(swing_hi-swing_lo):,.0f} "
                         f"({abs(swing_hi-swing_lo)/swing_lo*100:.1f}%)", t_css),
                    unsafe_allow_html=True)

    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 K線 + 費氏分析", "🎯 明日目標價", "📋 費氏層級表", "📊 多時框概覽",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1
    # ════════════════════════════════════════════════════════════════
    with tab1:
        fig = build_chart(df, cfg["ticker_label"], ret, ext,
                          swing_hi, sh_idx, swing_lo, sl_idx,
                          trend, cfg["show_ext"], tmr)
        st.plotly_chart(fig, use_container_width=True)

        # 用 is not None，避免 lb=0.0 被誤判為 False
        if lb is not None and ab is not None:
            fib_pos_txt = (
                f"現價 **{current_price:,.0f}** 位於費氏 "
                f"**{lb:.3f}** ({lb_p:,.0f}) ↔ **{ab:.3f}** ({ab_p:,.0f}) 之間，"
                f"位置 {pct_in*100:.1f}%"
            )
        elif lb is not None:
            fib_pos_txt = f"現價 **{current_price:,.0f}** 已站上所有費氏回檔位，費氏 {lb:.3f} 為支撐"
        elif ab is not None:
            fib_pos_txt = f"現價 **{current_price:,.0f}** 低於所有費氏回檔位，費氏 {ab:.3f} 為壓力"
        else:
            fib_pos_txt = f"現價 **{current_price:,.0f}** 費氏位置計算中…"

        st.info(f"📍 **費氏位置**: {fib_pos_txt}")
        signal_html = (
            f'<span class="{signal["css"]}" style="font-size:18px;font-weight:bold">'
            f'{signal["label"]}</span>'
            f'　<span style="color:#aaa;font-size:14px">{"　".join(signal["tags"])}</span>'
        )
        st.markdown(f"🚦 **多空判斷**: {signal_html}", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2
    # ════════════════════════════════════════════════════════════════
    with tab2:
        now_tw   = datetime.now(TW_TZ)
        next_day = now_tw + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        st.markdown(f"### 🎯 明日目標價　`{next_day.strftime('%Y-%m-%d (%A)')}`")
        st.markdown(f"> {tmr['description']}")
        st.markdown("")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(card("⬆ 上方目標（壓力）", f"{tmr['up_target']:,.0f}",
                             f"距現價 +{tmr['up_target']-current_price:,.0f} "
                             f"(+{(tmr['up_target']-current_price)/current_price*100:.2f}%)",
                             "card-red"), unsafe_allow_html=True)
        with c2:
            st.markdown(card("⏺ 今日收盤（樞軸）", f"{current_price:,.0f}",
                             f"波段位置 {(swing_hi-current_price)/(swing_hi-swing_lo)*100:.1f}% 回檔",
                             "card-gold"), unsafe_allow_html=True)
        with c3:
            st.markdown(card("⬇ 下方支撐", f"{tmr['dn_target']:,.0f}",
                             f"距現價 {tmr['dn_target']-current_price:,.0f} "
                             f"({(tmr['dn_target']-current_price)/current_price*100:.2f}%)",
                             "card-green"), unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### 📌 關鍵費氏支撐壓力一覽")
        all_fib = {**ret}
        if cfg["show_ext"]:
            all_fib.update(ext)
        sorted_f  = sorted(all_fib.items(), key=lambda x: x[1])
        prices_f  = [p for _, p in sorted_f]
        labels_f  = [FIB_LABELS.get(lv, f"{lv:.3f}") for lv, _ in sorted_f]
        fig2 = go.Figure(go.Bar(
            x=labels_f,
            y=[abs(p - current_price) for p in prices_f],
            marker_color=["#ef5350" if p > current_price else "#26a69a" for p in prices_f],
            text=[f"{p:,.0f}" for p in prices_f],
            textposition="outside",
        ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            title="各費氏層級與現價的距離（綠=支撐 / 紅=壓力）",
            showlegend=False, font=dict(size=11), xaxis_tickangle=-30,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 📅 近期收盤與費氏位階追蹤")
        hist = df.tail(20).copy()
        diff_total = swing_hi - swing_lo if swing_hi != swing_lo else 1.0
        hist["費氏回檔%"] = ((swing_hi - hist["Close"]) / diff_total * 100).round(2)
        hist["費氏位階"] = hist["費氏回檔%"].apply(lambda x:
            "0.000" if x <= 0 else "0.236" if x <= 23.6 else "0.382" if x <= 38.2 else
            "0.500" if x <= 50.0 else "0.618" if x <= 61.8 else "0.786" if x <= 78.6 else "1.000"
        )
        display_hist = hist[["Close", "High", "Low", "Volume", "費氏回檔%", "費氏位階"]].copy()
        if hasattr(display_hist.index, "strftime"):
            display_hist.index = display_hist.index.strftime("%Y-%m-%d %H:%M")
        display_hist.columns = ["收盤", "最高", "最低", "成交量", "費氏回檔%", "費氏位階"]
        st.dataframe(display_hist.style.background_gradient(subset=["費氏回檔%"], cmap="RdYlGn_r"),
                     use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 📋 費氏回檔層級詳表")
        st.markdown(fib_table_html(ret, current_price), unsafe_allow_html=True)
        if cfg["show_ext"]:
            st.markdown("### 📋 費氏延伸層級詳表")
            st.markdown(fib_table_html(ext, current_price), unsafe_allow_html=True)
        st.markdown("#### 費氏數列說明")
        st.markdown("""
| 層級 | 意義 |
|------|------|
| **0.382** | 最常見的回檔支撐/壓力，主力第一守備區 |
| **0.500** | 心理中線，多空分水嶺 |
| **0.618** | 黃金比例，突破或守住代表趨勢確立 |
| **1.618** | 延伸目標，波段完成位 |
        """)

    # ════════════════════════════════════════════════════════════════
    # TAB 4
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📊 多時間框架費氏位階概覽")
        st.caption("各時框自動抓取近期波段，計算現價費氏回檔位置（需時稍長）")

        mtf_cfg = {
            "日線 (3個月)": ("1d",  "3mo"),
            "日線 (1年)":   ("1d",  "1y"),
            "週線 (2年)":   ("1wk", "2y"),
            "月線 (5年)":   ("1mo", "5y"),
        }

        mtf_results, scores, labels_mtf = [], [], []
        prog = st.progress(0)
        for idx, (lbl, (iv, pr)) in enumerate(mtf_cfg.items()):
            prog.progress((idx + 1) / len(mtf_cfg), text=f"載入 {lbl}…")
            df_m = fetch_ohlcv(cfg["ticker"], iv, pr)
            if df_m.empty:
                continue
            sh_m, shi_m, sl_m, sli_m = dominant_swing(df_m, cfg["swing_window"])
            tr_m   = trend_from_swings(shi_m, sli_m)
            ret_m  = fib_retracement(sh_m, sl_m)
            ext_m  = fib_extension(sh_m, sl_m, tr_m)
            cp_m   = float(df_m["Close"].iloc[-1])
            sig_m  = bull_bear_signal(cp_m, ret_m, tr_m)
            tmr_m  = tomorrow_targets(cp_m, ret_m, ext_m, tr_m)
            ret_pct = (sh_m - cp_m) / (sh_m - sl_m) * 100 if sh_m != sl_m else 0
            mtf_results.append({
                "時間框架": lbl, "現價": f"{cp_m:,.0f}",
                "波段高": f"{sh_m:,.0f}", "波段低": f"{sl_m:,.0f}",
                "費氏回檔%": f"{ret_pct:.1f}%",
                "趨勢": "↑多" if tr_m == "up" else "↓空",
                "多空訊號": sig_m["label"],
                "明日目標↑": f"{tmr_m['up_target']:,.0f}",
                "明日支撐↓": f"{tmr_m['dn_target']:,.0f}",
            })
            labels_mtf.append(lbl)
            scores.append(sig_m["score"])
        prog.empty()

        if mtf_results:
            st.dataframe(pd.DataFrame(mtf_results).set_index("時間框架"),
                         use_container_width=True)
            fig3 = go.Figure(go.Bar(
                x=labels_mtf, y=scores,
                marker_color=["#26a69a" if s >= 60 else ("#ffd700" if s >= 45 else "#ef5350")
                              for s in scores],
                text=[f"{s}/100" for s in scores], textposition="outside",
            ))
            fig3.add_hline(y=60, line_color="#26a69a", line_dash="dash", annotation_text="多頭區")
            fig3.add_hline(y=40, line_color="#ef5350", line_dash="dash", annotation_text="空頭區")
            fig3.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                height=300, margin=dict(l=10, r=10, t=30, b=10),
                title="多時框多空強度評分", showlegend=False, yaxis=dict(range=[0, 110]),
            )
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#555;font-size:12px">'
        '資料來源：Yahoo Finance ｜ 費氏數列分析看板 ｜ 僅供參考，不構成投資建議'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
