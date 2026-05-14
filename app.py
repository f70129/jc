"""
台指期貨 & 台灣加權指數 費氏數列 × 移動平均 分析看板
Fibonacci + Moving Average Dashboard for Taiwan Futures & TAIEX
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
    page_title="📊 台指期 費氏×均線看板",
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
  .card-red   { border-left-color: #ef5350; }
  .card-green { border-left-color: #26a69a; }
  .card-gold  { border-left-color: #ffd700; }
  .card-blue  { border-left-color: #2196f3; }
  .card-purple{ border-left-color: #9c27b0; }
  .card h4    { margin: 0 0 4px 0; font-size: 13px; color: #aaa; }
  .card .val  { font-size: 22px; font-weight: bold; color: #fff; }
  .card .sub  { font-size: 12px; color: #888; margin-top: 4px; }
  .bull { color: #26a69a; }
  .bear { color: #ef5350; }
  .neut { color: #ffd700; }
  .fib-table td, .fib-table th { padding: 5px 10px; font-size: 13px; }
  .fib-table th { color: #aaa; }
  .fib-highlight { background: rgba(255,215,0,0.15); font-weight: bold; }
  .conf-strong { background: rgba(255,215,0,0.20); }
  .suggest-box {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
  }
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

MA_COLORS = {
    5:   "#FF9800",
    10:  "#FF5722",
    20:  "#2196F3",
    60:  "#9C27B0",
    120: "#00BCD4",
    240: "#F44336",
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

def detect_swings(df, window=10):
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


def dominant_swing(df, window=10):
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


def current_fib_position(price: float, levels: dict) -> tuple:
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
    return None, None, above[0], above[1], 0.0


def trend_from_swings(hi_idx: int, lo_idx: int) -> str:
    return "up" if lo_idx > hi_idx else "down"


# ─────────────────────────────────────────────────────────────────────────────
# Moving Average functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_mas(df: pd.DataFrame, periods: list) -> dict:
    mas = {}
    for p in periods:
        if len(df) >= p:
            mas[p] = df["Close"].rolling(window=p).mean()
        else:
            mas[p] = df["Close"].rolling(window=len(df)).mean()
    return mas


def ma_trend_signal(mas: dict, current_price: float) -> dict:
    if not mas:
        return {"score": 50, "label": "均線無資料", "css": "neut",
                "above": [], "below": [], "ordered": False}

    sorted_periods = sorted(mas.keys())
    ma_vals = {}
    for p in sorted_periods:
        last = mas[p].iloc[-1]
        if pd.notna(last):
            ma_vals[p] = float(last)

    if not ma_vals:
        return {"score": 50, "label": "均線計算中", "css": "neut",
                "above": [], "below": [], "ordered": False}

    above = [p for p, v in ma_vals.items() if current_price > v]
    below = [p for p, v in ma_vals.items() if current_price <= v]

    score = 50
    pct_above = len(above) / len(ma_vals)
    score += int((pct_above - 0.5) * 40)

    periods_list = sorted(ma_vals.keys())
    ordered_bull = all(
        ma_vals[periods_list[i]] >= ma_vals[periods_list[i+1]]
        for i in range(len(periods_list)-1)
    )
    ordered_bear = all(
        ma_vals[periods_list[i]] <= ma_vals[periods_list[i+1]]
        for i in range(len(periods_list)-1)
    )
    if ordered_bull:
        score += 20
    elif ordered_bear:
        score -= 20

    longest = max(ma_vals.keys())
    dist_pct = (current_price - ma_vals[longest]) / ma_vals[longest] * 100
    if dist_pct > 5:
        score += 10
    elif dist_pct < -5:
        score -= 10

    score = max(0, min(100, score))

    if score >= 70:
        label, css = "均線多頭排列 ▲▲", "bull"
    elif score >= 55:
        label, css = "均線偏多 ▲", "bull"
    elif score >= 45:
        label, css = "均線糾結 →", "neut"
    elif score >= 30:
        label, css = "均線偏空 ▼", "bear"
    else:
        label, css = "均線空頭排列 ▼▼", "bear"

    return {
        "score": score,
        "label": label,
        "css": css,
        "above": above,
        "below": below,
        "ordered_bull": ordered_bull,
        "ordered_bear": ordered_bear,
        "ma_vals": ma_vals,
    }


def fib_ma_confluence(all_levels: dict, mas: dict, current_price: float,
                      threshold_pct: float = 0.8) -> list:
    if not mas:
        return []

    ma_vals = {}
    for p, s in mas.items():
        last = s.iloc[-1]
        if pd.notna(last):
            ma_vals[p] = float(last)

    results = []
    for fib_lvl, fib_price in all_levels.items():
        for ma_period, ma_price in ma_vals.items():
            if ma_price == 0:
                continue
            diff_pct = abs(fib_price - ma_price) / ma_price * 100
            if diff_pct <= threshold_pct:
                dist_from_current = fib_price - current_price
                results.append({
                    "fib_level":  fib_lvl,
                    "fib_label":  FIB_LABELS.get(fib_lvl, f"{fib_lvl:.3f}"),
                    "fib_price":  fib_price,
                    "ma_period":  ma_period,
                    "ma_price":   ma_price,
                    "diff_pct":   diff_pct,
                    "dist_from_current": dist_from_current,
                    "is_above":   fib_price > current_price,
                    "strength":   1.0 - diff_pct / threshold_pct,
                })

    results.sort(key=lambda x: abs(x["dist_from_current"]))
    return results


def fib_prediction_table(ret_levels: dict, ext_levels: dict,
                         current_price: float, mas: dict,
                         swing_hi: float, swing_lo: float) -> pd.DataFrame:
    ma_vals = {}
    for p, s in mas.items():
        last = s.iloc[-1]
        if pd.notna(last):
            ma_vals[p] = float(last)

    all_levels = {**ret_levels, **ext_levels}
    rows = []
    for lvl, price in sorted(all_levels.items(), key=lambda x: x[1], reverse=True):
        dist = price - current_price
        dist_pct = dist / current_price * 100

        nearest_ma_period = None
        nearest_ma_dist_pct = None
        if ma_vals:
            dists_to_ma = {p: abs(price - v) / v * 100 for p, v in ma_vals.items()}
            nearest_ma_period = min(dists_to_ma, key=dists_to_ma.get)
            nearest_ma_dist_pct = dists_to_ma[nearest_ma_period]

        role = "壓力 ↑" if price > current_price else "支撐 ↓"
        conf = ""
        if nearest_ma_dist_pct is not None and nearest_ma_dist_pct <= 0.8:
            conf = f"★共振MA{nearest_ma_period}"

        key = "★" if lvl in (0.382, 0.500, 0.618, 1.618) else ""

        rows.append({
            "費氏層級":   FIB_LABELS.get(lvl, f"{lvl:.3f}") + key,
            "預測價位":   price,
            "距現價":     dist,
            "距離%":      dist_pct,
            "最近均線":   f"MA{nearest_ma_period}" if nearest_ma_period else "—",
            "均線距離%":  nearest_ma_dist_pct if nearest_ma_dist_pct is not None else None,
            "共振":       conf,
            "角色":       role,
        })

    return pd.DataFrame(rows)


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
        score += 10
        tags.append("上升趨勢")
    else:
        score -= 10
        tags.append("下降趨勢")

    if price > r382:
        score += 15
        tags.append("站上0.382")
    elif price < r618:
        score -= 15
        tags.append("跌破0.618")

    if price > r500:
        score += 10
        tags.append("站上0.5中線")
    else:
        score -= 5

    if price > h * 0.99:
        score += 15
        tags.append("逼近高點")
    if price < l * 1.01:
        score -= 15
        tags.append("逼近低點")

    score = max(0, min(100, score))

    if score >= 70:
        label, css = "強勢多頭 ▲▲", "bull"
    elif score >= 55:
        label, css = "偏多 ▲", "bull"
    elif score >= 45:
        label, css = "中性觀望 →", "neut"
    elif score >= 30:
        label, css = "偏空 ▼", "bear"
    else:
        label, css = "強勢空頭 ▼▼", "bear"

    return {"label": label, "css": css, "score": score, "tags": tags}


def combined_score(fib_score: int, ma_score: int) -> dict:
    score = int(fib_score * 0.6 + ma_score * 0.4)
    score = max(0, min(100, score))
    if score >= 70:
        label, css = "強力多頭共振 ▲▲", "bull"
    elif score >= 55:
        label, css = "偏多共振 ▲", "bull"
    elif score >= 45:
        label, css = "中性觀望 →", "neut"
    elif score >= 30:
        label, css = "偏空 ▼", "bear"
    else:
        label, css = "強力空頭 ▼▼", "bear"
    return {"score": score, "label": label, "css": css}


# ─────────────────────────────────────────────────────────────────────────────
# Tomorrow target
# ─────────────────────────────────────────────────────────────────────────────

def tomorrow_targets(price: float, ret_levels: dict, ext_levels: dict,
                     trend: str, ma_vals: dict = None) -> dict:
    all_levels = {**ret_levels, **ext_levels}
    if ma_vals:
        for p, v in ma_vals.items():
            all_levels[f"MA{p}"] = v

    sorted_asc = sorted(all_levels.items(), key=lambda x: x[1])

    resistance_list = [(l, p) for l, p in sorted_asc if p > price]
    support_list    = [(l, p) for l, p in sorted_asc if p < price]

    r1 = resistance_list[0] if resistance_list else None
    s1 = support_list[-1]   if support_list    else None

    if r1 and s1:
        up_tgt = r1[1]
        dn_tgt = s1[1]
        r1_lbl = FIB_LABELS.get(r1[0], str(r1[0])) if not isinstance(r1[0], str) else r1[0]
        s1_lbl = FIB_LABELS.get(s1[0], str(s1[0])) if not isinstance(s1[0], str) else s1[0]
        bias   = "偏多，關注壓力" if trend == "up" else "偏空，關注支撐"
        desc   = (f"趨勢{bias}｜"
                  f"上方目標 {up_tgt:,.0f}（{r1_lbl}）｜"
                  f"下方支撐 {dn_tgt:,.0f}（{s1_lbl}）")
    elif r1:
        up_tgt = r1[1]
        dn_tgt = price * 0.99
        r1_lbl = FIB_LABELS.get(r1[0], str(r1[0])) if not isinstance(r1[0], str) else r1[0]
        desc   = f"上方目標 {up_tgt:,.0f}（{r1_lbl}）"
    elif s1:
        up_tgt = price * 1.01
        dn_tgt = s1[1]
        s1_lbl = FIB_LABELS.get(s1[0], str(s1[0])) if not isinstance(s1[0], str) else s1[0]
        desc   = f"下方支撐 {dn_tgt:,.0f}（{s1_lbl}）"
    else:
        up_tgt, dn_tgt = price * 1.01, price * 0.99
        desc = "無明確費氏目標"

    return {
        "up_target":  up_tgt,
        "dn_target":  dn_tgt,
        "pivot":      price,
        "description": desc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart builder
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame,
                name: str,
                ret_levels: dict,
                ext_levels: dict,
                swing_hi: float,  swing_hi_idx: int,
                swing_lo: float,  swing_lo_idx: int,
                trend: str,
                show_ext: bool,
                tmr: dict,
                mas: dict = None) -> go.Figure:

    current = float(df["Close"].iloc[-1])
    x_axis  = df.index.tolist()
    x0, x1  = x_axis[0], x_axis[-1]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.20, 0.15],
        vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=x_axis,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="K線",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
        line_width=1,
    ), row=1, col=1)

    if mas:
        for period in sorted(mas.keys()):
            ma_series = mas[period]
            color = MA_COLORS.get(period, "#ffffff")
            valid = ma_series.dropna()
            if len(valid) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=valid.index.tolist(),
                y=valid.tolist(),
                mode="lines",
                name=f"MA{period}",
                line=dict(color=color, width=1.5),
                opacity=0.85,
            ), row=1, col=1)

    for lvl, price in ret_levels.items():
        color = FIB_COLORS.get(lvl, "#888")
        lw    = 2 if lvl in (0.382, 0.618) else 1
        dash  = "solid" if lvl in (0.382, 0.618) else "dash"
        fig.add_shape(type="line", x0=x0, x1=x1, y0=price, y1=price,
                      line=dict(color=color, width=lw, dash=dash),
                      row=1, col=1)
        fig.add_annotation(
            x=x1, y=price,
            text=f"  {FIB_LABELS.get(lvl, f'{lvl:.3f}')} : {price:,.0f}",
            xanchor="left", showarrow=False,
            font=dict(size=10, color=color),
            row=1, col=1,
        )

    if show_ext:
        for lvl, price in ext_levels.items():
            color = FIB_COLORS.get(lvl, "#888")
            fig.add_shape(type="line", x0=x0, x1=x1, y0=price, y1=price,
                          line=dict(color=color, width=1, dash="dot"),
                          row=1, col=1)
            fig.add_annotation(
                x=x1, y=price,
                text=f"  {FIB_LABELS.get(lvl, f'{lvl:.3f}')} : {price:,.0f}",
                xanchor="left", showarrow=False,
                font=dict(size=9, color=color),
                row=1, col=1,
            )

    fig.add_trace(go.Scatter(
        x=[x_axis[swing_hi_idx]], y=[swing_hi],
        mode="markers+text",
        marker=dict(symbol="triangle-down", size=14, color="#ef5350"),
        text=["▼波段高"], textposition="top center",
        textfont=dict(size=11, color="#ef5350"),
        name="波段高點", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[x_axis[swing_lo_idx]], y=[swing_lo],
        mode="markers+text",
        marker=dict(symbol="triangle-up", size=14, color="#26a69a"),
        text=["▲波段低"], textposition="bottom center",
        textfont=dict(size=11, color="#26a69a"),
        name="波段低點", showlegend=False,
    ), row=1, col=1)

    fig.add_shape(type="line", x0=x0, x1=x1,
                  y0=current, y1=current,
                  line=dict(color="white", width=1.5, dash="dot"),
                  row=1, col=1)
    fig.add_annotation(
        x=x0, y=current,
        text=f"  ▶ 現價 {current:,.0f}",
        xanchor="right", showarrow=False,
        font=dict(size=11, color="white"),
        row=1, col=1,
    )

    for tgt_price, tgt_color, tgt_lbl in [
        (tmr["up_target"], "#26a69a", f"↑目標 {tmr['up_target']:,.0f}"),
        (tmr["dn_target"], "#ef5350", f"↓支撐 {tmr['dn_target']:,.0f}"),
    ]:
        fig.add_shape(type="line", x0=x0, x1=x1,
                      y0=tgt_price, y1=tgt_price,
                      line=dict(color=tgt_color, width=2, dash="dashdot"),
                      row=1, col=1)
        fig.add_annotation(
            x=x0, y=tgt_price,
            text=f"  {tgt_lbl}",
            xanchor="right", showarrow=False,
            font=dict(size=10, color=tgt_color),
            row=1, col=1,
        )

    vol_colors = [
        "#26a69a" if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i])
        else "#ef5350"
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=x_axis, y=df["Volume"].tolist(),
        name="成交量", marker_color=vol_colors, opacity=0.8,
    ), row=2, col=1)

    diff = swing_hi - swing_lo if swing_hi != swing_lo else 1.0
    fib_osc = [
        max(0.0, min(1.0, (swing_hi - float(p)) / diff))
        for p in df["Close"]
    ]
    osc_colors = [
        "#ef5350" if v > 0.618 else ("#ffd700" if v > 0.382 else "#26a69a")
        for v in fib_osc
    ]
    fig.add_trace(go.Bar(
        x=x_axis, y=fib_osc,
        name="費氏位階", marker_color=osc_colors, opacity=0.9,
    ), row=3, col=1)
    for ref in [0.382, 0.618]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=ref, y1=ref,
                      line=dict(color="white", width=1, dash="dash"),
                      row=3, col=1)

    trend_arrow = "↑ 多頭" if trend == "up" else "↓ 空頭"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=850,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=160, t=55, b=10),
        font=dict(size=11, color="#ccc"),
        title=dict(
            text=(f"<b>{name}</b>　｜　費氏×均線分析　｜　"
                  f"現價 <b>{current:,.0f}</b>　｜　"
                  f"趨勢 <b>{trend_arrow}</b>"),
            x=0.02, font=dict(size=15, color="white"),
        ),
    )
    for row in [1, 2, 3]:
        fig.update_xaxes(
            gridcolor="#1e2130", zeroline=False,
            showticklabels=(row == 3), row=row, col=1,
        )
        fig.update_yaxes(gridcolor="#1e2130", zeroline=False, row=row, col=1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar() -> dict:
    st.sidebar.markdown("## 📊 台指期 費氏×均線看板")
    st.sidebar.markdown("---")

    ticker_label = st.sidebar.selectbox("商品選擇", list(TICKERS.keys()), index=0)
    ticker = TICKERS[ticker_label]

    st.sidebar.markdown("---")

    interval_label = st.sidebar.selectbox(
        "時間週期", list(INTERVAL_LABELS.values()), index=5,
    )
    interval = [k for k, v in INTERVAL_LABELS.items() if v == interval_label][0]

    valid_period_keys   = INTERVAL_PERIODS.get(interval, ["1y"])
    valid_period_labels = [PERIOD_LABELS[k] for k in valid_period_keys]
    default_period_idx  = min(2, len(valid_period_labels) - 1)
    period_label = st.sidebar.selectbox(
        "資料期間", valid_period_labels, index=default_period_idx,
    )
    period = [k for k, v in PERIOD_LABELS.items() if v == period_label][0]

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📈 均線設定")

    all_ma_options = [5, 10, 20, 60, 120, 240]
    ma_default     = [20, 60, 120]
    selected_mas   = st.sidebar.multiselect(
        "均線週期",
        options=all_ma_options,
        default=ma_default,
        format_func=lambda x: f"MA{x}",
    )
    if not selected_mas:
        selected_mas = ma_default

    ma_threshold = st.sidebar.slider(
        "共振靈敏度（費氏與均線距離%）",
        min_value=0.1, max_value=2.0, value=0.8, step=0.1,
        help="費氏位階與均線價格差在此%以內視為共振",
    )

    st.sidebar.markdown("---")

    swing_window = st.sidebar.slider(
        "自動波段偵測靈敏度（K棒數）",
        min_value=3, max_value=30, value=10, step=1,
    )

    show_ext = st.sidebar.checkbox("顯示費氏延伸線", value=True)

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
        "ticker":        ticker,
        "ticker_label":  ticker_label.split()[0],
        "interval":      interval,
        "period":        period,
        "swing_window":  swing_window,
        "show_ext":      show_ext,
        "auto_refresh":  auto_refresh,
        "selected_mas":  sorted(selected_mas),
        "ma_threshold":  ma_threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def card(title: str, value: str, sub: str = "", css_class: str = "") -> str:
    return (f'<div class="card {css_class}">'
            f'<h4>{title}</h4>'
            f'<div class="val">{value}</div>'
            f'<div class="sub">{sub}</div>'
            f'</div>')


def fib_table_html(all_levels: dict, current_price: float,
                   mas: dict = None) -> str:
    ma_vals = {}
    if mas:
        for p, s in mas.items():
            last = s.iloc[-1]
            if pd.notna(last):
                ma_vals[p] = float(last)

    rows = ""
    sorted_lvl = sorted(all_levels.items(), key=lambda x: x[1], reverse=True)
    for lvl, price in sorted_lvl:
        diff   = current_price - price
        pct    = diff / price * 100
        arrow  = "▲" if diff >= 0 else "▼"
        color  = "#26a69a" if diff >= 0 else "#ef5350"
        hl     = 'class="fib-highlight"' if lvl in (0.382, 0.618) else ""

        ma_note = ""
        if ma_vals:
            dists = {p: abs(price - v) / v * 100 for p, v in ma_vals.items()}
            nearest_p = min(dists, key=dists.get)
            nd = dists[nearest_p]
            if nd <= 1.0:
                mc = MA_COLORS.get(nearest_p, "#fff")
                ma_note = f' <span style="color:{mc};font-size:11px">★MA{nearest_p}({nd:.2f}%)</span>'

        rows += (f'<tr {hl}>'
                 f'<td>{FIB_LABELS.get(lvl, f"{lvl:.3f}")}{ma_note}</td>'
                 f'<td style="color:{FIB_COLORS.get(lvl,"#888")}"><b>{price:,.0f}</b></td>'
                 f'<td style="color:{color}">{arrow} {abs(diff):,.0f} ({abs(pct):.2f}%)</td>'
                 f'</tr>')
    return (f'<table class="fib-table" style="width:100%;border-collapse:collapse">'
            f'<thead><tr>'
            f'<th>費氏層級</th><th>價格</th><th>距現價</th>'
            f'</tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def _build_confluence_suggestion(confluence: list, ma_signal: dict,
                                  fib_signal: dict, trend: str,
                                  current_price: float) -> str:
    lines = []

    if trend == "up":
        lines.append("📈 **當前趨勢偏多**，以做多思維操作，逢費氏支撐買進。")
    else:
        lines.append("📉 **當前趨勢偏空**，以做空思維操作，逢費氏壓力賣出。")

    if ma_signal.get("ordered_bull"):
        lines.append("✅ **均線多頭排列**（短均在長均之上），趨勢力道強，可持倉待漲。")
    elif ma_signal.get("ordered_bear"):
        lines.append("⚠️ **均線空頭排列**（短均在長均之下），趨勢偏弱，宜輕倉或觀望。")
    else:
        lines.append("🔄 **均線糾結中**，方向不明，建議等待突破確認再進場。")

    above = ma_signal.get("above", [])
    below = ma_signal.get("below", [])
    if above:
        lines.append(f"💚 現價站上 {', '.join(f'MA{p}' for p in sorted(above))} 之上，形成支撐。")
    if below:
        lines.append(f"🔴 現價低於 {', '.join(f'MA{p}' for p in sorted(below))}，形成壓力。")

    if confluence:
        strong = [c for c in confluence if c["strength"] >= 0.5]
        if strong:
            lines.append("⭐ **費氏×均線共振點**（關鍵支撐/壓力）：")
            for c in strong[:5]:
                role = "壓力" if c["is_above"] else "支撐"
                lines.append(
                    f"　• **{c['fib_label']}** 與 **MA{c['ma_period']}** 在 "
                    f"**{c['fib_price']:,.0f}** 附近共振 → 強力{role}"
                )
    else:
        lines.append("📊 目前費氏位階與均線無明顯共振，各位階獨立運作。")

    fib_sc = fib_signal.get("score", 50)
    ma_sc  = ma_signal.get("score", 50)
    comb   = combined_score(fib_sc, ma_sc)
    if comb["score"] >= 65:
        lines.append(f"🚦 **綜合評分 {comb['score']}/100** → **積極做多**，逢回不追空。")
    elif comb["score"] >= 55:
        lines.append(f"🚦 **綜合評分 {comb['score']}/100** → **偏多操作**，守支撐做多。")
    elif comb["score"] >= 45:
        lines.append(f"🚦 **綜合評分 {comb['score']}/100** → **觀望為主**，等方向確認。")
    elif comb["score"] >= 35:
        lines.append(f"🚦 **綜合評分 {comb['score']}/100** → **偏空操作**，守壓力做空。")
    else:
        lines.append(f"🚦 **綜合評分 {comb['score']}/100** → **積極做空**，逢彈不追多。")

    return "\n\n".join(lines)


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

    # ── Fibonacci anchors (auto-detected from selected timeframe) ─────────────
    swing_hi, sh_idx, swing_lo, sl_idx = dominant_swing(df, cfg["swing_window"])
    trend = trend_from_swings(sh_idx, sl_idx)

    ret = fib_retracement(swing_hi, swing_lo)
    ext = fib_extension(swing_hi, swing_lo, trend)

    current_price = float(df["Close"].iloc[-1])
    prev_price    = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
    change        = current_price - prev_price
    change_pct    = change / prev_price * 100 if prev_price else 0

    # ── Moving Averages ───────────────────────────────────────────────────────
    mas         = compute_mas(df, cfg["selected_mas"])
    ma_signal   = ma_trend_signal(mas, current_price)
    ma_vals     = ma_signal.get("ma_vals", {})

    # ── Signals ───────────────────────────────────────────────────────────────
    signal     = bull_bear_signal(current_price, ret, trend)
    comb_sig   = combined_score(signal["score"], ma_signal["score"])
    tmr        = tomorrow_targets(current_price, ret, ext, trend, ma_vals)
    lb, lb_p, ab, ab_p, pct_in = current_fib_position(current_price, ret)

    # ── Confluence ────────────────────────────────────────────────────────────
    all_fib_levels = {**ret}
    if cfg["show_ext"]:
        all_fib_levels.update(ext)
    confluence = fib_ma_confluence(all_fib_levels, mas, current_price,
                                    cfg["ma_threshold"])

    # ── Header metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    price_css = "card-green" if change >= 0 else "card-red"
    with col1:
        st.markdown(card("現價", f"{current_price:,.0f}",
                         f"{'▲' if change>=0 else '▼'} {change:+,.0f} ({change_pct:+.2f}%)",
                         price_css), unsafe_allow_html=True)
    with col2:
        st.markdown(card("波段高點", f"{swing_hi:,.0f}", "錨點（Fib 0.0）", "card-red"),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(card("波段低點", f"{swing_lo:,.0f}", "錨點（Fib 1.0）", "card-green"),
                    unsafe_allow_html=True)
    with col4:
        sig_css = "card-green" if signal["css"] == "bull" else (
            "card-red" if signal["css"] == "bear" else "card-gold"
        )
        st.markdown(card("費氏訊號", signal["label"],
                         f"費氏強度 {signal['score']}/100", sig_css),
                    unsafe_allow_html=True)
    with col5:
        ma_css = "card-green" if ma_signal["css"] == "bull" else (
            "card-red" if ma_signal["css"] == "bear" else "card-gold"
        )
        st.markdown(card("均線訊號", ma_signal["label"],
                         f"均線強度 {ma_signal['score']}/100", ma_css),
                    unsafe_allow_html=True)
    with col6:
        comb_css = "card-green" if comb_sig["css"] == "bull" else (
            "card-red" if comb_sig["css"] == "bear" else "card-gold"
        )
        st.markdown(card("費氏×均線綜合", comb_sig["label"],
                         f"綜合評分 {comb_sig['score']}/100 (費氏60%+均線40%)", comb_css),
                    unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 K線 + 費氏 + 均線",
        "🎯 明日目標價 & 費氏預測",
        "⭐ 費氏×均線共振",
        "📋 費氏層級表",
        "📊 多時框概覽",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1
    # ════════════════════════════════════════════════════════════════
    with tab1:
        fig = build_chart(
            df, cfg["ticker_label"],
            ret, ext,
            swing_hi, sh_idx,
            swing_lo, sl_idx,
            trend, cfg["show_ext"],
            tmr, mas,
        )
        st.plotly_chart(fig, use_container_width=True)

        if mas:
            ma_chips = ""
            for p in sorted(mas.keys()):
                v = ma_vals.get(p)
                color = MA_COLORS.get(p, "#fff")
                val_str = f"{v:,.0f}" if v else "—"
                ma_chips += (
                    f'<span style="background:{color}22;border:1px solid {color};'
                    f'color:{color};border-radius:4px;padding:2px 8px;'
                    f'margin:2px;font-size:12px;display:inline-block">'
                    f'MA{p} {val_str}</span>'
                )
            st.markdown(ma_chips, unsafe_allow_html=True)

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
        st.markdown(
            f'🚦 **費氏訊號**: <span class="{signal["css"]}" style="font-size:18px;font-weight:bold">'
            f'{signal["label"]}</span>'
            f'　<span style="color:#aaa;font-size:14px">{"　".join(signal["tags"])}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'📊 **均線訊號**: <span class="{ma_signal["css"]}" style="font-size:16px;font-weight:bold">'
            f'{ma_signal["label"]}</span>'
            f'　<span style="color:#aaa;font-size:13px">強度 {ma_signal["score"]}/100</span>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════
    # TAB 2
    # ════════════════════════════════════════════════════════════════
    with tab2:
        now_tw   = datetime.now(TW_TZ)
        next_day = now_tw + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        next_date_str = next_day.strftime("%Y-%m-%d (%A)")

        st.markdown(f"### 🎯 下一交易日目標　`{next_date_str}`")
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
        st.markdown("#### 📌 各費氏層級預測點位（含均線參考）")

        pred_df = fib_prediction_table(ret, ext if cfg["show_ext"] else {},
                                        current_price, mas, swing_hi, swing_lo)

        display_pred = pred_df.copy()
        display_pred["預測價位"] = display_pred["預測價位"].apply(lambda x: f"{x:,.0f}")
        display_pred["距現價"]   = display_pred["距現價"].apply(
            lambda x: f"{'▲' if x > 0 else '▼'} {abs(x):,.0f}"
        )
        display_pred["距離%"]    = display_pred["距離%"].apply(lambda x: f"{x:+.2f}%")
        display_pred["均線距離%"] = display_pred["均線距離%"].apply(
            lambda x: f"{x:.2f}%" if x is not None else "—"
        )
        st.dataframe(display_pred, use_container_width=True, height=400)

        st.markdown("#### 📊 各費氏層級與現價距離")
        fig2 = go.Figure()
        all_fib = {**ret}
        if cfg["show_ext"]:
            all_fib.update(ext)

        sorted_f  = sorted(all_fib.items(), key=lambda x: x[1])
        prices_f  = [p for _, p in sorted_f]
        labels_f  = [FIB_LABELS.get(l, f"{l:.3f}") for l, _ in sorted_f]
        dists_f   = [abs(p - current_price) for p in prices_f]
        bar_colors = ["#ef5350" if p > current_price else "#26a69a" for p in prices_f]

        fig2.add_trace(go.Bar(
            x=labels_f, y=dists_f,
            marker_color=bar_colors,
            text=[f"{p:,.0f}" for p in prices_f],
            textposition="outside",
        ))
        if ma_vals:
            for p, v in sorted(ma_vals.items()):
                fig2.add_hline(
                    y=abs(v - current_price),
                    line_color=MA_COLORS.get(p, "#fff"),
                    line_dash="dot", line_width=1,
                    annotation_text=f"MA{p}={v:,.0f}",
                    annotation_font_size=9,
                )
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            title="各費氏層級與現價的距離（綠=支撐/紅=壓力）",
            showlegend=False, font=dict(size=11), xaxis_tickangle=-30,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 📅 近期收盤費氏位階追蹤")
        hist = df.tail(20).copy()
        diff_total = swing_hi - swing_lo if swing_hi != swing_lo else 1.0
        hist["波段位置%"] = ((hist["Close"] - swing_lo) / diff_total * 100).round(2)
        hist["費氏位階"] = hist["波段位置%"].apply(lambda x:
            "1.000 終點"   if x <= 0    else
            "0.786"        if x <= 21.4 else
            "0.618 ★"     if x <= 38.2 else
            "0.500 中線"   if x <= 50.0 else
            "0.382 ★"     if x <= 61.8 else
            "0.236"        if x <= 76.4 else
            "0.000 起點"   if x <= 100  else
            "1.272 延伸"   if x <= 127.2 else
            "1.414 延伸"   if x <= 141.4 else
            "1.618 延伸★"  if x <= 161.8 else
            "2.000 延伸"   if x <= 200  else
            "2.618+ 延伸"
        )
        display_hist = hist[["Close", "High", "Low", "Volume", "波段位置%", "費氏位階"]].copy()
        if hasattr(display_hist.index, "strftime"):
            display_hist.index = display_hist.index.strftime("%Y-%m-%d %H:%M")
        display_hist.columns = ["收盤", "最高", "最低", "成交量", "波段位置%", "費氏位階"]

        def _color_fib(val):
            try:
                v = float(val)
                if v > 100:    bg = "#2196F340"
                elif v > 61.8: bg = "#26a69a40"
                elif v > 38.2: bg = "#ffd70040"
                else:          bg = "#ef535040"
                return f"background-color: {bg}; color: white"
            except Exception:
                return ""

        st.dataframe(
            display_hist.style.map(_color_fib, subset=["波段位置%"]),
            use_container_width=True,
        )

    # ════════════════════════════════════════════════════════════════
    # TAB 3 – 費氏×均線共振
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### ⭐ 費氏×均線共振分析")
        st.markdown(
            f"共振靈敏度：費氏位階與均線價格差在 **{cfg['ma_threshold']:.1f}%** 以內視為共振"
        )
        st.markdown("")

        st.markdown("#### 🔖 自動錨點基準")
        anchor_cols = st.columns(4)
        with anchor_cols[0]:
            interval_name = INTERVAL_LABELS.get(cfg["interval"], cfg["interval"])
            st.markdown(card("時框錨點", interval_name, "自動偵測波段", "card-blue"), unsafe_allow_html=True)
        with anchor_cols[1]:
            st.markdown(card("偵測低點（Fib 1.0）", f"{swing_lo:,.0f}",
                             "費氏起點", "card-green"), unsafe_allow_html=True)
        with anchor_cols[2]:
            st.markdown(card("偵測高點（Fib 0.0）", f"{swing_hi:,.0f}",
                             "費氏終點", "card-red"), unsafe_allow_html=True)
        with anchor_cols[3]:
            diff_pts = swing_hi - swing_lo
            st.markdown(card("波幅", f"{diff_pts:,.0f}",
                             f"{diff_pts/swing_lo*100:.1f}%", "card-gold"), unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### 📐 各費氏點位預測（以錨點計算）")

        fib_detail_rows = []
        for lvl in sorted(FIB_RET + FIB_EXT):
            price = ret.get(lvl) or ext.get(lvl)
            if price is None:
                continue
            dist     = price - current_price
            dist_pct = dist / current_price * 100
            ma_proximity = []
            for p, v in sorted(ma_vals.items()):
                d_pct = abs(price - v) / v * 100
                if d_pct <= 2.0:
                    ma_proximity.append(f"MA{p}({d_pct:.2f}%)")
            role = "壓力" if price > current_price else "支撐"
            key_flag = "★★" if lvl in (0.382, 0.618, 1.618) else (
                "★" if lvl in (0.236, 0.500, 0.786, 1.272) else ""
            )
            fib_detail_rows.append({
                "費氏層級":   FIB_LABELS.get(lvl, f"{lvl:.3f}"),
                "重要性":     key_flag,
                "預測點位":   f"{price:,.0f}",
                "距現價":     f"{'▲' if dist > 0 else '▼'}{abs(dist):,.0f}",
                "距離%":      f"{dist_pct:+.2f}%",
                "角色":       role,
                "均線接近":   "  ".join(ma_proximity) if ma_proximity else "—",
            })

        st.dataframe(pd.DataFrame(fib_detail_rows), use_container_width=True, height=380)
        st.markdown("")

        st.markdown("#### 📈 當前均線數值")
        if ma_vals:
            ma_row_cols = st.columns(len(ma_vals))
            for idx, (period, val) in enumerate(sorted(ma_vals.items())):
                above_flag = current_price > val
                with ma_row_cols[idx]:
                    css = "card-green" if above_flag else "card-red"
                    sub = f"現價{'高於' if above_flag else '低於'}均線 {abs(current_price-val):,.0f}"
                    st.markdown(card(f"MA{period}", f"{val:,.0f}", sub, css),
                                unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### 🌟 費氏×均線共振點位")
        if confluence:
            conf_rows = []
            for c in confluence:
                conf_rows.append({
                    "費氏層級":   c["fib_label"],
                    "費氏價位":   f"{c['fib_price']:,.0f}",
                    "共振均線":   f"MA{c['ma_period']}",
                    "均線價位":   f"{c['ma_price']:,.0f}",
                    "價差%":      f"{c['diff_pct']:.3f}%",
                    "共振強度":   f"{'★'*int(c['strength']*5+1)} {c['strength']*100:.0f}%",
                    "角色":       "壓力 ↑" if c["is_above"] else "支撐 ↓",
                    "距現價":     f"{'▲' if c['dist_from_current']>0 else '▼'}{abs(c['dist_from_current']):,.0f}",
                })
            st.dataframe(pd.DataFrame(conf_rows), use_container_width=True)
        else:
            st.info(
                f"目前在 {cfg['ma_threshold']:.1f}% 靈敏度下無費氏×均線共振點。\n"
                "可調高側邊欄「共振靈敏度」以找到更多共振位置。"
            )

        st.markdown("")
        st.markdown("#### 📊 費氏點位 × 均線位置分佈圖")
        fig3 = go.Figure()
        all_fib_for_chart = {**ret}
        if cfg["show_ext"]:
            all_fib_for_chart.update(ext)
        fib_sorted = sorted(all_fib_for_chart.items(), key=lambda x: x[1])
        fig3.add_trace(go.Scatter(
            x=[FIB_LABELS.get(l, f"{l:.3f}") for l, _ in fib_sorted],
            y=[p for _, p in fib_sorted],
            mode="markers+lines",
            marker=dict(color=[FIB_COLORS.get(l, "#888") for l, _ in fib_sorted],
                        size=10, symbol="diamond"),
            line=dict(color="#444", width=1, dash="dot"),
            name="費氏點位",
        ))
        for period, val in sorted(ma_vals.items()):
            color = MA_COLORS.get(period, "#fff")
            fig3.add_hline(y=val, line_color=color, line_dash="solid", line_width=2,
                           annotation_text=f"MA{period} {val:,.0f}",
                           annotation_font_color=color, annotation_font_size=10)
        fig3.add_hline(y=current_price, line_color="white", line_dash="dashdot", line_width=2,
                       annotation_text=f"現價 {current_price:,.0f}", annotation_font_color="white")
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=420, margin=dict(l=10, r=120, t=40, b=10),
            title="費氏點位 vs 均線（共振點為黃金機會）",
            showlegend=True, xaxis_tickangle=-30, font=dict(size=11),
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### 💡 費氏×均線操作建議")
        suggestion = _build_confluence_suggestion(
            confluence, ma_signal, signal, trend, current_price
        )
        st.markdown(f'<div class="suggest-box">{suggestion}</div>', unsafe_allow_html=True)
        st.caption("⚠️ 以上分析僅供參考，不構成投資建議。市場有風險，操作請謹慎。")

    # ════════════════════════════════════════════════════════════════
    # TAB 4
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📋 費氏回檔層級詳表")
        st.markdown(fib_table_html(ret, current_price, mas), unsafe_allow_html=True)
        if cfg["show_ext"]:
            st.markdown("### 📋 費氏延伸層級詳表")
            st.markdown(fib_table_html(ext, current_price, mas), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("#### 費氏數列說明")
        st.markdown("""
| 層級 | 意義 |
|------|------|
| **0.382** | 最常見的回檔支撐/壓力，主力第一守備區 |
| **0.500** | 心理中線，多空分水嶺 |
| **0.618** | 黃金比例，突破或守住代表趨勢確立 |
| **1.618** | 延伸目標，波段完成位 |
""")
        if mas:
            st.markdown("#### 均線顏色說明")
            for p in sorted(mas.keys()):
                color = MA_COLORS.get(p, "#fff")
                st.markdown(
                    f'<span style="color:{color}">■</span> **MA{p}**　',
                    unsafe_allow_html=True,
                )

    # ════════════════════════════════════════════════════════════════
    # TAB 5
    # ════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("### 📊 多時間框架費氏位階概覽")
        st.caption("各時框自動抓取近期波段，計算現價費氏回檔位置")

        mtf_intervals = {
            "日線 (3個月)":  ("1d",  "3mo"),
            "日線 (1年)":    ("1d",  "1y"),
            "週線 (2年)":    ("1wk", "2y"),
            "月線 (5年)":    ("1mo", "5y"),
        }

        mtf_results = []
        prog = st.progress(0)
        for idx, (lbl, (iv, pr)) in enumerate(mtf_intervals.items()):
            prog.progress((idx + 1) / len(mtf_intervals), text=f"載入 {lbl}…")
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
                "時間框架":   lbl,
                "現價":       f"{cp_m:,.0f}",
                "波段高":     f"{sh_m:,.0f}",
                "波段低":     f"{sl_m:,.0f}",
                "費氏回檔%":  f"{ret_pct:.1f}%",
                "趨勢":       "↑多" if tr_m == "up" else "↓空",
                "多空訊號":   sig_m["label"],
                "明日目標↑":  f"{tmr_m['up_target']:,.0f}",
                "明日支撐↓":  f"{tmr_m['dn_target']:,.0f}",
            })
        prog.empty()

        if mtf_results:
            mtf_df = pd.DataFrame(mtf_results).set_index("時間框架")
            st.dataframe(mtf_df, use_container_width=True)

            scores, labels_mtf = [], []
            for row in mtf_results:
                labels_mtf.append(row["時間框架"])
                try:
                    df_m2 = fetch_ohlcv(cfg["ticker"],
                                        mtf_intervals[row["時間框架"]][0],
                                        mtf_intervals[row["時間框架"]][1])
                    sh2, shi2, sl2, sli2 = dominant_swing(df_m2, cfg["swing_window"])
                    tr2  = trend_from_swings(shi2, sli2)
                    ret2 = fib_retracement(sh2, sl2)
                    cp2  = float(df_m2["Close"].iloc[-1])
                    sc2  = bull_bear_signal(cp2, ret2, tr2)
                    scores.append(sc2["score"])
                except Exception:
                    scores.append(50)

            fig_mtf = go.Figure(go.Bar(
                x=labels_mtf, y=scores,
                marker_color=[
                    "#26a69a" if s >= 60 else ("#ffd700" if s >= 45 else "#ef5350")
                    for s in scores
                ],
                text=[f"{s}/100" for s in scores],
                textposition="outside",
            ))
            fig_mtf.add_hline(y=60, line_color="#26a69a", line_dash="dash",
                               annotation_text="多頭區")
            fig_mtf.add_hline(y=40, line_color="#ef5350", line_dash="dash",
                               annotation_text="空頭區")
            fig_mtf.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                height=300, margin=dict(l=10, r=10, t=30, b=10),
                title="多時框多空強度評分",
                showlegend=False, yaxis=dict(range=[0, 110]),
            )
            st.plotly_chart(fig_mtf, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#555;font-size:12px">'
        '資料來源：Yahoo Finance ｜ 費氏數列×移動平均分析看板 ｜ '
        '僅供參考，不構成投資建議'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
