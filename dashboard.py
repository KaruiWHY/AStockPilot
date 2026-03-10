# -*- coding: utf-8 -*-
"""
Interactive dashboard for stock analysis.

Run:
    streamlit run dashboard.py
"""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from tools import stock_tools


INDEX_CODE_MAP = {
    "上证指数": "sh.000001",
    "深证成指": "sz.399001",
    "创业板指": "sz.399006",
}


st.set_page_config(page_title="Quant Dashboard", layout="wide")
st.title("Quant Dashboard")
st.caption("K-line, market overview, screening, backtest, and sentiment prototype.")

if "watchlist_codes" not in st.session_state:
    st.session_state["watchlist_codes"] = "600000,000001,300750"
if "active_symbol" not in st.session_state:
    st.session_state["active_symbol"] = "600000"
if "screen_result" not in st.session_state:
    st.session_state["screen_result"] = None
if "trader_select_result" not in st.session_state:
    st.session_state["trader_select_result"] = None


@st.cache_data(ttl=300)
def cached_market_overview() -> dict:
    return stock_tools.get_market_overview()


@st.cache_data(ttl=300)
def cached_stock_history(symbol: str, start_date: str, end_date: str, frequency: str) -> dict:
    return stock_tools.get_stock_history(symbol=symbol, start_date=start_date, end_date=end_date, frequency=frequency)


@st.cache_data(ttl=300)
def cached_stock_quote(symbol: str) -> dict:
    return stock_tools.get_stock_quote(symbol)


@st.cache_data(ttl=300)
def cached_technical(symbol: str, start_date: str, end_date: str) -> dict:
    return stock_tools.get_technical_analysis(symbol=symbol, start_date=start_date, end_date=end_date)


@st.cache_data(ttl=300)
def cached_backtest(symbol: str, start_date: str, end_date: str, fast: int, slow: int, capital: float) -> dict:
    return stock_tools.backtest_moving_average_strategy(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fast_period=fast,
        slow_period=slow,
        initial_capital=capital,
    )


@st.cache_data(ttl=300)
def cached_grid_search(
    symbol: str,
    start_date: str,
    end_date: str,
    fast_candidates: str,
    slow_candidates: str,
    top_n: int,
) -> dict:
    return stock_tools.backtest_ma_grid_search(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fast_candidates=fast_candidates,
        slow_candidates=slow_candidates,
        top_n=top_n,
    )


@st.cache_data(ttl=300)
def cached_screening(
    start_date: str,
    end_date: str,
    universe_codes: str,
    universe_limit: int,
    top_n: int,
    min_avg_amount: float,
) -> dict:
    return stock_tools.screen_stocks_by_factors(
        start_date=start_date,
        end_date=end_date,
        universe_codes=universe_codes,
        universe_limit=universe_limit,
        top_n=top_n,
        min_avg_amount=min_avg_amount,
    )


@st.cache_data(ttl=300)
def cached_trader_selection(
    start_date: str,
    end_date: str,
    universe_codes: str,
    universe_limit: int,
    candidate_n: int,
    pick_n: int,
    min_avg_amount: float,
    fast_period: int,
    slow_period: int,
    initial_capital: float,
) -> dict:
    return stock_tools.select_stocks_for_trader(
        start_date=start_date,
        end_date=end_date,
        universe_codes=universe_codes,
        universe_limit=universe_limit,
        candidate_n=candidate_n,
        pick_n=pick_n,
        min_avg_amount=min_avg_amount,
        fast_period=fast_period,
        slow_period=slow_period,
        initial_capital=initial_capital,
    )


def _history_to_df(raw: dict) -> pd.DataFrame:
    data = raw.get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume", "amount", "pctChg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_codes(raw_codes: str) -> list:
    values = [x.strip() for x in str(raw_codes).split(",") if x.strip()]
    return values


def _merge_codes(base_codes: list, new_codes: list) -> list:
    merged = []
    seen = set()
    for code in base_codes + new_codes:
        c = str(code).strip()
        if not c:
            continue
        if c not in seen:
            merged.append(c)
            seen.add(c)
    return merged


def _compute_curve_stats(df: pd.DataFrame) -> dict:
    if df.empty or "close" not in df.columns:
        return {}

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(close) < 2:
        return {}

    ret = close.pct_change().dropna()
    total_return = close.iloc[-1] / close.iloc[0] - 1
    annual_vol = ret.std() * (252 ** 0.5) if not ret.empty else 0.0
    running_max = close.cummax()
    drawdown = close / running_max - 1
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    sharpe = 0.0
    if ret.std() and ret.std() > 0:
        sharpe = ret.mean() / ret.std() * (252 ** 0.5)

    return {
        "total_return_pct": round(float(total_return * 100), 2),
        "annual_vol_pct": round(float(annual_vol * 100), 2),
        "max_drawdown_pct": round(float(max_drawdown * 100), 2),
        "sharpe": round(float(sharpe), 3),
    }


def _build_compare_figure(curves: dict) -> go.Figure:
    fig = go.Figure()
    for code, df in curves.items():
        if df.empty or "close" not in df.columns:
            continue
        close = pd.to_numeric(df["close"], errors="coerce")
        if close.empty:
            continue
        base = close.iloc[0]
        if pd.isna(base) or base == 0:
            continue
        normalized = close / base * 100
        fig.add_trace(go.Scatter(x=df["date"], y=normalized, mode="lines", name=code))

    fig.update_layout(
        title="Multi-Stock Relative Performance (normalized=100)",
        height=440,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        yaxis_title="Normalized",
    )
    return fig


def _build_kline_figure(df: pd.DataFrame) -> go.Figure:
    chart_df = df.copy()
    chart_df["ma5"] = chart_df["close"].rolling(5).mean()
    chart_df["ma20"] = chart_df["close"].rolling(20).mean()
    chart_df["ma60"] = chart_df["close"].rolling(60).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="K",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["ma5"], mode="lines", name="MA5", line={"width": 1.2, "color": "#ff7f0e"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["ma20"], mode="lines", name="MA20", line={"width": 1.2, "color": "#2ca02c"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["ma60"], mode="lines", name="MA60", line={"width": 1.2, "color": "#1f77b4"}), row=1, col=1)

    if "volume" in chart_df.columns:
        fig.add_trace(
            go.Bar(x=chart_df["date"], y=chart_df["volume"], name="Volume", marker_color="#7f7f7f", opacity=0.5),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="K-line / MA / Volume",
        xaxis_rangeslider_visible=False,
        height=640,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h", "y": 1.03},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def _build_index_trend_figure(index_histories: dict) -> go.Figure:
    fig = go.Figure()
    for name, df in index_histories.items():
        if df.empty or "close" not in df.columns:
            continue
        close = pd.to_numeric(df["close"], errors="coerce")
        if close.empty:
            continue
        base = close.iloc[0]
        if base == 0 or pd.isna(base):
            continue
        normalized = close / base * 100
        fig.add_trace(go.Scatter(x=df["date"], y=normalized, mode="lines", name=name))

    fig.update_layout(
        title="Major Indices Trend (normalized=100)",
        height=360,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        yaxis_title="Normalized",
    )
    return fig


def _sentiment_from_indices(indices_df: pd.DataFrame) -> dict:
    if indices_df.empty:
        return {"score": 50.0, "label": "Neutral"}

    pctchg = pd.to_numeric(indices_df.get("pctChg"), errors="coerce").dropna()
    ret5 = pd.to_numeric(indices_df.get("return_5d_pct"), errors="coerce").dropna()
    avg_pctchg = pctchg.mean() if not pctchg.empty else 0.0
    avg_ret5 = ret5.mean() if not ret5.empty else 0.0

    score = 50 + avg_pctchg * 6 + avg_ret5 * 2
    score = max(0.0, min(100.0, float(score)))

    if score >= 70:
        label = "Risk-On"
    elif score <= 35:
        label = "Risk-Off"
    else:
        label = "Neutral"

    return {
        "score": round(score, 1),
        "label": label,
        "avg_pctchg": round(float(avg_pctchg), 2),
        "avg_ret5": round(float(avg_ret5), 2),
    }


def _show_error(raw: dict):
    if "error" in raw:
        st.error(raw["error"])
        return True
    return False


with st.sidebar:
    st.header("Controls")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.text_input("Symbol", key="active_symbol")
    st.text_input("Watchlist", key="watchlist_codes")

    default_end = date.today()
    default_start = default_end - timedelta(days=365)

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)
    frequency = st.selectbox("K frequency", options=["d", "w", "m"], index=0)

    st.divider()
    st.subheader("Backtest params")
    fast_period = st.number_input("Fast MA", min_value=2, max_value=100, value=5, step=1)
    slow_period = st.number_input("Slow MA", min_value=3, max_value=250, value=20, step=1)
    initial_capital = st.number_input("Initial capital", min_value=10000.0, value=100000.0, step=10000.0)

    st.divider()
    st.subheader("Screening params")
    universe_codes = st.text_area(
        "Universe codes (comma-separated, optional)",
        value="",
        help="Leave empty to auto sample from market.",
    )
    universe_limit = st.number_input("Universe limit", min_value=20, max_value=500, value=80, step=10)
    screen_top_n = st.number_input("Top picks", min_value=3, max_value=50, value=10, step=1)
    min_avg_amount = st.number_input("Min avg amount(20d)", min_value=0.0, value=0.0, step=10000000.0)

    st.subheader("Trader selection params")
    trader_candidate_n = st.number_input("Trader candidate N", min_value=5, max_value=80, value=20, step=1)
    trader_pick_n = st.number_input("Trader pick N", min_value=1, max_value=20, value=5, step=1)

symbol = st.session_state["active_symbol"]
watchlist_codes = st.session_state["watchlist_codes"]


market_raw = cached_market_overview()
if _show_error(market_raw):
    st.stop()

indices = pd.DataFrame(market_raw.get("indices", []))
if indices.empty:
    st.warning("No market index data.")

sentiment = _sentiment_from_indices(indices)

col_top_1, col_top_2, col_top_3, col_top_4, col_top_5 = st.columns(5)
if not indices.empty:
    show_df = indices.copy()
    if "name" in show_df.columns:
        show_df = show_df.set_index("name")

    with col_top_1:
        if "上证指数" in show_df.index:
            st.metric("SSE", f"{show_df.loc['上证指数', 'close']}", f"{show_df.loc['上证指数', 'pctChg']}%")
    with col_top_2:
        if "深证成指" in show_df.index:
            st.metric("SZSE", f"{show_df.loc['深证成指', 'close']}", f"{show_df.loc['深证成指', 'pctChg']}%")
    with col_top_3:
        if "创业板指" in show_df.index:
            st.metric("ChiNext", f"{show_df.loc['创业板指', 'close']}", f"{show_df.loc['创业板指', 'pctChg']}%")
    with col_top_4:
        avg_5d = pd.to_numeric(show_df["return_5d_pct"], errors="coerce").mean() if "return_5d_pct" in show_df.columns else 0.0
        st.metric("Market 5D avg", f"{_to_float(avg_5d):.2f}%")
with col_top_5:
    st.metric("Sentiment", sentiment["label"], f"{sentiment['score']}")


tabs = st.tabs(["Overview", "Compare", "K-line", "Technical", "Backtest", "Screening", "Sentiment(TODO+)"])

with tabs[0]:
    st.subheader("Market Overview")
    trend_start = (end_date - timedelta(days=120)).strftime("%Y-%m-%d")
    trend_end = end_date.strftime("%Y-%m-%d")

    index_histories = {}
    for idx_name, idx_code in INDEX_CODE_MAP.items():
        raw = cached_stock_history(idx_code, trend_start, trend_end, "d")
        if _show_error(raw):
            continue
        index_histories[idx_name] = _history_to_df(raw)

    st.plotly_chart(_build_index_trend_figure(index_histories), use_container_width=True)

    st.subheader("Watchlist Snapshot")
    watchlist_rows = []
    for code in _parse_codes(watchlist_codes):
        quote = cached_stock_quote(code)
        if "error" in quote:
            watchlist_rows.append({"symbol": code, "error": quote["error"]})
            continue
        watchlist_rows.append(
            {
                "symbol": quote.get("symbol", code),
                "code": quote.get("code"),
                "date": quote.get("date"),
                "close": quote.get("close"),
                "pctChg": quote.get("pctChg"),
                "volume": quote.get("volume"),
            }
        )
    if watchlist_rows:
        st.dataframe(pd.DataFrame(watchlist_rows), use_container_width=True)
    else:
        st.info("Add watchlist symbols in sidebar.")

with tabs[1]:
    st.subheader("Multi-Stock Compare")
    compare_codes_input = st.text_input("Compare symbols", value=watchlist_codes, key="compare_codes")
    compare_codes = _parse_codes(compare_codes_input)

    compare_curves = {}
    compare_rows = []
    for code in compare_codes:
        raw = cached_stock_history(code, str(start_date), str(end_date), "d")
        if "error" in raw:
            compare_rows.append({"code": code, "error": raw["error"]})
            continue

        df = _history_to_df(raw)
        if df.empty:
            compare_rows.append({"code": code, "error": "empty_history"})
            continue

        compare_curves[code] = df
        stats = _compute_curve_stats(df)
        compare_rows.append({"code": code, **stats})

    if compare_curves:
        st.plotly_chart(_build_compare_figure(compare_curves), use_container_width=True)

    if compare_rows:
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)
    else:
        st.info("Input at least one symbol to compare.")

with tabs[2]:
    hist_raw = cached_stock_history(symbol, str(start_date), str(end_date), frequency)
    if not _show_error(hist_raw):
        hist_df = _history_to_df(hist_raw)
        if hist_df.empty:
            st.warning("No history data.")
        else:
            st.plotly_chart(_build_kline_figure(hist_df), use_container_width=True)
            st.dataframe(hist_df.tail(30), use_container_width=True)

with tabs[3]:
    tech_raw = cached_technical(symbol, str(start_date), str(end_date))
    if not _show_error(tech_raw):
        perf_cols = st.columns(4)
        perf_cols[0].metric("Close", tech_raw.get("close"))
        perf_cols[1].metric("RSI", tech_raw.get("RSI"))
        perf_cols[2].metric("MACD_HIST", tech_raw.get("MACD_HIST"))
        perf_cols[3].metric("Vol annual", f"{tech_raw.get('volatility_annual_pct')}%")

        with st.expander("Technical JSON", expanded=False):
            st.json(tech_raw)

with tabs[4]:
    if fast_period >= slow_period:
        st.warning("Fast MA should be smaller than Slow MA.")
    else:
        bt_raw = cached_backtest(
            symbol,
            str(start_date),
            str(end_date),
            int(fast_period),
            int(slow_period),
            float(initial_capital),
        )
        if not _show_error(bt_raw):
            bt_perf = bt_raw.get("performance", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total return", f"{bt_perf.get('total_return_pct')}%")
            c2.metric("Annual return", f"{bt_perf.get('annual_return_pct')}%")
            c3.metric("Max drawdown", f"{bt_perf.get('max_drawdown_pct')}%")
            c4.metric("Sharpe", bt_perf.get("sharpe"))

            curve = pd.DataFrame(bt_raw.get("equity_curve_tail", []))
            if not curve.empty:
                curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
                st.line_chart(curve.set_index("date")["equity"], use_container_width=True)

            st.subheader("Recent trades")
            st.dataframe(pd.DataFrame(bt_raw.get("trades", [])), use_container_width=True)

        st.subheader("Grid search")
        fast_candidates = st.text_input("Fast candidates", value="5,8,10,12", key="grid_fast")
        slow_candidates = st.text_input("Slow candidates", value="20,30,40,60", key="grid_slow")
        grid_top_n = st.slider("Grid top N", min_value=3, max_value=20, value=5)

        if st.button("Run grid search", type="primary"):
            grid_raw = cached_grid_search(
                symbol,
                str(start_date),
                str(end_date),
                fast_candidates,
                slow_candidates,
                grid_top_n,
            )
            if not _show_error(grid_raw):
                st.json(grid_raw.get("best_parameters", {}))
                st.dataframe(pd.DataFrame(grid_raw.get("top_results", [])), use_container_width=True)

with tabs[5]:
    btn_col_1, btn_col_2 = st.columns(2)
    if btn_col_1.button("Run screening", type="primary", key="run_screening_btn"):
        st.session_state["screen_result"] = cached_screening(
            str(start_date),
            str(end_date),
            universe_codes,
            int(universe_limit),
            int(screen_top_n),
            float(min_avg_amount),
        )
    if btn_col_2.button("Generate trader plan", type="primary", key="run_trader_plan_btn"):
        st.session_state["trader_select_result"] = cached_trader_selection(
            str(start_date),
            str(end_date),
            universe_codes,
            int(universe_limit),
            int(trader_candidate_n),
            int(trader_pick_n),
            float(min_avg_amount),
            int(fast_period),
            int(slow_period),
            float(initial_capital),
        )

    screen_raw = st.session_state.get("screen_result")
    if screen_raw and not _show_error(screen_raw):
        st.subheader("Top picks")
        top_df = pd.DataFrame(screen_raw.get("top_picks", []))
        st.dataframe(top_df, use_container_width=True)

        if not top_df.empty and "score" in top_df.columns and "code" in top_df.columns:
            rank_fig = go.Figure()
            rank_fig.add_trace(go.Bar(x=top_df["code"], y=top_df["score"], name="Score"))
            rank_fig.update_layout(title="Top Picks Score", height=360, margin={"l": 20, "r": 20, "t": 45, "b": 20})
            st.plotly_chart(rank_fig, use_container_width=True)

            default_selected = top_df["code"].astype(str).head(min(3, len(top_df))).tolist()
            selected_codes = st.multiselect(
                "Select picks for watchlist/backtest",
                options=top_df["code"].astype(str).tolist(),
                default=default_selected,
                key="screen_selected_codes",
            )

            left_col, right_col = st.columns(2)
            if left_col.button("Add selected to watchlist", key="add_selected_watchlist"):
                updated = _merge_codes(_parse_codes(st.session_state["watchlist_codes"]), selected_codes)
                st.session_state["watchlist_codes"] = ",".join(updated)
                if selected_codes:
                    st.session_state["active_symbol"] = selected_codes[0]
                st.success("Selected symbols added to watchlist.")

            if right_col.button("Backtest first selected", key="backtest_selected_pick"):
                if not selected_codes:
                    st.warning("Please select at least one symbol.")
                else:
                    target = selected_codes[0]
                    st.session_state["active_symbol"] = target
                    bt_pick_raw = cached_backtest(
                        target,
                        str(start_date),
                        str(end_date),
                        int(fast_period),
                        int(slow_period),
                        float(initial_capital),
                    )
                    if not _show_error(bt_pick_raw):
                        st.markdown(f"**Backtest result for {target}**")
                        bt_perf = bt_pick_raw.get("performance", {})
                        p1, p2, p3, p4 = st.columns(4)
                        p1.metric("Total return", f"{bt_perf.get('total_return_pct')}%")
                        p2.metric("Annual return", f"{bt_perf.get('annual_return_pct')}%")
                        p3.metric("Max drawdown", f"{bt_perf.get('max_drawdown_pct')}%")
                        p4.metric("Sharpe", bt_perf.get("sharpe"))

        st.subheader("Universe summary")
        st.json(screen_raw.get("universe", {}))

    trader_raw = st.session_state.get("trader_select_result")
    if trader_raw and not _show_error(trader_raw):
        st.subheader("Trader plan")
        regime = trader_raw.get("market_regime", {})
        regime_cols = st.columns(4)
        regime_cols[0].metric("Regime", regime.get("label", "unknown"))
        regime_cols[1].metric("Regime score", regime.get("score"))
        regime_cols[2].metric("Avg day pctChg", f"{regime.get('avg_pctChg', 0)}%")
        regime_cols[3].metric("Avg 5D return", f"{regime.get('avg_return_5d_pct', 0)}%")

        trader_top_df = pd.DataFrame(trader_raw.get("top_picks", []))
        st.dataframe(trader_top_df, use_container_width=True)

        if not trader_top_df.empty and "code" in trader_top_df.columns:
            plan_selected = st.multiselect(
                "Select trader picks for watchlist/backtest",
                options=trader_top_df["code"].astype(str).tolist(),
                default=trader_top_df["code"].astype(str).head(min(3, len(trader_top_df))).tolist(),
                key="trader_plan_selected_codes",
            )

            p_left, p_right = st.columns(2)
            if p_left.button("Add trader picks to watchlist", key="add_trader_watchlist"):
                updated = _merge_codes(_parse_codes(st.session_state["watchlist_codes"]), plan_selected)
                st.session_state["watchlist_codes"] = ",".join(updated)
                if plan_selected:
                    st.session_state["active_symbol"] = plan_selected[0]
                st.success("Trader picks added to watchlist.")

            if p_right.button("Backtest first trader pick", key="backtest_trader_pick"):
                if not plan_selected:
                    st.warning("Please select at least one symbol.")
                else:
                    target = plan_selected[0]
                    bt_pick_raw = cached_backtest(
                        target,
                        str(start_date),
                        str(end_date),
                        int(fast_period),
                        int(slow_period),
                        float(initial_capital),
                    )
                    if not _show_error(bt_pick_raw):
                        st.markdown(f"**Backtest result for {target}**")
                        bt_perf = bt_pick_raw.get("performance", {})
                        p1, p2, p3, p4 = st.columns(4)
                        p1.metric("Total return", f"{bt_perf.get('total_return_pct')}%")
                        p2.metric("Annual return", f"{bt_perf.get('annual_return_pct')}%")
                        p3.metric("Max drawdown", f"{bt_perf.get('max_drawdown_pct')}%")
                        p4.metric("Sharpe", bt_perf.get("sharpe"))

        with st.expander("Trader plan raw JSON", expanded=False):
            st.json(trader_raw)

with tabs[6]:
    st.subheader("Sentiment Prototype")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=sentiment["score"],
            title={"text": f"Market sentiment: {sentiment['label']}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 35], "color": "#f2b6b6"},
                    {"range": [35, 70], "color": "#f7f0c2"},
                    {"range": [70, 100], "color": "#b8e6c0"},
                ],
            },
        )
    )
    gauge.update_layout(height=320, margin={"l": 20, "r": 20, "t": 45, "b": 20})
    st.plotly_chart(gauge, use_container_width=True)

    st.info("Current sentiment is a lightweight prototype based on major-index momentum only.")
    st.markdown(
        """
- TODO 1: Add limit-up/limit-down breadth and turnover concentration.
- TODO 2: Add northbound flow and margin financing indicators.
- TODO 3: Add social/news sentiment factors.
- TODO 4: Train a composite sentiment index with rolling calibration.
        """
    )
