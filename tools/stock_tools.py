# -*- coding: utf-8 -*-
"""
股票数据工具层：基于 baostock 获取 A 股行情、历史 K 线、股票搜索。
支持技术指标与量化分析（MA、MACD、RSI、布林带等），为投资建议提供数据基础。
"""
from datetime import datetime, timedelta
import os

import baostock as bs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "plots")


# ---------- 证券代码规范化 ----------
def _normalize_symbol(symbol: str) -> str:
    """
    将用户输入的代码转为 baostock 格式。
    6 开头 -> sh.xxxxxx，0/3 开头 -> sz.xxxxxx。
    """
    s = str(symbol).strip().split(".")[0]
    if not s.isdigit():
        return symbol.strip()
    if s.startswith("6"):
        return f"sh.{s}"
    if s.startswith(("0", "3")):
        return f"sz.{s}"
    return symbol.strip()


def _ensure_login():
    """确保已登录，用于在工具内统一调用。"""
    lg = bs.login()
    return lg.error_code == "0"


def _ensure_logout():
    bs.logout()


def _ensure_plot_dir() -> str:
    os.makedirs(_PLOT_DIR, exist_ok=True)
    return _PLOT_DIR


def _slugify_symbol(symbol: str) -> str:
    text = str(symbol or "unknown").strip().replace(".", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _save_plotly_html(fig, symbol: str, chart_name: str) -> str:
    folder = _ensure_plot_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_slugify_symbol(symbol)}_{chart_name}_{ts}.html"
    path = os.path.join(folder, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def _history_records_to_df(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pctChg"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date", ascending=True).reset_index(drop=True)


# ---------- 工具 1：最新行情（取最近交易日日线） ----------
def get_stock_quote(symbol: str) -> dict:
    """
    获取单只股票最新行情（最近一个交易日的 OHLC、涨跌幅等）。
    :param symbol: 股票代码，如 600000、000001、sh.600000
    :return: 包含 date, open, high, low, close, volume, pctChg 等字段的 dict；失败返回 error 信息
    """
    print("获取最新行情...")
    code = _normalize_symbol(symbol)
    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        end = datetime.now()
        start = end - timedelta(days=30)
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount,pctChg",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="3",
        )
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未获取到 {symbol} 的行情数据，请检查代码或日期。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=True)
        latest = df.iloc[-1]

        return {
            "symbol": symbol,
            "code": code,
            "date": str(latest["date"].date()) if hasattr(latest["date"], "date") else str(latest["date"]),
            "open": latest["open"],
            "high": latest["high"],
            "low": latest["low"],
            "close": latest["close"],
            "volume": latest["volume"],
            "amount": latest["amount"],
            "pctChg": latest["pctChg"],
        }
    except Exception as e:
        _ensure_logout()
        return {"error": f"获取行情异常: {str(e)}"}


# ---------- 工具 2：历史 K 线 ----------
def get_stock_history(
    symbol: str,
    start_date: str,
    end_date: str,
    frequency: str = "d",
) -> dict:
    """
    获取股票历史 K 线。
    :param symbol: 股票代码
    :param start_date: 开始日期 yyyy-mm-dd
    :param end_date: 结束日期 yyyy-mm-dd
    :param frequency: 频率 d=日k w=周 m=月，默认 d
    :return: 列表形式的 K 线数据；失败返回 error
    """
    print(f"获取历史K线数据... {symbol} {start_date} ~ {end_date}")
    code = _normalize_symbol(symbol)
    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag="3",
        )
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未获取到 {symbol} 在 {start_date}~{end_date} 的历史数据。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        df = df.sort_values("date", ascending=True)
        records = df.to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if pd.isna(v):
                    r[k] = None
                elif isinstance(v, (pd.Timestamp, datetime)):
                    r[k] = str(v)[:10]
                else:
                    r[k] = str(v) if v is not None else None

        return {"symbol": symbol, "code": code, "count": len(records), "data": records}
    except Exception as e:
        _ensure_logout()
        return {"error": f"获取历史数据异常: {str(e)}"}


# ---------- 工具 3：股票搜索（名称/代码模糊） ----------
def search_stock(keyword: str) -> dict:
    """
    按股票名称或代码模糊搜索。
    :param keyword: 名称或代码关键词，如 浦发、6000
    :return: 匹配的股票列表，每项含 code, code_name；失败返回 error
    """
    print(f"搜索股票... {keyword}")
    keyword = (keyword or "").strip()
    if not keyword:
        return {"error": "请输入股票名称或代码关键词。"}

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        # 按名称模糊查询
        rs = bs.query_stock_basic(code_name=keyword)
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"搜索失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"keyword": keyword, "count": 0, "list": []}

        df = pd.DataFrame(rows, columns=rs.fields)
        # 只保留股票类型，过滤指数等
        if "type" in df.columns:
            df = df[df["type"] == "1"]
        if "status" in df.columns:
            df = df[df["status"] == "1"]

        records = df[["code", "code_name"]].drop_duplicates().head(20)
        list_ = [{"code": r["code"], "code_name": r["code_name"]} for _, r in records.iterrows()]
        return {"keyword": keyword, "count": len(list_), "list": list_}
    except Exception as e:
        _ensure_logout()
        return {"error": f"搜索异常: {str(e)}"}


# ---------- 工具 4：股票基本信息 ----------
def get_stock_basic(symbol: str) -> dict:
    """
    获取股票基本信息（名称、上市日期、类型等）。
    :param symbol: 股票代码
    :return: 基本信息 dict；失败返回 error
    """
    print(f"获取股票基本信息... {symbol}")
    code = _normalize_symbol(symbol)
    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        rs = bs.query_stock_basic(code=code)
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未找到代码 {symbol} 对应的证券信息。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        row = df.iloc[0].to_dict()
        for k, v in row.items():
            if pd.isna(v):
                row[k] = None
            elif hasattr(v, "item"):
                row[k] = v.item()
            else:
                row[k] = str(v)
        return {"symbol": symbol, "code": code, **row}
    except Exception as e:
        _ensure_logout()
        return {"error": f"获取基本信息异常: {str(e)}"}


# ---------- 工具 5：大盘概览（上证、深证、创业板） ----------
_MAJOR_INDICES = [
    ("sh.000001", "上证指数"),
    ("sz.399001", "深证成指"),
    ("sz.399006", "创业板指"),
]


def get_market_overview() -> dict:
    """
    获取主要股指（上证指数、深证成指、创业板指）最新行情，用于分析个股时结合大盘环境。
    """
    print("获取大盘概览...")
    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        end = datetime.now()
        start = end - timedelta(days=30)
        indices_data = []

        for code, name in _MAJOR_INDICES:
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,amount,pctChg",
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="1",  # 指数不复权
            )
            if rs.error_code != "0":
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=rs.fields)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date", ascending=True)
            latest = df.iloc[-1]
            close = pd.to_numeric(latest["close"], errors="coerce")
            pct = pd.to_numeric(latest["pctChg"], errors="coerce")
            pct_5d = None
            if len(df) >= 5:
                close_5d = pd.to_numeric(df["close"].iloc[-5], errors="coerce")
                if not pd.isna(close) and not pd.isna(close_5d) and close_5d != 0:
                    pct_5d = round((float(close) / float(close_5d) - 1) * 100, 2)
            indices_data.append({
                "name": name,
                "code": code,
                "date": str(latest["date"])[:10],
                "close": round(float(close), 2) if not pd.isna(close) else None,
                "pctChg": round(float(pct), 2) if not pd.isna(pct) else None,
                "return_5d_pct": pct_5d,
            })
        _ensure_logout()

        if not indices_data:
            return {"error": "未获取到大盘指数数据。"}
        return {"indices": indices_data}
    except Exception as e:
        _ensure_logout()
        return {"error": f"获取大盘数据异常: {str(e)}"}


# ---------- 技术指标计算（纯 pandas/numpy，无外部 TA 库） ----------
def _ensure_numeric(series):
    """将 series 转为 float64。"""
    return pd.to_numeric(series, errors="coerce")


def _compute_ma(df: pd.DataFrame) -> pd.DataFrame:
    """MA5/10/20/60"""
    close = _ensure_numeric(df["close"])
    df = df.copy()
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    return df


def _compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """MACD 及信号线、柱状图"""
    close = _ensure_numeric(df["close"])
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    df = df.copy()
    df["MACD_DIF"] = ema_fast - ema_slow
    df["MACD_DEA"] = df["MACD_DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD_HIST"] = (df["MACD_DIF"] - df["MACD_DEA"]) * 2
    return df


def _compute_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """RSI"""
    close = _ensure_numeric(df["close"])
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df = df.copy()
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def _compute_bollinger(df: pd.DataFrame, period=20, std_mult=2) -> pd.DataFrame:
    """布林带"""
    close = _ensure_numeric(df["close"])
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    df = df.copy()
    df["BB_MID"] = ma
    df["BB_UPPER"] = ma + std_mult * std
    df["BB_LOWER"] = ma - std_mult * std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / (df["BB_MID"] + 1e-10) * 100
    return df


def _compute_volatility_and_trend(df: pd.DataFrame) -> dict:
    """波动率、近 5/20 日涨跌幅、MA20 斜率（趋势强度）。"""
    close = _ensure_numeric(df["close"]).dropna()
    if len(close) < 5:
        return {}
    ret = close.pct_change().dropna()
    vol_20 = ret.tail(20).std() * (252**0.5) * 100 if len(ret) >= 20 else ret.std() * (252**0.5) * 100
    pct_5 = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
    pct_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
    ma20 = close.rolling(20).mean()
    slope = ((ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100) if len(ma20.dropna()) >= 5 else 0
    return {
        "volatility_annual_pct": round(float(vol_20), 2) if not np.isnan(vol_20) else None,
        "return_5d_pct": round(float(pct_5), 2),
        "return_20d_pct": round(float(pct_20), 2) if len(close) >= 20 else None,
        "ma20_slope_5d_pct": round(float(slope), 2) if not np.isnan(slope) else None,
    }


def _build_technical_plot(symbol: str, df: pd.DataFrame) -> dict:
    """绘制技术分析常用曲线并保存为 HTML。"""
    if df.empty or "date" not in df.columns:
        return {"error": "无可绘制数据。"}

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.18, 0.2, 0.17],
        subplot_titles=("K线 + MA + 布林带", "成交量", "MACD", "RSI"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
        ),
        row=1,
        col=1,
    )
    for col, color in [("MA5", "#ff7f0e"), ("MA10", "#2ca02c"), ("MA20", "#1f77b4"), ("MA60", "#d62728")]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["date"], y=df[col], mode="lines", name=col, line={"width": 1.2, "color": color}),
                row=1,
                col=1,
            )

    for col, color, dash in [("BB_UPPER", "#9467bd", "dot"), ("BB_MID", "#8c564b", "dash"), ("BB_LOWER", "#9467bd", "dot")]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["date"], y=df[col], mode="lines", name=col, line={"width": 1.0, "color": color, "dash": dash}),
                row=1,
                col=1,
            )

    if "volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df["date"], y=df["volume"], name="Volume", marker_color="#7f7f7f", opacity=0.5),
            row=2,
            col=1,
        )

    if "MACD_HIST" in df.columns:
        hist_colors = ["#d62728" if x >= 0 else "#2ca02c" for x in df["MACD_HIST"].fillna(0)]
        fig.add_trace(
            go.Bar(x=df["date"], y=df["MACD_HIST"], name="MACD_HIST", marker_color=hist_colors, opacity=0.55),
            row=3,
            col=1,
        )
    if "MACD_DIF" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD_DIF"], mode="lines", name="MACD_DIF", line={"color": "#1f77b4"}), row=3, col=1)
    if "MACD_DEA" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD_DEA"], mode="lines", name="MACD_DEA", line={"color": "#ff7f0e"}), row=3, col=1)

    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["RSI"], mode="lines", name="RSI", line={"color": "#1f77b4"}), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#d62728", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#2ca02c", row=4, col=1)

    fig.update_layout(
        title=f"{symbol} 技术分析曲线",
        height=1080,
        xaxis_rangeslider_visible=False,
        margin={"l": 28, "r": 20, "t": 64, "b": 24},
        legend={"orientation": "h", "y": 1.03},
    )

    path = _save_plotly_html(fig, symbol=symbol, chart_name="technical")
    return {
        "chart_type": "technical_overview",
        "path": path,
        "points": int(len(df)),
    }


def _build_backtest_plot(symbol: str, df: pd.DataFrame, curve_df: pd.DataFrame, trades: list, initial_capital: float) -> dict:
    """绘制回测曲线：价格/均线与交易点、策略净值 vs 基准、回撤。"""
    if df.empty or curve_df.empty:
        return {"error": "无可绘制的回测数据。"}

    bt_df = df.copy()
    bt_df["date"] = pd.to_datetime(bt_df["date"], errors="coerce")
    curve = curve_df.copy()
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")

    running_max = curve["equity"].cummax()
    curve["drawdown_pct"] = (curve["equity"] / running_max - 1.0) * 100

    first_close = float(bt_df["close"].iloc[0]) if len(bt_df) > 0 else 0.0
    if first_close > 0:
        curve["benchmark_equity"] = bt_df["close"] / first_close * float(initial_capital)
    else:
        curve["benchmark_equity"] = np.nan

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.4, 0.35, 0.25],
        subplot_titles=("价格与均线(含交易点)", "策略净值 vs 买入持有", "回撤曲线"),
    )

    fig.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["close"], mode="lines", name="Close", line={"color": "#111111"}), row=1, col=1)
    if "fast_ma" in bt_df.columns:
        fig.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["fast_ma"], mode="lines", name="Fast MA", line={"color": "#ff7f0e", "width": 1.2}), row=1, col=1)
    if "slow_ma" in bt_df.columns:
        fig.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["slow_ma"], mode="lines", name="Slow MA", line={"color": "#1f77b4", "width": 1.2}), row=1, col=1)

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df["date"] = pd.to_datetime(trades_df["date"], errors="coerce")
        for side, color, symbol_mark in [("BUY", "#d62728", "triangle-up"), ("SELL", "#2ca02c", "triangle-down")]:
            sub = trades_df[trades_df["side"] == side]
            if not sub.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sub["date"],
                        y=sub["price"],
                        mode="markers",
                        name=side,
                        marker={"color": color, "symbol": symbol_mark, "size": 10},
                    ),
                    row=1,
                    col=1,
                )

    fig.add_trace(go.Scatter(x=curve["date"], y=curve["equity"], mode="lines", name="Strategy Equity", line={"color": "#1f77b4"}), row=2, col=1)
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["benchmark_equity"], mode="lines", name="Buy&Hold Equity", line={"color": "#ff7f0e", "dash": "dash"}), row=2, col=1)
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["drawdown_pct"], mode="lines", name="Drawdown %", line={"color": "#d62728"}), row=3, col=1)

    fig.update_layout(
        title=f"{symbol} 双均线回测曲线",
        height=920,
        margin={"l": 28, "r": 20, "t": 60, "b": 24},
        legend={"orientation": "h", "y": 1.03},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)

    path = _save_plotly_html(fig, symbol=symbol, chart_name="backtest")
    return {
        "chart_type": "backtest_overview",
        "path": path,
        "points": int(len(curve)),
    }


def _build_cross_section_plots(
    symbol_tag: str,
    ranked_df: pd.DataFrame,
    history_map: dict,
    score_col: str,
    return_col: str,
    risk_col: str,
    title_prefix: str,
) -> list:
    """生成横向对比图：评分排行、风险收益散点、相对净值曲线。"""
    plots = []
    if ranked_df is None or ranked_df.empty:
        return plots

    work_df = ranked_df.copy().reset_index(drop=True)
    if "code" not in work_df.columns:
        return plots

    work_df["label"] = work_df["code"].astype(str)
    if "code_name" in work_df.columns:
        names = work_df["code_name"].fillna("").astype(str)
        work_df["label"] = [f"{c} {n}".strip() for c, n in zip(work_df["code"], names)]

    if score_col in work_df.columns:
        rank_view = work_df.head(15).copy()
        fig_rank = go.Figure()
        fig_rank.add_trace(
            go.Bar(
                x=rank_view["label"],
                y=pd.to_numeric(rank_view[score_col], errors="coerce"),
                marker_color="#1f77b4",
                name=score_col,
            )
        )
        fig_rank.update_layout(
            title=f"{title_prefix}评分排行",
            xaxis_title="标的",
            yaxis_title=score_col,
            height=520,
            margin={"l": 24, "r": 20, "t": 56, "b": 120},
        )
        fig_rank.update_xaxes(tickangle=-25)
        plots.append({
            "chart_type": "score_ranking",
            "path": _save_plotly_html(fig_rank, symbol=symbol_tag, chart_name="score_ranking"),
            "points": int(len(rank_view)),
        })

    if return_col in work_df.columns and risk_col in work_df.columns:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=pd.to_numeric(work_df[risk_col], errors="coerce"),
                y=pd.to_numeric(work_df[return_col], errors="coerce"),
                text=work_df["label"],
                mode="markers+text",
                textposition="top center",
                marker={
                    "size": 10,
                    "color": pd.to_numeric(work_df.get(score_col, 0), errors="coerce"),
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": score_col},
                },
                name="risk_return",
            )
        )
        fig_scatter.update_layout(
            title=f"{title_prefix}风险收益散点图",
            xaxis_title=risk_col,
            yaxis_title=return_col,
            height=520,
            margin={"l": 24, "r": 24, "t": 56, "b": 24},
        )
        plots.append({
            "chart_type": "risk_return_scatter",
            "path": _save_plotly_html(fig_scatter, symbol=symbol_tag, chart_name="risk_return"),
            "points": int(len(work_df)),
        })

    fig_curve = go.Figure()
    curve_count = 0
    for _, row in work_df.head(10).iterrows():
        code = str(row["code"])
        hist_df = history_map.get(code)
        if hist_df is None or hist_df.empty or "date" not in hist_df.columns or "close" not in hist_df.columns:
            continue
        close = pd.to_numeric(hist_df["close"], errors="coerce")
        if close.empty or pd.isna(close.iloc[0]) or float(close.iloc[0]) == 0:
            continue
        normalized = close / float(close.iloc[0]) * 100
        label = row.get("label", code)
        fig_curve.add_trace(go.Scatter(x=hist_df["date"], y=normalized, mode="lines", name=str(label)))
        curve_count += 1

    if curve_count > 0:
        fig_curve.update_layout(
            title=f"{title_prefix}Top标的相对净值曲线(起点=100)",
            xaxis_title="Date",
            yaxis_title="Normalized",
            height=520,
            margin={"l": 24, "r": 20, "t": 56, "b": 24},
            legend={"orientation": "h", "y": 1.05},
        )
        plots.append({
            "chart_type": "relative_performance",
            "path": _save_plotly_html(fig_curve, symbol=symbol_tag, chart_name="relative_curve"),
            "points": int(curve_count),
        })

    return plots


# ---------- 工具 5：技术指标与量化分析 ----------
def get_technical_analysis(
    symbol: str,
    start_date: str,
    end_date: str,
    with_plots: bool = False,
) -> dict:
    """
    基于历史 K 线计算技术指标，用于量化分析与投资建议。
    返回：MA、MACD、RSI、布林带、波动率、近 5/20 日涨跌幅等。
    """
    print(f"计算技术指标分析... {symbol} {start_date} ~ {end_date}")
    raw = get_stock_history(symbol, start_date, end_date, frequency="d")
    if "error" in raw:
        return raw

    df = _history_records_to_df(raw["data"])

    close = _ensure_numeric(df["close"])
    if close.isna().all() or len(close.dropna()) < 5:
        return {"error": f"历史数据不足，无法计算技术指标。需要至少 5 个交易日。"}

    df = _compute_ma(df)
    df = _compute_macd(df)
    df = _compute_rsi(df)
    df = _compute_bollinger(df)

    latest = df.iloc[-1]
    metrics = {
        "symbol": symbol,
        "code": raw.get("code", _normalize_symbol(symbol)),
        "date_range": f"{start_date} ~ {end_date}",
        "latest_date": str(latest["date"])[:10],
        "close": float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else None,
        "MA5": round(float(latest["MA5"]), 2) if pd.notna(latest.get("MA5")) else None,
        "MA10": round(float(latest["MA10"]), 2) if pd.notna(latest.get("MA10")) else None,
        "MA20": round(float(latest["MA20"]), 2) if pd.notna(latest.get("MA20")) else None,
        "MA60": round(float(latest["MA60"]), 2) if pd.notna(latest.get("MA60")) else None,
        "MACD_DIF": round(float(latest["MACD_DIF"]), 4) if pd.notna(latest.get("MACD_DIF")) else None,
        "MACD_DEA": round(float(latest["MACD_DEA"]), 4) if pd.notna(latest.get("MACD_DEA")) else None,
        "MACD_HIST": round(float(latest["MACD_HIST"]), 4) if pd.notna(latest.get("MACD_HIST")) else None,
        "RSI": round(float(latest["RSI"]), 1) if pd.notna(latest.get("RSI")) else None,
        "BB_UPPER": round(float(latest["BB_UPPER"]), 2) if pd.notna(latest.get("BB_UPPER")) else None,
        "BB_MID": round(float(latest["BB_MID"]), 2) if pd.notna(latest.get("BB_MID")) else None,
        "BB_LOWER": round(float(latest["BB_LOWER"]), 2) if pd.notna(latest.get("BB_LOWER")) else None,
        **_compute_volatility_and_trend(df),
    }

    if with_plots:
        plot_result = _build_technical_plot(symbol=raw.get("code", _normalize_symbol(symbol)), df=df)
        metrics["plots"] = [plot_result]

    return metrics


def backtest_moving_average_strategy(
    symbol: str,
    start_date: str,
    end_date: str,
    fast_period: int = 5,
    slow_period: int = 20,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    tax_rate: float = 0.001,
    with_plots: bool = False,
) -> dict:
    """
    简单双均线回测引擎（A 股日线场景）。

    规则：
    1) 使用收盘价计算信号，次日开盘执行，避免未来函数。
    2) fast MA 上穿 slow MA 后持有；下穿后清仓。
    3) 交易按 100 股整数倍下单，卖出收取印花税，买入不收印花税。
    4) 约束 T+1：买入当日不可卖出。
    """
    print(
        "运行回测... {} {} ~ {} MA({},{})".format(
            symbol, start_date, end_date, fast_period, slow_period
        )
    )

    if fast_period <= 0 or slow_period <= 0:
        return {"error": "fast_period 与 slow_period 必须为正整数。"}
    if fast_period >= slow_period:
        return {"error": "fast_period 必须小于 slow_period。"}
    if initial_capital <= 0:
        return {"error": "initial_capital 必须大于 0。"}

    raw = get_stock_history(symbol, start_date, end_date, frequency="d")
    if "error" in raw:
        return raw

    df = _history_records_to_df(raw["data"])
    if df.empty:
        return {"error": "回测数据为空。"}

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
    if len(df) <= slow_period + 2:
        return {"error": "样本长度不足，建议至少提供 slow_period + 20 个交易日。"}

    df["fast_ma"] = df["close"].rolling(fast_period).mean()
    df["slow_ma"] = df["close"].rolling(slow_period).mean()
    df["signal"] = (df["fast_ma"] > df["slow_ma"]).astype(int)
    # 用 t 日收盘信号在 t+1 日开盘执行。
    df["exec_signal"] = df["signal"].shift(1).fillna(0).astype(int)

    cash = float(initial_capital)
    shares = 0
    last_buy_date = None
    equity_curve = []
    trades = []
    roundtrip_returns = []
    buy_trade = None

    for _, row in df.iterrows():
        current_date = row["date"]
        open_price = float(row["open"])
        close_price = float(row["close"])
        target_hold = int(row["exec_signal"])

        # 买入：空仓且目标持有。
        if target_hold == 1 and shares == 0 and open_price > 0:
            lot_size = 100
            shares_to_buy = int(cash / (open_price * (1 + fee_rate + slippage_rate)) / lot_size) * lot_size
            if shares_to_buy > 0:
                gross = shares_to_buy * open_price
                fee = gross * fee_rate
                slippage = gross * slippage_rate
                total_cost = gross + fee + slippage
                cash -= total_cost
                shares = shares_to_buy
                last_buy_date = current_date
                buy_trade = {
                    "date": str(current_date.date()),
                    "side": "BUY",
                    "price": round(open_price, 4),
                    "shares": int(shares_to_buy),
                    "fee": round(float(fee), 4),
                    "slippage": round(float(slippage), 4),
                    "tax": 0.0,
                    "cash_after": round(float(cash), 2),
                }
                trades.append(buy_trade)

        # 卖出：持仓且目标空仓，同时满足 T+1。
        elif target_hold == 0 and shares > 0 and open_price > 0:
            if last_buy_date is None or current_date > last_buy_date:
                gross = shares * open_price
                fee = gross * fee_rate
                slippage = gross * slippage_rate
                tax = gross * tax_rate
                net = gross - fee - slippage - tax
                cash += net
                sell_trade = {
                    "date": str(current_date.date()),
                    "side": "SELL",
                    "price": round(open_price, 4),
                    "shares": int(shares),
                    "fee": round(float(fee), 4),
                    "slippage": round(float(slippage), 4),
                    "tax": round(float(tax), 4),
                    "cash_after": round(float(cash), 2),
                }
                trades.append(sell_trade)

                if buy_trade and buy_trade.get("price", 0) > 0:
                    buy_cost_price = buy_trade["price"] * (1 + fee_rate + slippage_rate)
                    sell_net_price = open_price * (1 - fee_rate - slippage_rate - tax_rate)
                    rt_ret = sell_net_price / buy_cost_price - 1
                    roundtrip_returns.append(rt_ret)

                shares = 0
                last_buy_date = None
                buy_trade = None

        equity = cash + shares * close_price
        equity_curve.append({
            "date": str(current_date.date()),
            "equity": float(equity),
            "cash": float(cash),
            "shares": int(shares),
            "close": float(close_price),
        })

    curve_df = pd.DataFrame(equity_curve)
    curve_df["ret"] = curve_df["equity"].pct_change().fillna(0.0)

    final_equity = float(curve_df["equity"].iloc[-1])
    total_return = final_equity / float(initial_capital) - 1
    trading_days = len(curve_df)
    annual_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1

    running_max = curve_df["equity"].cummax()
    drawdown = curve_df["equity"] / running_max - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    ret_std = float(curve_df["ret"].std())
    sharpe = 0.0
    if ret_std > 0:
        sharpe = float(curve_df["ret"].mean()) / ret_std * np.sqrt(252)

    wins = sum(1 for r in roundtrip_returns if r > 0)
    total_roundtrips = len(roundtrip_returns)
    win_rate = wins / total_roundtrips if total_roundtrips > 0 else None

    benchmark_return = None
    if len(df) > 1:
        first_close = float(df["close"].iloc[0])
        last_close = float(df["close"].iloc[-1])
        if first_close > 0:
            benchmark_return = last_close / first_close - 1

    result = {
        "symbol": symbol,
        "code": raw.get("code", _normalize_symbol(symbol)),
        "strategy": "double_moving_average",
        "date_range": f"{start_date} ~ {end_date}",
        "parameters": {
            "fast_period": int(fast_period),
            "slow_period": int(slow_period),
            "initial_capital": float(initial_capital),
            "fee_rate": float(fee_rate),
            "slippage_rate": float(slippage_rate),
            "tax_rate": float(tax_rate),
            "lot_size": 100,
            "execution": "signal_on_close_execute_on_next_open",
            "t_plus_one": True,
        },
        "performance": {
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return * 100, 2),
            "annual_return_pct": round(annual_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe": round(sharpe, 3),
            "benchmark_buy_hold_return_pct": round(benchmark_return * 100, 2) if benchmark_return is not None else None,
            "excess_return_pct": round((total_return - benchmark_return) * 100, 2) if benchmark_return is not None else None,
            "trading_days": int(trading_days),
            "trade_count": int(len(trades)),
            "roundtrip_count": int(total_roundtrips),
            "win_rate_pct": round(win_rate * 100, 2) if win_rate is not None else None,
        },
        "latest_state": {
            "latest_date": curve_df["date"].iloc[-1],
            "latest_equity": round(float(curve_df["equity"].iloc[-1]), 2),
            "latest_cash": round(float(curve_df["cash"].iloc[-1]), 2),
            "latest_shares": int(curve_df["shares"].iloc[-1]),
        },
        "trades": trades[-20:],
        "equity_curve_tail": [
            {
                "date": r["date"],
                "equity": round(float(r["equity"]), 2),
                "cash": round(float(r["cash"]), 2),
                "shares": int(r["shares"]),
                "close": round(float(r["close"]), 4),
            }
            for _, r in curve_df.tail(20).iterrows()
        ],
    }

    if with_plots:
        plot_result = _build_backtest_plot(
            symbol=raw.get("code", _normalize_symbol(symbol)),
            df=df,
            curve_df=curve_df,
            trades=trades,
            initial_capital=initial_capital,
        )
        result["plots"] = [plot_result]

    return result


def _simulate_ma_from_df(
    df: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    initial_capital: float,
    fee_rate: float,
    slippage_rate: float,
    tax_rate: float,
) -> dict:
    """在给定历史数据上模拟双均线策略，返回核心绩效。"""
    work_df = df.copy()
    work_df["fast_ma"] = work_df["close"].rolling(fast_period).mean()
    work_df["slow_ma"] = work_df["close"].rolling(slow_period).mean()
    work_df["signal"] = (work_df["fast_ma"] > work_df["slow_ma"]).astype(int)
    work_df["exec_signal"] = work_df["signal"].shift(1).fillna(0).astype(int)

    cash = float(initial_capital)
    shares = 0
    last_buy_date = None
    equity_curve = []
    roundtrip_returns = []
    buy_price = None

    for _, row in work_df.iterrows():
        current_date = row["date"]
        open_price = float(row["open"])
        close_price = float(row["close"])
        target_hold = int(row["exec_signal"])

        if target_hold == 1 and shares == 0 and open_price > 0:
            lot_size = 100
            shares_to_buy = int(cash / (open_price * (1 + fee_rate + slippage_rate)) / lot_size) * lot_size
            if shares_to_buy > 0:
                gross = shares_to_buy * open_price
                fee = gross * fee_rate
                slippage = gross * slippage_rate
                cash -= (gross + fee + slippage)
                shares = shares_to_buy
                last_buy_date = current_date
                buy_price = open_price

        elif target_hold == 0 and shares > 0 and open_price > 0:
            if last_buy_date is None or current_date > last_buy_date:
                gross = shares * open_price
                fee = gross * fee_rate
                slippage = gross * slippage_rate
                tax = gross * tax_rate
                cash += (gross - fee - slippage - tax)
                if buy_price and buy_price > 0:
                    buy_cost_price = buy_price * (1 + fee_rate + slippage_rate)
                    sell_net_price = open_price * (1 - fee_rate - slippage_rate - tax_rate)
                    roundtrip_returns.append(sell_net_price / buy_cost_price - 1)
                shares = 0
                last_buy_date = None
                buy_price = None

        equity_curve.append(cash + shares * close_price)

    curve = pd.Series(equity_curve, dtype="float64")
    ret = curve.pct_change().fillna(0.0)
    final_equity = float(curve.iloc[-1])
    total_return = final_equity / float(initial_capital) - 1
    trading_days = len(curve)
    annual_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1

    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    ret_std = float(ret.std())
    sharpe = 0.0
    if ret_std > 0:
        sharpe = float(ret.mean()) / ret_std * np.sqrt(252)

    wins = sum(1 for r in roundtrip_returns if r > 0)
    total_roundtrips = len(roundtrip_returns)
    win_rate = wins / total_roundtrips if total_roundtrips > 0 else None

    return {
        "fast_period": int(fast_period),
        "slow_period": int(slow_period),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe": round(sharpe, 3),
        "roundtrip_count": int(total_roundtrips),
        "win_rate_pct": round(win_rate * 100, 2) if win_rate is not None else None,
    }


def backtest_ma_grid_search(
    symbol: str,
    start_date: str,
    end_date: str,
    fast_candidates: str = "5,8,10,12",
    slow_candidates: str = "20,30,40,60",
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    tax_rate: float = 0.001,
    top_n: int = 5,
) -> dict:
    """
    双均线参数网格搜索。

    参数 fast_candidates / slow_candidates 传入逗号分隔整数列表，如 "5,8,10"。
    返回按年化收益、夏普、回撤三者综合排序后的 Top 参数组合。
    """
    print(
        "运行参数优化... {} {} ~ {}".format(symbol, start_date, end_date)
    )

    def _parse_candidates(value: str) -> list:
        nums = []
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                nums.append(int(token))
        return sorted(set(nums))

    fast_list = _parse_candidates(fast_candidates)
    slow_list = _parse_candidates(slow_candidates)
    if not fast_list or not slow_list:
        return {"error": "fast_candidates 与 slow_candidates 需为逗号分隔的正整数列表。"}
    if top_n <= 0:
        return {"error": "top_n 必须大于 0。"}

    raw = get_stock_history(symbol, start_date, end_date, frequency="d")
    if "error" in raw:
        return raw

    df = pd.DataFrame(raw["data"])
    if df.empty:
        return {"error": "回测数据为空。"}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    for col in ["open", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "close"]).reset_index(drop=True)

    max_slow = max(slow_list)
    if len(df) <= max_slow + 2:
        return {"error": "样本长度不足，建议至少提供 max(slow_candidates) + 20 个交易日。"}

    all_results = []
    skipped = []
    for fast in fast_list:
        for slow in slow_list:
            if fast >= slow:
                skipped.append({"fast_period": fast, "slow_period": slow, "reason": "fast_period 必须小于 slow_period"})
                continue
            perf = _simulate_ma_from_df(
                df=df,
                fast_period=fast,
                slow_period=slow,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                tax_rate=tax_rate,
            )
            all_results.append(perf)

    if not all_results:
        return {"error": "没有可用参数组合，请检查 fast/slow 候选集合。", "skipped": skipped}

    score_df = pd.DataFrame(all_results)
    # 综合评分：重收益和夏普，轻惩罚回撤。
    score_df["score"] = (
        score_df["annual_return_pct"].fillna(0.0) * 0.45
        + score_df["sharpe"].fillna(0.0) * 30 * 0.40
        + score_df["max_drawdown_pct"].fillna(0.0) * 0.15
    )
    score_df = score_df.sort_values(["score", "annual_return_pct", "sharpe"], ascending=[False, False, False])

    top_df = score_df.head(top_n)
    best = top_df.iloc[0].to_dict()

    return {
        "symbol": symbol,
        "code": raw.get("code", _normalize_symbol(symbol)),
        "strategy": "double_moving_average_grid_search",
        "date_range": f"{start_date} ~ {end_date}",
        "search_space": {
            "fast_candidates": fast_list,
            "slow_candidates": slow_list,
            "total_combinations": int(len(fast_list) * len(slow_list)),
            "valid_combinations": int(len(all_results)),
            "skipped_combinations": skipped,
        },
        "cost_model": {
            "initial_capital": float(initial_capital),
            "fee_rate": float(fee_rate),
            "slippage_rate": float(slippage_rate),
            "tax_rate": float(tax_rate),
            "lot_size": 100,
            "execution": "signal_on_close_execute_on_next_open",
            "t_plus_one": True,
        },
        "best_parameters": {
            "fast_period": int(best["fast_period"]),
            "slow_period": int(best["slow_period"]),
            "annual_return_pct": round(float(best["annual_return_pct"]), 2),
            "max_drawdown_pct": round(float(best["max_drawdown_pct"]), 2),
            "sharpe": round(float(best["sharpe"]), 3),
            "score": round(float(best["score"]), 3),
        },
        "top_results": [
            {
                "fast_period": int(r["fast_period"]),
                "slow_period": int(r["slow_period"]),
                "total_return_pct": round(float(r["total_return_pct"]), 2),
                "annual_return_pct": round(float(r["annual_return_pct"]), 2),
                "max_drawdown_pct": round(float(r["max_drawdown_pct"]), 2),
                "sharpe": round(float(r["sharpe"]), 3),
                "roundtrip_count": int(r["roundtrip_count"]),
                "win_rate_pct": round(float(r["win_rate_pct"]), 2) if r["win_rate_pct"] is not None else None,
                "score": round(float(r["score"]), 3),
            }
            for _, r in top_df.iterrows()
        ],
    }


def screen_stocks_by_factors(
    start_date: str,
    end_date: str,
    universe_codes: str = "",
    universe_limit: int = 80,
    top_n: int = 10,
    min_avg_amount: float = 0.0,
    with_plots: bool = False,
) -> dict:
    """
    因子选股（A 股）。

    评分逻辑（示例，可继续扩展）：
    1) 动量：20/60 日收益率越高越好。
    2) 趋势：close > MA20 > MA60 给予加分。
    3) 风险：20 日年化波动率、60 日最大回撤越低越好。
    4) 流动性：近 20 日平均成交额越高越好。

    :param start_date: 回看开始日期 yyyy-mm-dd，建议覆盖至少 80 个交易日
    :param end_date: 回看结束日期 yyyy-mm-dd
    :param universe_codes: 候选股票池，逗号分隔代码（可省略前缀），如 "600000,000001,300750"
    :param universe_limit: 不指定股票池时，从全市场截取前 N 只作为候选
    :param top_n: 返回前 N 只
    :param min_avg_amount: 最低近 20 日平均成交额过滤（单位：元）
    :param with_plots: 是否输出横向对比图（HTML 文件路径）
    """
    print("运行选股... {} ~ {}".format(start_date, end_date))
    print(
        "选股参数: universe_limit={}, top_n={}, min_avg_amount={}, input_codes={}".format(
            universe_limit,
            top_n,
            min_avg_amount,
            "provided" if str(universe_codes).strip() else "auto",
        )
    )
    print("阶段1/4: 构建股票池...")

    if universe_limit <= 0:
        return {"error": "universe_limit 必须大于 0。"}
    if top_n <= 0:
        return {"error": "top_n 必须大于 0。"}

    # 当数据接口异常或返回空集时，使用内置高流动性股票池兜底。
    built_in_core_universe = [
        "600519", "601318", "600036", "600000", "600887", "600276", "600309", "600900", "601899", "601688",
        "601012", "601166", "601288", "601398", "601328", "601988", "601888", "601601", "601628", "601211",
        "601857", "600030", "600050", "600104", "600031", "600196", "600048", "600809", "603288", "603259",
        "000001", "000333", "000651", "000858", "000725", "000100", "000568", "000002", "000063", "000776",
        "300750", "300760", "300015", "300059", "300124", "300274", "300014", "300408", "002594", "002415",
    ]

    def _parse_universe(raw_codes: str) -> list:
        if not raw_codes:
            return []
        items = [x.strip() for x in str(raw_codes).split(",") if x.strip()]
        return [_normalize_symbol(x) for x in items]

    def _candidate_days(day_text: str, lookback_days: int = 12) -> list:
        dt = pd.to_datetime(day_text, errors="coerce")
        if pd.isna(dt):
            dt = pd.Timestamp(datetime.now().date())
        return [(dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(lookback_days)]

    def _build_universe_from_all_stock() -> tuple:
        """优先使用 query_all_stock，自动回退到最近交易日。"""
        for d in _candidate_days(end_date):
            rs_all = bs.query_all_stock(day=d)
            if rs_all.error_code != "0":
                continue

            rows = []
            while rs_all.next():
                rows.append(rs_all.get_row_data())

            if not rows:
                continue

            all_df = pd.DataFrame(rows, columns=rs_all.fields)
            if "code" not in all_df.columns:
                continue

            all_df = all_df[all_df["code"].str.match(r"^(sh|sz)\\.\\d{6}$", na=False)]
            all_df = all_df[
                all_df["code"].str.startswith("sh.6")
                | all_df["code"].str.startswith("sz.0")
                | all_df["code"].str.startswith("sz.3")
            ]
            if "tradeStatus" in all_df.columns:
                all_df = all_df[all_df["tradeStatus"].astype(str) == "1"]

            all_df = all_df.drop_duplicates(subset=["code"]).head(universe_limit)
            if not all_df.empty:
                info = {
                    "source": "query_all_stock",
                    "day": d,
                    "size": int(len(all_df)),
                }
                return all_df, info

        return pd.DataFrame(), {"source": "query_all_stock", "day": None, "size": 0}

    def _build_universe_from_stock_basic() -> tuple:
        """query_all_stock 失败时，退化到 query_stock_basic 全量列表。"""
        rs_basic = bs.query_stock_basic()
        if rs_basic.error_code != "0":
            # 某些环境下 query_stock_basic() 可能不接受空参，这里做二次尝试。
            rs_basic = bs.query_stock_basic(code_name="")
            if rs_basic.error_code != "0":
                return pd.DataFrame(), {"source": "query_stock_basic", "error": rs_basic.error_msg}

        rows = []
        while rs_basic.next():
            rows.append(rs_basic.get_row_data())
        if not rows:
            return pd.DataFrame(), {"source": "query_stock_basic", "error": "empty"}

        all_df = pd.DataFrame(rows, columns=rs_basic.fields)
        if "code" not in all_df.columns:
            return pd.DataFrame(), {"source": "query_stock_basic", "error": "missing_code"}

        raw_df = all_df.copy()
        all_df = all_df[all_df["code"].str.match(r"^(sh|sz)\\.\\d{6}$", na=False)]
        all_df = all_df[
            all_df["code"].str.startswith("sh.6")
            | all_df["code"].str.startswith("sz.0")
            | all_df["code"].str.startswith("sz.3")
        ]

        filtered_df = all_df.copy()
        if "type" in all_df.columns:
            filtered_df = filtered_df[filtered_df["type"].astype(str) == "1"]
        if "status" in all_df.columns:
            filtered_df = filtered_df[filtered_df["status"].astype(str) == "1"]

        # 若严格过滤后为空，回退到基础 A 股集合，避免全量为空。
        if not filtered_df.empty:
            all_df = filtered_df
        elif not all_df.empty:
            pass
        else:
            # 兜底：从原始数据重新按代码规则筛一次。
            all_df = raw_df[raw_df["code"].str.match(r"^(sh|sz)\\.\\d{6}$", na=False)]
            all_df = all_df[
                all_df["code"].str.startswith("sh.6")
                | all_df["code"].str.startswith("sz.0")
                | all_df["code"].str.startswith("sz.3")
            ]

        all_df = all_df.drop_duplicates(subset=["code"]).head(universe_limit)
        return all_df, {"source": "query_stock_basic", "size": int(len(all_df))}

    def _call_index_query(func, d: str):
        """兼容不同 baostock 版本的指数成分查询参数。"""
        call_candidates = [
            lambda: func(date=d),
            lambda: func(day=d),
            lambda: func(d),
            lambda: func(),
        ]
        last = None
        for call in call_candidates:
            try:
                rs = call()
            except TypeError:
                continue
            except Exception as e:
                last = str(e)
                continue
            if rs is not None:
                return rs, None
        return None, last

    def _build_universe_from_index_constituents() -> tuple:
        """进一步兜底：从主流指数成分股构建股票池。"""
        query_funcs = [
            ("hs300", getattr(bs, "query_hs300_stocks", None)),
            ("zz500", getattr(bs, "query_zz500_stocks", None)),
            ("sz50", getattr(bs, "query_sz50_stocks", None)),
        ]

        for d in _candidate_days(end_date, lookback_days=20):
            frames = []
            for name, func in query_funcs:
                if func is None:
                    continue
                rs, call_err = _call_index_query(func, d)
                if rs is None:
                    continue
                if rs.error_code != "0":
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    continue
                df = pd.DataFrame(rows, columns=rs.fields)
                if "code" not in df.columns:
                    continue
                df["_src_index"] = name
                frames.append(df)

            if not frames:
                continue

            all_df = pd.concat(frames, ignore_index=True)
            all_df = all_df[all_df["code"].str.match(r"^(sh|sz)\\.\\d{6}$", na=False)]
            all_df = all_df[
                all_df["code"].str.startswith("sh.6")
                | all_df["code"].str.startswith("sz.0")
                | all_df["code"].str.startswith("sz.3")
            ]
            all_df = all_df.drop_duplicates(subset=["code"]).head(universe_limit)

            if not all_df.empty:
                return all_df, {
                    "source": "index_constituents",
                    "day": d,
                    "size": int(len(all_df)),
                }

        return pd.DataFrame(), {"source": "index_constituents", "day": None, "size": 0}

    def _build_universe_from_builtin() -> tuple:
        codes = [_normalize_symbol(c) for c in built_in_core_universe]
        codes = list(dict.fromkeys(codes))[:universe_limit]
        if not codes:
            return pd.DataFrame(), {"source": "built_in_core_universe", "size": 0}
        all_df = pd.DataFrame({"code": codes})
        return all_df, {"source": "built_in_core_universe", "size": int(len(all_df))}

    def _calc_max_drawdown(close_series: pd.Series) -> float:
        running_max = close_series.cummax()
        dd = close_series / running_max - 1
        return float(dd.min()) if len(dd) else 0.0

    codes = _parse_universe(universe_codes)
    if codes:
        print("使用输入股票池, 数量={}".format(len(codes)))

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        code_name_map = {}
        universe_meta = {"source": "input_codes" if codes else "auto", "day": end_date, "size": len(codes)}

        # 未指定股票池时，从全市场自动构建候选池。
        if not codes:
            all_df, universe_meta = _build_universe_from_all_stock()
            if all_df.empty:
                print("股票池回退: query_all_stock 无结果，尝试 query_stock_basic")
                all_df, universe_meta = _build_universe_from_stock_basic()
            if all_df.empty:
                print("股票池回退: query_stock_basic 无结果，尝试指数成分股")
                all_df, universe_meta = _build_universe_from_index_constituents()
            if all_df.empty:
                print("股票池回退: 指数成分股无结果，使用内置核心股票池")
                all_df, universe_meta = _build_universe_from_builtin()
            if all_df.empty:
                _ensure_logout()
                print("自动股票池构建失败: {}".format(universe_meta))
                return {
                    "error": "未获取到可用股票池，请检查数据服务或交易日。",
                    "universe_meta": universe_meta,
                }

            codes = all_df["code"].tolist()
            print("自动股票池构建完成: source={}, day={}, size={}".format(
                universe_meta.get("source"),
                universe_meta.get("day"),
                len(codes),
            ))
            if "code_name" in all_df.columns:
                code_name_map = dict(zip(all_df["code"], all_df["code_name"]))

        print(
            "阶段1/4完成: 股票池来源={}, 数量={}".format(
                universe_meta.get("source"),
                len(codes),
            )
        )

        start_dt = pd.to_datetime(start_date, errors="coerce")
        end_dt = pd.to_datetime(end_date, errors="coerce")
        window_days = (end_dt - start_dt).days if not pd.isna(start_dt) and not pd.isna(end_dt) else 120
        # 样本不足是常见失败原因，短窗口时降低最小 bar 门槛。
        min_required_bars = 60 if window_days >= 120 else max(30, min(60, window_days // 2))
        print("样本窗口={}天, 最小K线门槛={}".format(window_days, min_required_bars))
        print("阶段2/4: 因子计算与打分...")

        scored = []
        skipped = []
        history_map = {}

        total_codes = len(codes)
        for idx, code in enumerate(codes, start=1):
            if idx == 1 or idx % 20 == 0 or idx == total_codes:
                print("选股进度: {}/{} ({})".format(idx, total_codes, code))

            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,amount,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )

            if rs.error_code != "0":
                skipped.append({"code": code, "reason": rs.error_msg or "query_failed"})
                if len(skipped) <= 8:
                    print("跳过 {}: query_failed {}".format(code, rs.error_msg))
                continue

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                skipped.append({"code": code, "reason": "empty_history"})
                if len(skipped) <= 8:
                    print("跳过 {}: empty_history".format(code))
                continue

            df = pd.DataFrame(rows, columns=rs.fields)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "volume", "amount", "pctChg"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("date", ascending=True).dropna(subset=["close"]).reset_index(drop=True)
            if len(df) < min_required_bars:
                skipped.append({"code": code, "reason": f"insufficient_bars(<{min_required_bars})"})
                if len(skipped) <= 8:
                    print("跳过 {}: insufficient_bars={}".format(code, len(df)))
                continue

            close = df["close"]
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()

            ret20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else None
            ret60 = (close.iloc[-1] / close.iloc[-60] - 1) * 100 if len(close) >= 60 else None

            ret = close.pct_change().dropna()
            vol20 = ret.tail(20).std() * np.sqrt(252) * 100 if len(ret) >= 20 else None
            mdd60 = _calc_max_drawdown(close.tail(60)) * 100 if len(close) >= 60 else None
            avg_amount20 = float(df["amount"].tail(20).mean()) if "amount" in df.columns else None

            rsi_df = _compute_rsi(df[["date", "close"]].copy(), period=14)
            latest_rsi = rsi_df["RSI"].iloc[-1] if "RSI" in rsi_df.columns else np.nan

            if avg_amount20 is not None and avg_amount20 < float(min_avg_amount):
                skipped.append({"code": code, "reason": "below_min_avg_amount"})
                if len(skipped) <= 8:
                    print("跳过 {}: avg_amount20={} < {}".format(code, round(avg_amount20, 2), float(min_avg_amount)))
                continue

            trend_flag = int(
                pd.notna(ma20.iloc[-1])
                and pd.notna(ma60.iloc[-1])
                and close.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]
            )

            ret20_val = float(ret20) if ret20 is not None and not np.isnan(ret20) else 0.0
            ret60_val = float(ret60) if ret60 is not None and not np.isnan(ret60) else 0.0
            vol20_val = float(vol20) if vol20 is not None and not np.isnan(vol20) else 0.0
            mdd60_val = abs(float(mdd60)) if mdd60 is not None and not np.isnan(mdd60) else 0.0
            rsi_val = float(latest_rsi) if not pd.isna(latest_rsi) else 50.0
            amount_bonus = 0.0
            if avg_amount20 is not None and avg_amount20 > 0:
                amount_bonus = min(np.log10(avg_amount20 / 1e8 + 1.0) * 2.0, 5.0)

            # 综合打分：重动量与趋势，同时惩罚高波动和深回撤。
            score = (
                ret20_val * 0.35
                + ret60_val * 0.35
                + trend_flag * 12.0
                - vol20_val * 0.12
                - mdd60_val * 0.18
                - abs(rsi_val - 55.0) * 0.08
                + float(amount_bonus)
            )

            scored.append({
                "code": code,
                "code_name": code_name_map.get(code),
                "latest_close": round(float(close.iloc[-1]), 3),
                "return_20d_pct": round(ret20_val, 2),
                "return_60d_pct": round(ret60_val, 2),
                "volatility_20d_annual_pct": round(vol20_val, 2),
                "max_drawdown_60d_pct": round(-mdd60_val, 2),
                "rsi": round(rsi_val, 2),
                "trend_flag": trend_flag,
                "avg_amount_20d": round(float(avg_amount20), 2) if avg_amount20 is not None else None,
                "score": round(float(score), 3),
            })
            history_map[code] = df[["date", "close"]].copy()
            if len(scored) <= 3 or len(scored) % 20 == 0:
                print(
                    "打分样本: code={}, score={}, ret20={}, ret60={}, vol20={}".format(
                        code,
                        round(float(score), 3),
                        round(ret20_val, 2),
                        round(ret60_val, 2),
                        round(vol20_val, 2),
                    )
                )

        if not scored:
            _ensure_logout()
            return {
                "error": "无符合条件的股票，请放宽筛选条件或扩大股票池。",
                "skipped": skipped[:50],
            }

        rank_df = pd.DataFrame(scored).sort_values("score", ascending=False).reset_index(drop=True)
        top_df = rank_df.head(top_n)
        preview_codes = top_df["code"].astype(str).tolist()[:5] if not top_df.empty else []
        print("阶段2/4完成: total={}, scored={}, skipped={}".format(total_codes, len(scored), len(skipped)))
        print("阶段3/4: 生成Top候选与结果汇总...")
        print("Top候选预览: {}".format(preview_codes))

        response = {
            "strategy": "factor_screening_v1",
            "date_range": f"{start_date} ~ {end_date}",
            "universe": {
                "input_codes": universe_codes,
                "universe_size": int(len(codes)),
                "evaluated": int(len(scored)),
                "skipped": int(len(skipped)),
                "min_avg_amount": float(min_avg_amount),
                "meta": universe_meta,
                "min_required_bars": int(min_required_bars),
            },
            "scoring": {
                "formula": "0.35*ret20 + 0.35*ret60 + 12*trend - 0.12*vol20 - 0.18*abs(mdd60) - 0.08*abs(rsi-55) + liquidity_bonus",
                "fields": [
                    "return_20d_pct",
                    "return_60d_pct",
                    "trend_flag",
                    "volatility_20d_annual_pct",
                    "max_drawdown_60d_pct",
                    "rsi",
                    "avg_amount_20d",
                ],
            },
            "top_picks": [
                {
                    "rank": int(i + 1),
                    "code": r["code"],
                    "code_name": r.get("code_name"),
                    "score": round(float(r["score"]), 3),
                    "latest_close": round(float(r["latest_close"]), 3),
                    "return_20d_pct": round(float(r["return_20d_pct"]), 2),
                    "return_60d_pct": round(float(r["return_60d_pct"]), 2),
                    "volatility_20d_annual_pct": round(float(r["volatility_20d_annual_pct"]), 2),
                    "max_drawdown_60d_pct": round(float(r["max_drawdown_60d_pct"]), 2),
                    "rsi": round(float(r["rsi"]), 2),
                    "trend_flag": int(r["trend_flag"]),
                    "avg_amount_20d": round(float(r["avg_amount_20d"]), 2) if r["avg_amount_20d"] is not None else None,
                }
                for i, (_, r) in enumerate(top_df.iterrows())
            ],
            "skipped_samples": skipped[:30],
        }

        if with_plots:
            print("阶段4/4: 生成横向对比图...")
            response["plots"] = _build_cross_section_plots(
                symbol_tag="factor_screening",
                ranked_df=rank_df,
                history_map=history_map,
                score_col="score",
                return_col="return_60d_pct",
                risk_col="volatility_20d_annual_pct",
                title_prefix="因子选股",
            )
            print("图表生成完成: {} 张".format(len(response.get("plots", []))))

        print("选股流程完成: 返回Top {}".format(len(response.get("top_picks", []))))

        _ensure_logout()
        return response
    except Exception as e:
        _ensure_logout()
        return {"error": f"选股执行异常: {str(e)}"}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _infer_market_regime(end_date: str) -> dict:
    """根据大盘短期表现推断市场状态。"""
    market = get_market_overview()
    if "error" in market:
        return {
            "label": "unknown",
            "score": 50.0,
            "avg_pctChg": None,
            "avg_return_5d_pct": None,
            "date": end_date,
            "note": f"market_overview_unavailable: {market['error']}",
        }

    idx = pd.DataFrame(market.get("indices", []))
    if idx.empty:
        return {
            "label": "unknown",
            "score": 50.0,
            "avg_pctChg": None,
            "avg_return_5d_pct": None,
            "date": end_date,
            "note": "empty_indices",
        }

    avg_day = pd.to_numeric(idx.get("pctChg"), errors="coerce").dropna().mean()
    avg_5d = pd.to_numeric(idx.get("return_5d_pct"), errors="coerce").dropna().mean()
    avg_day = float(avg_day) if not pd.isna(avg_day) else 0.0
    avg_5d = float(avg_5d) if not pd.isna(avg_5d) else 0.0

    score = 50 + avg_day * 8 + avg_5d * 2
    score = _clip(score, 0.0, 100.0)
    if score >= 68:
        label = "risk_on"
    elif score <= 38:
        label = "risk_off"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": round(float(score), 1),
        "avg_pctChg": round(avg_day, 2),
        "avg_return_5d_pct": round(avg_5d, 2),
        "date": end_date,
    }


def select_stocks_for_trader(
    start_date: str,
    end_date: str,
    universe_codes: str = "",
    universe_limit: int = 120,
    candidate_n: int = 20,
    pick_n: int = 5,
    min_avg_amount: float = 200000000.0,
    fast_period: int = 5,
    slow_period: int = 20,
    initial_capital: float = 100000.0,
    with_plots: bool = False,
) -> dict:
    """
    交易员级选股：因子筛选 + 策略回测验证 + 市场状态适配 + 风险预算建议。

    输出不仅包含候选名单，还包含每个标的的建议仓位、止损/止盈与入选理由。
    """
    print("交易员选股... {} ~ {}".format(start_date, end_date))
    print(
        "交易员参数: universe_limit={}, candidate_n={}, pick_n={}, min_avg_amount={}, ma=({},{})".format(
            universe_limit,
            candidate_n,
            pick_n,
            min_avg_amount,
            fast_period,
            slow_period,
        )
    )
    print("阶段1/5: 市场状态识别...")

    if candidate_n <= 0 or pick_n <= 0:
        return {"error": "candidate_n 与 pick_n 必须大于 0。"}
    if fast_period >= slow_period:
        return {"error": "fast_period 必须小于 slow_period。"}

    regime = _infer_market_regime(end_date)
    print("市场状态: label={}, score={}".format(regime.get("label"), regime.get("score")))

    # 采用多轮回退策略，降低非交易日/样本不足导致的空结果概率。
    end_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(end_dt):
        return {"error": "end_date 格式无效，请使用 yyyy-mm-dd。"}

    attempts = []
    for day_shift in [0, 1, 2, 3, 4, 5]:
        d = (end_dt - timedelta(days=day_shift)).strftime("%Y-%m-%d")
        attempts.extend([
            {
                "end_date": d,
                "universe_limit": int(max(universe_limit, 80)),
                "candidate_n": int(max(candidate_n, pick_n)),
                "min_avg_amount": float(min_avg_amount),
                "tag": "base",
            },
            {
                "end_date": d,
                "universe_limit": int(max(universe_limit, 180)),
                "candidate_n": int(max(candidate_n, 40)),
                "min_avg_amount": 0.0,
                "tag": "relax_liquidity",
            },
        ])

    screen = None
    picks = []
    attempt_logs = []

    for plan in attempts:
        print(
            "尝试筛选: tag={}, end_date={}, universe_limit={}, candidate_n={}, min_avg_amount={}".format(
                plan["tag"],
                plan["end_date"],
                plan["universe_limit"],
                plan["candidate_n"],
                plan["min_avg_amount"],
            )
        )
        screen_try = screen_stocks_by_factors(
            start_date=start_date,
            end_date=plan["end_date"],
            universe_codes=universe_codes,
            universe_limit=plan["universe_limit"],
            top_n=plan["candidate_n"],
            min_avg_amount=plan["min_avg_amount"],
            with_plots=False,
        )
        if "error" in screen_try:
            print("尝试失败: {}".format(screen_try.get("error")))
            attempt_logs.append(
                {
                    "tag": plan["tag"],
                    "end_date": plan["end_date"],
                    "universe_limit": plan["universe_limit"],
                    "candidate_n": plan["candidate_n"],
                    "min_avg_amount": plan["min_avg_amount"],
                    "status": "error",
                    "message": screen_try.get("error"),
                }
            )
            continue

        picks_try = screen_try.get("top_picks", [])
        if not picks_try:
            print("尝试结果为空: no_top_picks")
            attempt_logs.append(
                {
                    "tag": plan["tag"],
                    "end_date": plan["end_date"],
                    "universe_limit": plan["universe_limit"],
                    "candidate_n": plan["candidate_n"],
                    "min_avg_amount": plan["min_avg_amount"],
                    "status": "empty",
                    "message": "no_top_picks",
                }
            )
            continue

        screen = screen_try
        picks = picks_try
        print("尝试成功: 获取候选 {} 只".format(len(picks_try)))
        attempt_logs.append(
            {
                "tag": plan["tag"],
                "end_date": plan["end_date"],
                "universe_limit": plan["universe_limit"],
                "candidate_n": plan["candidate_n"],
                "min_avg_amount": plan["min_avg_amount"],
                "status": "ok",
                "message": f"picked={len(picks_try)}",
            }
        )
        break

    print("阶段2/5结束: 尝试次数={}, 有效候选={}".format(len(attempt_logs), len(picks)))

    if not picks:
        print("交易员选股失败: 所有尝试均未得到候选。")
        return {
            "error": "无符合条件的股票，请放宽筛选条件或扩大股票池。",
            "market_regime": regime,
            "screening_context": {
                "candidate_n": candidate_n,
                "pick_n": pick_n,
                "min_avg_amount": min_avg_amount,
                "universe_limit": universe_limit,
            },
            "attempt_logs": attempt_logs[-12:],
            "suggestion": "可显式指定 universe_codes 或延长起止区间，例如最近 9-12 个月。",
        }

    results = []
    skipped = []

    print("阶段3/5: 回测验证与二次评分, 候选数量={}".format(len(picks)))

    for i, row in enumerate(picks, start=1):
        code = str(row.get("code", "")).strip()
        if not code:
            continue

        print("回测验证进度: {}/{} ({})".format(i, len(picks), code))

        bt = backtest_moving_average_strategy(
            symbol=code,
            start_date=start_date,
            end_date=end_date,
            fast_period=fast_period,
            slow_period=slow_period,
            initial_capital=initial_capital,
        )
        if "error" in bt:
            skipped.append({"code": code, "reason": f"backtest_failed: {bt['error']}"})
            if len(skipped) <= 8:
                print("回测跳过 {}: {}".format(code, bt["error"]))
            continue

        perf = bt.get("performance", {})
        latest_close = float(row.get("latest_close") or 0.0)
        screen_score = float(row.get("score") or 0.0)
        vol20 = abs(float(row.get("volatility_20d_annual_pct") or 0.0))
        ret20 = float(row.get("return_20d_pct") or 0.0)
        ret60 = float(row.get("return_60d_pct") or 0.0)

        ann = float(perf.get("annual_return_pct") or 0.0)
        mdd = abs(float(perf.get("max_drawdown_pct") or 0.0))
        sharpe = float(perf.get("sharpe") or 0.0)
        win_rate = float(perf.get("win_rate_pct") or 0.0)

        quality = (
            screen_score * 0.35
            + ann * 0.25
            + sharpe * 10.0
            - mdd * 0.18
            - vol20 * 0.08
            + win_rate * 0.03
        )

        # 市场状态适配：risk_off 时更强调回撤控制与波动抑制。
        if regime["label"] == "risk_off":
            quality += -mdd * 0.08 - vol20 * 0.05 + min(win_rate, 70) * 0.02
            base_pos = 0.06
            target_vol = 14.0
        elif regime["label"] == "risk_on":
            quality += ann * 0.06 + ret20 * 0.05
            base_pos = 0.16
            target_vol = 24.0
        else:
            base_pos = 0.10
            target_vol = 18.0

        vol_ref = max(vol20, 6.0)
        suggested_position = _clip(base_pos * target_vol / vol_ref, 0.03, 0.30)
        stop_loss_pct = _clip(max(5.0, vol20 * 0.45), 4.0, 14.0)
        take_profit_pct = _clip(stop_loss_pct * 1.8, 8.0, 28.0)

        reasons = []
        if ret60 > 0:
            reasons.append(f"中期动量较强({ret60:.2f}%)")
        if row.get("trend_flag") == 1:
            reasons.append("趋势结构良好(close>MA20>MA60)")
        if sharpe > 1:
            reasons.append(f"回测夏普较高({sharpe:.2f})")
        if mdd < 15:
            reasons.append(f"回撤可控({mdd:.2f}%)")
        if not reasons:
            reasons.append("综合评分领先")

        results.append(
            {
                "code": code,
                "code_name": row.get("code_name"),
                "latest_close": round(latest_close, 3) if latest_close > 0 else None,
                "selection_score": round(float(quality), 3),
                "factor_score": round(screen_score, 3),
                "return_20d_pct": round(ret20, 2),
                "return_60d_pct": round(ret60, 2),
                "volatility_20d_annual_pct": round(vol20, 2),
                "backtest_annual_return_pct": round(ann, 2),
                "backtest_max_drawdown_pct": round(mdd, 2),
                "backtest_sharpe": round(sharpe, 3),
                "backtest_win_rate_pct": round(win_rate, 2),
                "suggested_position_pct": round(float(suggested_position * 100), 2),
                "suggested_stop_loss_pct": round(float(stop_loss_pct), 2),
                "suggested_take_profit_pct": round(float(take_profit_pct), 2),
                "stop_loss_price": round(float(latest_close * (1 - stop_loss_pct / 100)), 3) if latest_close > 0 else None,
                "take_profit_price": round(float(latest_close * (1 + take_profit_pct / 100)), 3) if latest_close > 0 else None,
                "reasons": reasons,
            }
        )
        if len(results) <= 3 or len(results) % 10 == 0:
            print(
                "验证样本: code={}, quality={}, ann={}, mdd={}, sharpe={}, pos={}%".format(
                    code,
                    round(float(quality), 3),
                    round(ann, 2),
                    round(mdd, 2),
                    round(sharpe, 3),
                    round(float(suggested_position * 100), 2),
                )
            )

    if not results:
        print("交易员选股失败: 回测验证后无可用候选, skipped={}".format(len(skipped)))
        return {
            "error": "候选股票回测验证后为空，请扩大样本或放宽筛选。",
            "market_regime": regime,
            "skipped": skipped[:50],
        }

    ranked = pd.DataFrame(results).sort_values("selection_score", ascending=False).reset_index(drop=True)
    top_df = ranked.head(pick_n)
    print("阶段3/5结束: validated={}, skipped={}".format(len(results), len(skipped)))
    print("阶段4/5: 生成最终候选列表, final_pick={}".format(len(top_df)))

    response = {
        "strategy": "trader_stock_selection_v1",
        "date_range": f"{start_date} ~ {end_date}",
        "market_regime": regime,
        "attempt_logs": attempt_logs[-12:],
        "selection_context": {
            "universe_codes": universe_codes,
            "universe_limit": int(universe_limit),
            "candidate_n": int(candidate_n),
            "pick_n": int(pick_n),
            "min_avg_amount": float(min_avg_amount),
            "validation_backtest": {
                "fast_period": int(fast_period),
                "slow_period": int(slow_period),
                "initial_capital": float(initial_capital),
            },
        },
        "top_picks": [
            {
                "rank": int(i + 1),
                "code": r["code"],
                "code_name": r.get("code_name"),
                "selection_score": round(float(r["selection_score"]), 3),
                "factor_score": round(float(r["factor_score"]), 3),
                "return_20d_pct": round(float(r["return_20d_pct"]), 2),
                "return_60d_pct": round(float(r["return_60d_pct"]), 2),
                "volatility_20d_annual_pct": round(float(r["volatility_20d_annual_pct"]), 2),
                "backtest_annual_return_pct": round(float(r["backtest_annual_return_pct"]), 2),
                "backtest_max_drawdown_pct": round(float(r["backtest_max_drawdown_pct"]), 2),
                "backtest_sharpe": round(float(r["backtest_sharpe"]), 3),
                "backtest_win_rate_pct": round(float(r["backtest_win_rate_pct"]), 2),
                "suggested_position_pct": round(float(r["suggested_position_pct"]), 2),
                "suggested_stop_loss_pct": round(float(r["suggested_stop_loss_pct"]), 2),
                "suggested_take_profit_pct": round(float(r["suggested_take_profit_pct"]), 2),
                "stop_loss_price": round(float(r["stop_loss_price"]), 3) if r.get("stop_loss_price") is not None else None,
                "take_profit_price": round(float(r["take_profit_price"]), 3) if r.get("take_profit_price") is not None else None,
                "reasons": r.get("reasons", []),
            }
            for i, (_, r) in enumerate(top_df.iterrows())
        ],
        "skipped_candidates": skipped[:30],
    }

    if with_plots:
        print("阶段5/5: 生成横向对比图与风控图...")
        top_codes = top_df["code"].astype(str).tolist()
        history_map = {}
        history_skipped = []
        for code in top_codes:
            hist_raw = get_stock_history(code, start_date, end_date, frequency="d")
            if "error" in hist_raw:
                history_skipped.append({"code": code, "reason": hist_raw.get("error")})
                continue
            hist_df = _history_records_to_df(hist_raw.get("data", []))
            if hist_df.empty:
                history_skipped.append({"code": code, "reason": "empty_history"})
                continue
            history_map[code] = hist_df[["date", "close"]].copy()

        plots = _build_cross_section_plots(
            symbol_tag="trader_selection",
            ranked_df=ranked,
            history_map=history_map,
            score_col="selection_score",
            return_col="backtest_annual_return_pct",
            risk_col="backtest_max_drawdown_pct",
            title_prefix="交易员选股",
        )

        alloc_fig = go.Figure()
        alloc_fig.add_trace(
            go.Bar(
                x=top_df["code"],
                y=pd.to_numeric(top_df["suggested_position_pct"], errors="coerce"),
                name="建议仓位%",
                marker_color="#1f77b4",
            )
        )
        alloc_fig.add_trace(
            go.Bar(
                x=top_df["code"],
                y=pd.to_numeric(top_df["suggested_stop_loss_pct"], errors="coerce"),
                name="止损%",
                marker_color="#d62728",
            )
        )
        alloc_fig.add_trace(
            go.Bar(
                x=top_df["code"],
                y=pd.to_numeric(top_df["suggested_take_profit_pct"], errors="coerce"),
                name="止盈%",
                marker_color="#2ca02c",
            )
        )
        alloc_fig.update_layout(
            title="交易员选股仓位与风控参数对比",
            xaxis_title="Code",
            yaxis_title="Percent",
            barmode="group",
            height=520,
            margin={"l": 24, "r": 20, "t": 56, "b": 24},
        )
        plots.append({
            "chart_type": "allocation_and_risk",
            "path": _save_plotly_html(alloc_fig, symbol="trader_selection", chart_name="allocation_risk"),
            "points": int(len(top_df)),
        })

        response["plots"] = plots
        if history_skipped:
            response["plot_history_skipped"] = history_skipped[:10]
            print("图表数据缺失样本: {}".format(len(history_skipped)))
        print("图表生成完成: {} 张".format(len(response.get("plots", []))))

    print("交易员选股流程完成: 返回Top {}".format(len(response.get("top_picks", []))))

    return response
