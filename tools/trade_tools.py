# -*- coding: utf-8 -*-
"""
交易工具层：组合优化、风险管理、执行规划。
支持风险平价模型、均值方差优化、VaR/CVaR计算、分批建仓规划。
"""
from datetime import datetime, timedelta
import os
import json

import baostock as bs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "plots")
_PORTFOLIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "portfolios")


# ---------- 证券代码规范化 ----------
def _normalize_symbol(symbol: str) -> str:
    """将用户输入的代码转为 baostock 格式。"""
    s = str(symbol).strip().split(".")[0]
    if not s.isdigit():
        return symbol.strip()
    if s.startswith("6"):
        return f"sh.{s}"
    if s.startswith(("0", "3")):
        return f"sz.{s}"
    return symbol.strip()


def _ensure_login():
    """确保已登录。"""
    lg = bs.login()
    return lg.error_code == "0"


def _ensure_logout():
    bs.logout()


def _ensure_plot_dir() -> str:
    os.makedirs(_PLOT_DIR, exist_ok=True)
    return _PLOT_DIR


def _ensure_portfolio_dir() -> str:
    os.makedirs(_PORTFOLIO_DIR, exist_ok=True)
    return _PORTFOLIO_DIR


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


def _get_history_prices(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取多只股票的历史收盘价。
    :param symbols: 股票代码列表（已标准化）
    :param start_date: 开始日期 yyyy-mm-dd
    :param end_date: 结束日期 yyyy-mm-dd
    :return: DataFrame，index为日期，columns为股票代码
    """
    if not _ensure_login():
        return pd.DataFrame()

    price_data = {}
    for code in symbols:
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,close",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",  # 前复权
            )
            if rs.error_code != "0":
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=rs.fields)
                df["date"] = pd.to_datetime(df["date"])
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df.set_index("date")["close"]
                price_data[code] = df
        except Exception:
            continue

    _ensure_logout()

    if not price_data:
        return pd.DataFrame()

    return pd.DataFrame(price_data).dropna()


# ---------- 工具 1：组合优化 ----------
def optimize_portfolio(
    symbols: str,
    total_capital: float = 1000000,
    start_date: str = None,
    end_date: str = None,
    method: str = "risk_parity",
    max_position_pct: float = 0.3,
    max_sector_pct: float = 0.5,
    risk_free_rate: float = 0.03,
    with_plots: bool = False,
) -> dict:
    """
    组合优化：根据候选股票列表，计算最优权重分配。
    :param symbols: 股票代码，逗号分隔，如 "600000,000001,300750"
    :param total_capital: 总资金，默认100万
    :param start_date: 历史数据开始日期，默认近一年
    :param end_date: 历史数据结束日期，默认今天
    :param method: 优化方法，risk_parity（风险平价）或 mean_variance（均值方差）
    :param max_position_pct: 单标的最大仓位比例，默认30%
    :param max_sector_pct: 同行业最大仓位比例，默认50%（暂未实现行业分类）
    :param risk_free_rate: 无风险利率，默认3%
    :param with_plots: 是否输出图表
    :return: 组合优化结果
    """
    print(f"组合优化... {symbols}")

    # 解析股票代码
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return {"error": "请提供有效的股票代码列表"}

    # 标准化代码
    codes = [_normalize_symbol(s) for s in symbol_list]

    # 默认日期范围：近一年
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    # 获取历史价格
    prices = _get_history_prices(codes, start_date, end_date)
    if prices.empty:
        return {"error": "无法获取历史价格数据"}

    # 计算收益率
    returns = prices.pct_change().dropna()
    if returns.empty:
        return {"error": "收益率数据不足"}

    n_assets = len(codes)

    # 计算协方差矩阵
    cov_matrix = returns.cov() * 252  # 年化

    # 根据方法计算权重
    if method == "risk_parity":
        weights = _risk_parity_weights(cov_matrix)
    elif method == "mean_variance":
        expected_returns = returns.mean() * 252  # 年化
        weights = _mean_variance_weights(expected_returns, cov_matrix, risk_free_rate)
    elif method == "equal":
        weights = np.array([1.0 / n_assets] * n_assets)
    else:
        return {"error": f"不支持的优化方法: {method}"}

    # 应用约束：单标的最大仓位
    weights = np.minimum(weights, max_position_pct)
    weights = weights / weights.sum()  # 重新归一化

    # 计算组合指标
    portfolio_return = _calculate_portfolio_return(returns, weights)
    portfolio_volatility = _calculate_portfolio_volatility(returns, weights)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

    # 构建结果
    result = {
        "method": method,
        "symbols": symbol_list,
        "codes": codes,
        "total_capital": total_capital,
        "weights": {symbol_list[i]: round(w * 100, 2) for i, w in enumerate(weights)},
        "allocation": {
            symbol_list[i]: round(w * total_capital, 2) for i, w in enumerate(weights)
        },
        "portfolio_metrics": {
            "expected_return_pct": round(portfolio_return * 100, 2),
            "volatility_pct": round(portfolio_volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
        },
        "constraints": {
            "max_position_pct": max_position_pct * 100,
            "max_sector_pct": max_sector_pct * 100,
        },
    }

    # 计算各标的的风险贡献
    risk_contributions = _calculate_risk_contribution(cov_matrix, weights)
    result["risk_contribution"] = {
        symbol_list[i]: round(rc * 100, 2) for i, rc in enumerate(risk_contributions)
    }

    # 生成图表
    if with_plots:
        plot_result = _build_portfolio_plot(symbol_list, weights, result)
        result["plot"] = plot_result

    return result


def _risk_parity_weights(cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    风险平价权重计算。
    目标：各资产的风险贡献相等。
    """
    n = len(cov_matrix)

    # 使用风险平价的解析解（逆波动率加权）
    volatilities = np.sqrt(np.diag(cov_matrix))
    weights = 1.0 / volatilities
    weights = weights / weights.sum()

    return weights


def _mean_variance_weights(
    expected_returns: np.ndarray,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> np.ndarray:
    """
    均值方差优化权重计算（最大化夏普比率）。
    使用解析解：w = Σ^(-1) * (μ - rf) / 1'Σ^(-1)(μ - rf)
    """
    n = len(cov_matrix)

    # 超额收益
    excess_returns = expected_returns.values - risk_free_rate

    # 协方差矩阵的逆
    try:
        cov_inv = np.linalg.inv(cov_matrix.values)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        cov_inv = np.linalg.pinv(cov_matrix.values)

    # 计算权重
    weights = cov_inv @ excess_returns
    weights = weights / weights.sum()

    # 确保权重非负
    weights = np.maximum(weights, 0)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.array([1.0 / n] * n)

    return weights


def _calculate_portfolio_return(returns: pd.DataFrame, weights: np.ndarray) -> float:
    """计算组合预期收益率。"""
    return (returns.mean() * 252).values @ weights


def _calculate_portfolio_volatility(returns: pd.DataFrame, weights: np.ndarray) -> float:
    """计算组合波动率。"""
    cov_matrix = returns.cov() * 252
    return np.sqrt(weights @ cov_matrix.values @ weights)


def _calculate_risk_contribution(cov_matrix: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """计算各资产的风险贡献。"""
    portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
    marginal_contrib = cov_matrix.values @ weights
    risk_contrib = weights * marginal_contrib / portfolio_vol
    return risk_contrib / risk_contrib.sum()


def _build_portfolio_plot(symbols: list, weights: np.ndarray, result: dict) -> dict:
    """生成组合配置图表。"""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("权重分配", "风险贡献"),
        specs=[[{"type": "pie"}, {"type": "pie"}]],
    )

    # 权重饼图
    fig.add_trace(
        go.Pie(
            labels=symbols,
            values=[w * 100 for w in weights],
            name="权重",
            hole=0.3,
        ),
        row=1,
        col=1,
    )

    # 风险贡献饼图
    if "risk_contribution" in result:
        fig.add_trace(
            go.Pie(
                labels=symbols,
                values=list(result["risk_contribution"].values()),
                name="风险贡献",
                hole=0.3,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="组合配置分析",
        height=450,
        showlegend=True,
    )

    path = _save_plotly_html(fig, symbol="portfolio", chart_name="allocation")
    return {"chart_type": "portfolio_allocation", "path": path}


# ---------- 工具 2：风险计算 ----------
def calculate_portfolio_risk(
    symbols: str,
    weights: str = None,
    total_capital: float = 1000000,
    start_date: str = None,
    end_date: str = None,
    confidence_level: float = 0.95,
    holding_days: int = 10,
    with_plots: bool = False,
) -> dict:
    """
    计算组合风险指标：VaR、CVaR、预期最大回撤。
    :param symbols: 股票代码，逗号分隔
    :param weights: 权重，逗号分隔，如 "0.3,0.3,0.4"，默认等权
    :param total_capital: 总资金
    :param start_date: 历史数据开始日期
    :param end_date: 历史数据结束日期
    :param confidence_level: 置信水平，默认95%
    :param holding_days: 持仓天数，默认10天
    :param with_plots: 是否输出图表
    :return: 风险指标
    """
    print(f"计算组合风险... {symbols}")

    # 解析股票代码
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return {"error": "请提供有效的股票代码列表"}

    codes = [_normalize_symbol(s) for s in symbol_list]
    n_assets = len(codes)

    # 解析权重
    if weights:
        weight_list = [float(w.strip()) for w in weights.split(",")]
        if len(weight_list) != n_assets:
            return {"error": "权重数量与股票数量不匹配"}
        weight_array = np.array(weight_list)
        weight_array = weight_array / weight_array.sum()  # 归一化
    else:
        weight_array = np.array([1.0 / n_assets] * n_assets)

    # 默认日期范围
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")  # 近两年

    # 获取历史价格
    prices = _get_history_prices(codes, start_date, end_date)
    if prices.empty:
        return {"error": "无法获取历史价格数据"}

    # 计算组合收益率
    returns = prices.pct_change().dropna()
    portfolio_returns = (returns * weight_array).sum(axis=1)

    # 计算VaR（历史模拟法）
    var_pct = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    var_amount = abs(var_pct * total_capital)

    # 计算CVaR（条件风险价值）
    tail_returns = portfolio_returns[portfolio_returns <= var_pct]
    cvar_pct = tail_returns.mean() if len(tail_returns) > 0 else var_pct
    cvar_amount = abs(cvar_pct * total_capital)

    # 计算预期最大回撤
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()
    avg_drawdown = drawdowns.mean()

    # 计算年化波动率
    volatility = portfolio_returns.std() * np.sqrt(252)

    # 计算持仓期调整
    var_holding = var_pct * np.sqrt(holding_days)
    cvar_holding = cvar_pct * np.sqrt(holding_days)

    result = {
        "symbols": symbol_list,
        "weights": {symbol_list[i]: round(w * 100, 2) for i, w in enumerate(weight_array)},
        "total_capital": total_capital,
        "confidence_level": confidence_level,
        "holding_days": holding_days,
        "risk_metrics": {
            "var_1d_pct": round(var_pct * 100, 2),
            "var_1d_amount": round(var_amount, 2),
            "cvar_1d_pct": round(cvar_pct * 100, 2),
            "cvar_1d_amount": round(cvar_amount, 2),
            "var_holding_pct": round(var_holding * 100, 2),
            "cvar_holding_pct": round(cvar_holding * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "avg_drawdown_pct": round(avg_drawdown * 100, 2),
            "volatility_annual_pct": round(volatility * 100, 2),
        },
        "risk_assessment": {},
    }

    # 风险评估
    if abs(max_drawdown) > 0.3:
        result["risk_assessment"]["drawdown"] = "历史最大回撤超30%，风险较高"
    elif abs(max_drawdown) > 0.2:
        result["risk_assessment"]["drawdown"] = "历史最大回撤在20%-30%，风险中等"
    else:
        result["risk_assessment"]["drawdown"] = "历史最大回撤小于20%，风险较低"

    if volatility > 0.3:
        result["risk_assessment"]["volatility"] = "年化波动率超30%，波动较大"
    elif volatility > 0.2:
        result["risk_assessment"]["volatility"] = "年化波动率在20%-30%，波动中等"
    else:
        result["risk_assessment"]["volatility"] = "年化波动率小于20%，波动较小"

    # 风险预算建议
    risk_budget = total_capital * abs(var_holding)
    result["risk_budget"] = {
        "suggested_stop_loss_amount": round(risk_budget, 2),
        "suggested_stop_loss_pct": round(abs(var_holding) * 100, 2),
    }

    # 生成图表
    if with_plots:
        plot_result = _build_risk_plot(portfolio_returns, cumulative, drawdowns)
        result["plot"] = plot_result

    return result


def _build_risk_plot(
    portfolio_returns: pd.Series,
    cumulative: pd.Series,
    drawdowns: pd.Series,
) -> dict:
    """生成风险分析图表。"""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("组合净值曲线", "日收益率分布", "回撤曲线"),
    )

    # 净值曲线
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            name="净值",
            line={"color": "#1f77b4"},
        ),
        row=1,
        col=1,
    )

    # 收益率分布
    fig.add_trace(
        go.Histogram(
            x=portfolio_returns * 100,
            name="日收益率(%)",
            marker_color="#ff7f0e",
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    # 回撤曲线
    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values * 100,
            mode="lines",
            name="回撤(%)",
            fill="tozeroy",
            line={"color": "#d62728"},
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title="组合风险分析",
        height=900,
        showlegend=False,
    )

    path = _save_plotly_html(fig, symbol="risk", chart_name="analysis")
    return {"chart_type": "risk_analysis", "path": path}


# ---------- 工具 3：执行规划 ----------
def plan_trade_execution(
    symbol: str,
    target_position: float,
    current_price: float = None,
    total_capital: float = 1000000,
    execution_days: int = 5,
    market_state: str = "neutral",
    volatility_estimate: float = None,
) -> dict:
    """
    制定分批建仓/减仓计划。
    :param symbol: 单只股票代码
    :param target_position: 目标仓位金额
    :param current_price: 当前价格，为空则自动获取
    :param total_capital: 总资金
    :param execution_days: 执行天数
    :param market_state: 市场状态，bull/bear/neutral
    :param volatility_estimate: 波动率估计，为空则自动计算
    :return: 执行计划
    """
    print(f"制定执行计划... {symbol}")

    code = _normalize_symbol(symbol)

    # 获取当前价格
    if current_price is None:
        quote_result = _get_current_quote(code)
        if "error" in quote_result:
            return quote_result
        current_price = quote_result["close"]

    # 计算目标股数（A股一手100股）
    target_shares = int(target_position / current_price / 100) * 100
    actual_position = target_shares * current_price

    # 获取波动率估计
    if volatility_estimate is None:
        volatility_estimate = _estimate_volatility(code)
        if volatility_estimate is None:
            volatility_estimate = 0.25  # 默认25%年化波动率

    # 根据市场状态调整执行节奏
    if market_state == "bull":
        # 牛市：快速建仓，避免踏空
        daily_weights = [0.3, 0.25, 0.2, 0.15, 0.1][:execution_days]
    elif market_state == "bear":
        # 熊市：慢速建仓，等待更好价格
        daily_weights = [0.1, 0.15, 0.2, 0.25, 0.3][:execution_days]
    else:
        # 中性：均匀建仓
        daily_weights = [1.0 / execution_days] * execution_days

    # 归一化权重
    total_weight = sum(daily_weights)
    daily_weights = [w / total_weight for w in daily_weights]

    # 构建执行计划
    execution_plan = []
    remaining_shares = target_shares

    for day, weight in enumerate(daily_weights, 1):
        if day == execution_days:
            # 最后一天执行剩余全部
            shares = remaining_shares
        else:
            shares = int(target_shares * weight / 100) * 100
            remaining_shares -= shares

        if shares <= 0:
            continue

        execution_plan.append({
            "day": day,
            "shares": shares,
            "estimated_price": current_price,  # 简化假设
            "estimated_amount": round(shares * current_price, 2),
            "cumulative_pct": round(sum(daily_weights[:day]) * 100, 1),
        })

    # 计算风控参数
    stop_loss_pct = volatility_estimate * 2  # 2倍波动率作为止损
    stop_loss_price = current_price * (1 - stop_loss_pct)
    take_profit_pct = volatility_estimate * 3  # 3倍波动率作为止盈
    take_profit_price = current_price * (1 + take_profit_pct)

    result = {
        "symbol": symbol,
        "code": code,
        "current_price": current_price,
        "target_position": target_position,
        "actual_position": round(actual_position, 2),
        "target_shares": target_shares,
        "position_pct": round(actual_position / total_capital * 100, 2),
        "execution_plan": execution_plan,
        "market_state": market_state,
        "risk_controls": {
            "stop_loss_price": round(stop_loss_price, 2),
            "stop_loss_pct": round(stop_loss_pct * 100, 2),
            "take_profit_price": round(take_profit_price, 2),
            "take_profit_pct": round(take_profit_pct * 100, 2),
            "volatility_estimate": round(volatility_estimate * 100, 2),
        },
        "execution_tips": [],
    }

    # 执行建议
    if market_state == "bull":
        result["execution_tips"].append("市场偏强，建议前几日快速建仓，避免踏空风险")
    elif market_state == "bear":
        result["execution_tips"].append("市场偏弱，建议后几日逐步建仓，等待更好价格")
    else:
        result["execution_tips"].append("市场中性，均匀建仓，控制执行成本")

    if target_shares * current_price > total_capital * 0.3:
        result["execution_tips"].append("注意：单标的仓位超过30%，建议分批执行以降低冲击成本")

    result["execution_tips"].append(f"建议止损位：{stop_loss_price:.2f}元（-{stop_loss_pct*100:.1f}%）")
    result["execution_tips"].append(f"建议止盈位：{take_profit_price:.2f}元（+{take_profit_pct*100:.1f}%）")

    return result


def _get_current_quote(code: str) -> dict:
    """获取当前报价。"""
    if not _ensure_login():
        return {"error": "数据服务登录失败"}

    try:
        end = datetime.now()
        start = end - timedelta(days=10)
        rs = bs.query_history_k_data_plus(
            code,
            "date,close,open,high,low",
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
            return {"error": "无法获取最新价格"}

        df = pd.DataFrame(rows, columns=rs.fields)
        latest = df.iloc[-1]
        return {
            "date": latest["date"],
            "close": float(latest["close"]),
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
        }
    except Exception as e:
        _ensure_logout()
        return {"error": f"获取价格异常: {str(e)}"}


def _estimate_volatility(code: str, days: int = 60) -> float:
    """估计年化波动率。"""
    if not _ensure_login():
        return None

    try:
        end = datetime.now()
        start = end - timedelta(days=days * 2)  # 多取一些数据
        rs = bs.query_history_k_data_plus(
            code,
            "date,close",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",
        )
        if rs.error_code != "0":
            _ensure_logout()
            return None

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if len(rows) < days:
            return None

        df = pd.DataFrame(rows, columns=rs.fields)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        returns = df["close"].pct_change().dropna()

        if len(returns) < days // 2:
            return None

        return float(returns.std() * np.sqrt(252))
    except Exception:
        _ensure_logout()
        return None


# ---------- 工具 4：持仓快照 ----------
def get_portfolio_snapshot(
    symbols: str,
    shares: str,
    cost_prices: str = None,
) -> dict:
    """
    获取持仓快照，计算当前市值、盈亏等。
    :param symbols: 股票代码，逗号分隔
    :param shares: 持仓股数，逗号分隔
    :param cost_prices: 成本价，逗号分隔，可选
    :return: 持仓快照
    """
    print(f"获取持仓快照...")

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    share_list = [int(s.strip()) for s in shares.split(",")]

    if len(symbol_list) != len(share_list):
        return {"error": "股票数量与持仓数量不匹配"}

    if cost_prices:
        cost_list = [float(p.strip()) for p in cost_prices.split(",")]
        if len(cost_list) != len(symbol_list):
            return {"error": "成本价数量与股票数量不匹配"}
    else:
        cost_list = [None] * len(symbol_list)

    if not _ensure_login():
        return {"error": "数据服务登录失败"}

    try:
        positions = []
        total_market_value = 0
        total_cost = 0
        total_pnl = 0

        for i, symbol in enumerate(symbol_list):
            code = _normalize_symbol(symbol)
            share = share_list[i]
            cost = cost_list[i]

            # 获取最新价格
            end = datetime.now()
            start = end - timedelta(days=10)
            rs = bs.query_history_k_data_plus(
                code,
                "date,close,code_name",
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="3",
            )
            if rs.error_code != "0":
                continue

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                continue

            df = pd.DataFrame(rows, columns=rs.fields)
            latest = df.iloc[-1]
            current_price = float(latest["close"])
            code_name = latest.get("code_name", symbol)

            market_value = share * current_price
            total_market_value += market_value

            position = {
                "symbol": symbol,
                "code": code,
                "name": code_name,
                "shares": share,
                "current_price": current_price,
                "market_value": round(market_value, 2),
            }

            if cost:
                cost_value = share * cost
                pnl = market_value - cost_value
                pnl_pct = (current_price - cost) / cost * 100

                position["cost_price"] = cost
                position["cost_value"] = round(cost_value, 2)
                position["pnl"] = round(pnl, 2)
                position["pnl_pct"] = round(pnl_pct, 2)

                total_cost += cost_value
                total_pnl += pnl

            positions.append(position)

        _ensure_logout()

        result = {
            "positions": positions,
            "total_market_value": round(total_market_value, 2),
            "position_count": len(positions),
        }

        if total_cost > 0:
            result["total_cost"] = round(total_cost, 2)
            result["total_pnl"] = round(total_pnl, 2)
            result["total_pnl_pct"] = round(total_pnl / total_cost * 100, 2)

        return result

    except Exception as e:
        _ensure_logout()
        return {"error": f"获取持仓快照异常: {str(e)}"}


if __name__ == "__main__":
    print("测试交易工具...")

    # 测试组合优化
    print("\n=== 组合优化 ===")
    result = optimize_portfolio(
        symbols="600000,000001,300750",
        total_capital=500000,
        with_plots=False,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 测试风险计算
    print("\n=== 风险计算 ===")
    result = calculate_portfolio_risk(
        symbols="600000,000001",
        total_capital=500000,
        with_plots=False,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 测试执行规划
    print("\n=== 执行规划 ===")
    result = plan_trade_execution(
        symbol="600000",
        target_position=100000,
        total_capital=500000,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
