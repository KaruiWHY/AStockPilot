# -*- coding: utf-8 -*-
"""
财报分析工具层：基于 baostock 获取 A 股财务报表数据。
支持资产负债表、利润表、现金流量表查询，财务指标计算，成长性分析与杜邦分析。
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


def _safe_float(value):
    """安全转换为浮点数。"""
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _format_report_date(year: int, quarter: int) -> str:
    """格式化报告期。"""
    quarter_map = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
    return f"{year}{quarter_map.get(quarter, '1231')}"


# ---------- 工具 1：资产负债表（基于偿债能力数据） ----------
def get_balance_sheet(symbol: str, year: int = None, quarter: int = None) -> dict:
    """
    获取资产负债表相关数据（基于 baostock query_balance_data）。
    :param symbol: 股票代码，如 600000、000001
    :param year: 报告年份，默认取最近一年
    :param quarter: 报告季度 1-4，默认取第4季度（年报）
    :return: 资产负债表数据 dict
    """
    print(f"获取资产负债表... {symbol}")
    code = _normalize_symbol(symbol)

    if year is None:
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month < 5:
            year = current_year - 2
        else:
            year = current_year - 1
    if quarter is None:
        quarter = 4

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        # 使用 query_balance_data 获取偿债能力数据
        rs = bs.query_balance_data(
            code=code,
            year=year,
            quarter=quarter,
        )
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未获取到 {symbol} {year}年Q{quarter}的资产负债表数据。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        record = df.iloc[0].to_dict()

        # 格式化输出
        result = {
            "symbol": symbol,
            "code": code,
            "report_date": f"{year}-Q{quarter}",
            "pub_date": record.get("pubDate"),
            "stat_date": record.get("statDate"),
        }

        # baostock balance_data 返回的字段
        # currentRatio, quickRatio, cashRatio, YOYLiability, liabilityToAsset, assetToEquity
        key_metrics = [
            ("current_ratio", "流动比率", record.get("currentRatio")),
            ("quick_ratio", "速动比率", record.get("quickRatio")),
            ("cash_ratio", "现金比率", record.get("cashRatio")),
            ("yoy_liability", "负债同比增长率", record.get("YOYLiability")),
            ("liability_to_asset", "资产负债率", record.get("liabilityToAsset")),
            ("asset_to_equity", "权益乘数", record.get("assetToEquity")),
        ]

        result["metrics"] = []
        for key, name, value in key_metrics:
            float_val = _safe_float(value)
            result["metrics"].append({
                "key": key,
                "name": name,
                "value": float_val,
            })

        # 计算关键比率评估
        liability_to_asset = _safe_float(record.get("liabilityToAsset"))
        current_ratio = _safe_float(record.get("currentRatio"))
        asset_to_equity = _safe_float(record.get("assetToEquity"))

        result["ratios"] = {
            "liability_to_asset": liability_to_asset,
            "current_ratio": current_ratio,
            "equity_multiplier": asset_to_equity,
        }

        # 偿债能力评估
        result["evaluation"] = {}

        if liability_to_asset is not None:
            if liability_to_asset <= 0.5:
                result["evaluation"]["debt_level"] = "资产负债率≤50%，财务风险较低"
            elif liability_to_asset <= 0.7:
                result["evaluation"]["debt_level"] = "资产负债率在50%-70%，财务杠杆适中"
            else:
                result["evaluation"]["debt_level"] = "资产负债率>70%，财务风险较高"

        if current_ratio is not None and current_ratio > 0:
            if current_ratio >= 2:
                result["evaluation"]["liquidity"] = "流动比率≥2，短期偿债能力强"
            elif current_ratio >= 1:
                result["evaluation"]["liquidity"] = "流动比率在1-2之间，短期偿债能力正常"
            else:
                result["evaluation"]["liquidity"] = "流动比率<1，短期偿债压力较大"
        else:
            result["evaluation"]["liquidity"] = "流动比率数据缺失"

        return result

    except Exception as e:
        _ensure_logout()
        return {"error": f"获取资产负债表异常: {str(e)}"}


# ---------- 工具 2：利润表（基于盈利能力数据） ----------
def get_income_statement(symbol: str, year: int = None, quarter: int = None) -> dict:
    """
    获取利润表相关数据（基于 baostock query_profit_data）。
    :param symbol: 股票代码
    :param year: 报告年份，默认取最近一年
    :param quarter: 报告季度 1-4，默认取第4季度（年报）
    :return: 利润表数据 dict
    """
    print(f"获取利润表... {symbol}")
    code = _normalize_symbol(symbol)

    if year is None:
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month < 5:
            year = current_year - 2
        else:
            year = current_year - 1
    if quarter is None:
        quarter = 4

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        # 使用 query_profit_data 获取盈利能力数据
        rs = bs.query_profit_data(
            code=code,
            year=year,
            quarter=quarter,
        )
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未获取到 {symbol} {year}年Q{quarter}的利润表数据。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        record = df.iloc[0].to_dict()

        result = {
            "symbol": symbol,
            "code": code,
            "report_date": f"{year}-Q{quarter}",
            "pub_date": record.get("pubDate"),
            "stat_date": record.get("statDate"),
        }

        # baostock profit_data 返回的字段
        # roeAvg, npMargin, gpMargin, netProfit, epsTTM, MBRevenue, totalShare, liqaShare
        key_metrics = [
            ("net_profit", "净利润", record.get("netProfit")),
            ("revenue", "营业收入", record.get("MBRevenue")),
            ("eps_ttm", "每股收益TTM", record.get("epsTTM")),
            ("total_share", "总股本", record.get("totalShare")),
            ("liquid_share", "流通股本", record.get("liqaShare")),
        ]

        result["metrics"] = []
        for key, name, value in key_metrics:
            float_val = _safe_float(value)
            result["metrics"].append({
                "key": key,
                "name": name,
                "value": float_val,
            })

        # 利润率指标
        roe_avg = _safe_float(record.get("roeAvg"))
        np_margin = _safe_float(record.get("npMargin"))
        gp_margin = _safe_float(record.get("gpMargin"))

        result["ratios"] = {
            "roe": roe_avg,
            "net_profit_margin": np_margin,
            "gross_profit_margin": gp_margin,
        }

        # 盈利能力评估
        result["evaluation"] = {}

        if np_margin is not None:
            if np_margin >= 0.2:
                result["evaluation"]["profit_margin"] = "净利率≥20%，盈利能力强"
            elif np_margin >= 0.1:
                result["evaluation"]["profit_margin"] = "净利率在10%-20%，盈利能力良好"
            elif np_margin >= 0.05:
                result["evaluation"]["profit_margin"] = "净利率在5%-10%，盈利能力一般"
            else:
                result["evaluation"]["profit_margin"] = "净利率<5%，盈利能力较弱"

        if roe_avg is not None:
            if roe_avg >= 0.15:
                result["evaluation"]["roe_level"] = "ROE≥15%，股东回报优秀"
            elif roe_avg >= 0.10:
                result["evaluation"]["roe_level"] = "ROE在10%-15%，股东回报良好"
            elif roe_avg >= 0.05:
                result["evaluation"]["roe_level"] = "ROE在5%-10%，股东回报一般"
            else:
                result["evaluation"]["roe_level"] = "ROE<5%，股东回报较弱"

        return result

    except Exception as e:
        _ensure_logout()
        return {"error": f"获取利润表异常: {str(e)}"}


# ---------- 工具 3：现金流量表（基于现金流比率数据） ----------
def get_cash_flow_statement(symbol: str, year: int = None, quarter: int = None) -> dict:
    """
    获取现金流量表相关数据（基于 baostock query_cash_flow_data）。
    :param symbol: 股票代码
    :param year: 报告年份，默认取最近一年
    :param quarter: 报告季度 1-4，默认取第4季度（年报）
    :return: 现金流量表数据 dict
    """
    print(f"获取现金流量表... {symbol}")
    code = _normalize_symbol(symbol)

    if year is None:
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month < 5:
            year = current_year - 2
        else:
            year = current_year - 1
    if quarter is None:
        quarter = 4

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        rs = bs.query_cash_flow_data(
            code=code,
            year=year,
            quarter=quarter,
        )
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未获取到 {symbol} {year}年Q{quarter}的现金流量表数据。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        record = df.iloc[0].to_dict()

        result = {
            "symbol": symbol,
            "code": code,
            "report_date": f"{year}-Q{quarter}",
            "pub_date": record.get("pubDate"),
            "stat_date": record.get("statDate"),
        }

        # baostock cash_flow_data 返回的字段
        # CAToAsset, NCAToAsset, tangibleAssetToAsset, ebitToInterest, CFOToOR, CFOToNP, CFOToGr
        key_metrics = [
            ("ca_to_asset", "流动资产占比", record.get("CAToAsset")),
            ("nca_to_asset", "非流动资产占比", record.get("NCAToAsset")),
            ("tangible_asset_to_asset", "有形资产占比", record.get("tangibleAssetToAsset")),
            ("ebit_to_interest", "EBIT利息保障倍数", record.get("ebitToInterest")),
            ("cfo_to_or", "经营现金流/营业收入", record.get("CFOToOR")),
            ("cfo_to_np", "经营现金流/净利润", record.get("CFOToNP")),
            ("cfo_to_gr", "经营现金流/总营收", record.get("CFOToGr")),
        ]

        result["metrics"] = []
        for key, name, value in key_metrics:
            float_val = _safe_float(value)
            result["metrics"].append({
                "key": key,
                "name": name,
                "value": float_val,
            })

        # 现金流质量分析
        cfo_to_np = _safe_float(record.get("CFOToNP"))
        cfo_to_or = _safe_float(record.get("CFOToOR"))

        result["analysis"] = {}

        if cfo_to_np is not None:
            if cfo_to_np > 1:
                result["analysis"]["cash_quality"] = "经营现金流/净利润>1，盈利质量高，利润含金量足"
            elif cfo_to_np > 0:
                result["analysis"]["cash_quality"] = "经营现金流/净利润在0-1之间，盈利质量一般"
            else:
                result["analysis"]["cash_quality"] = "经营现金流/净利润<0，需关注现金流状况"

        if cfo_to_or is not None:
            if cfo_to_or > 0:
                result["analysis"]["revenue_quality"] = "经营现金流/营业收入>0，收入转化为现金能力良好"
            else:
                result["analysis"]["revenue_quality"] = "经营现金流/营业收入<0，收入现金转化需关注"

        return result

    except Exception as e:
        _ensure_logout()
        return {"error": f"获取现金流量表异常: {str(e)}"}


# ---------- 工具 4：盈利能力指标 ----------
def get_profitability_indicators(symbol: str, year: int = None, quarter: int = None) -> dict:
    """
    获取盈利能力指标。
    :param symbol: 股票代码
    :param year: 报告年份
    :param quarter: 报告季度
    :return: 盈利能力指标 dict
    """
    print(f"获取盈利能力指标... {symbol}")
    code = _normalize_symbol(symbol)

    if year is None:
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month < 5:
            year = current_year - 2
        else:
            year = current_year - 1
    if quarter is None:
        quarter = 4

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        rs = bs.query_profit_data(
            code=code,
            year=year,
            quarter=quarter,
        )
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未获取到 {symbol} {year}年Q{quarter}的盈利能力数据。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        record = df.iloc[0].to_dict()

        result = {
            "symbol": symbol,
            "code": code,
            "report_date": f"{year}-Q{quarter}",
            "pub_date": record.get("pubDate"),
            "stat_date": record.get("statDate"),
        }

        # baostock profit_data 返回字段: roeAvg, npMargin, gpMargin, netProfit, epsTTM, MBRevenue, totalShare, liqaShare
        # 注意：roeAvg, npMargin, gpMargin 是小数形式，需要转换为百分比
        roe_val = _safe_float(record.get("roeAvg"))
        np_margin_val = _safe_float(record.get("npMargin"))
        gp_margin_val = _safe_float(record.get("gpMargin"))

        indicators = [
            ("roe", "净资产收益率ROE(%)", roe_val * 100 if roe_val is not None else None),
            ("net_profit_margin", "销售净利率(%)", np_margin_val * 100 if np_margin_val is not None else None),
            ("gross_profit_margin", "销售毛利率(%)", gp_margin_val * 100 if gp_margin_val is not None else None),
            ("net_profit", "净利润", _safe_float(record.get("netProfit"))),
            ("eps_ttm", "每股收益TTM", _safe_float(record.get("epsTTM"))),
            ("revenue", "营业收入", _safe_float(record.get("MBRevenue"))),
            ("total_share", "总股本", _safe_float(record.get("totalShare"))),
        ]

        result["indicators"] = []
        for key, name, value in indicators:
            result["indicators"].append({
                "key": key,
                "name": name,
                "value": value,
            })

        # ROE 评估 (roeAvg 是小数形式，如 0.061950)
        roe = _safe_float(record.get("roeAvg"))
        if roe is not None:
            roe_pct = roe * 100  # 转换为百分比
            if roe_pct >= 15:
                result["roe_evaluation"] = f"优秀，ROE={roe_pct:.2f}%≥15%，股东回报率较高"
            elif roe_pct >= 10:
                result["roe_evaluation"] = f"良好，ROE={roe_pct:.2f}%在10%-15%之间"
            elif roe_pct >= 5:
                result["roe_evaluation"] = f"一般，ROE={roe_pct:.2f}%在5%-10%之间"
            else:
                result["roe_evaluation"] = f"较弱，ROE={roe_pct:.2f}%<5%需关注盈利能力"
        else:
            result["roe_evaluation"] = "ROE数据缺失"

        return result

    except Exception as e:
        _ensure_logout()
        return {"error": f"获取盈利能力指标异常: {str(e)}"}


# ---------- 工具 5：偿债能力指标 ----------
def get_solvency_indicators(symbol: str, year: int = None, quarter: int = None) -> dict:
    """
    获取偿债能力指标。
    :param symbol: 股票代码
    :param year: 报告年份
    :param quarter: 报告季度
    :return: 偿债能力指标 dict
    """
    print(f"获取偿债能力指标... {symbol}")
    code = _normalize_symbol(symbol)

    if year is None:
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month < 5:
            year = current_year - 2
        else:
            year = current_year - 1
    if quarter is None:
        quarter = 4

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        # 使用 query_balance_data 获取偿债能力数据
        rs = bs.query_balance_data(
            code=code,
            year=year,
            quarter=quarter,
        )
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未获取到 {symbol} {year}年Q{quarter}的偿债能力数据。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        record = df.iloc[0].to_dict()

        result = {
            "symbol": symbol,
            "code": code,
            "report_date": f"{year}-Q{quarter}",
            "pub_date": record.get("pubDate"),
            "stat_date": record.get("statDate"),
        }

        # baostock balance_data 返回字段: currentRatio, quickRatio, cashRatio, YOYLiability, liabilityToAsset, assetToEquity
        indicators = [
            ("current_ratio", "流动比率", _safe_float(record.get("currentRatio"))),
            ("quick_ratio", "速动比率", _safe_float(record.get("quickRatio"))),
            ("cash_ratio", "现金比率", _safe_float(record.get("cashRatio"))),
            ("yoy_liability", "负债同比增长率(%)", _safe_float(record.get("YOYLiability"))),
            ("asset_liability_ratio", "资产负债率(%)", _safe_float(record.get("liabilityToAsset"))),
            ("equity_multiplier", "权益乘数", _safe_float(record.get("assetToEquity"))),
        ]

        result["indicators"] = []
        for key, name, value in indicators:
            result["indicators"].append({
                "key": key,
                "name": name,
                "value": value,
            })

        # 偿债能力评估
        current_ratio = _safe_float(record.get("currentRatio"))
        quick_ratio = _safe_float(record.get("quickRatio"))
        debt_ratio = _safe_float(record.get("liabilityToAsset"))

        result["evaluation"] = {}

        if current_ratio is not None and current_ratio > 0:
            if current_ratio >= 2:
                result["evaluation"]["liquidity"] = "流动比率≥2，短期偿债能力强"
            elif current_ratio >= 1:
                result["evaluation"]["liquidity"] = "流动比率在1-2之间，短期偿债能力正常"
            else:
                result["evaluation"]["liquidity"] = "流动比率<1，短期偿债压力较大"
        else:
            result["evaluation"]["liquidity"] = "流动比率数据缺失"

        if quick_ratio is not None and quick_ratio > 0:
            if quick_ratio >= 1:
                result["evaluation"]["quick_liquidity"] = "速动比率≥1，快速变现能力良好"
            else:
                result["evaluation"]["quick_liquidity"] = "速动比率<1，需关注快速变现能力"
        else:
            result["evaluation"]["quick_liquidity"] = "速动比率数据缺失"

        if debt_ratio is not None:
            debt_pct = debt_ratio * 100  # 转换为百分比
            if debt_pct <= 50:
                result["evaluation"]["debt_level"] = f"资产负债率={debt_pct:.2f}%≤50%，财务风险较低"
            elif debt_pct <= 70:
                result["evaluation"]["debt_level"] = f"资产负债率={debt_pct:.2f}%在50%-70%，财务杠杆适中"
            else:
                result["evaluation"]["debt_level"] = f"资产负债率={debt_pct:.2f}%>70%，财务风险较高"
        else:
            result["evaluation"]["debt_level"] = "资产负债率数据缺失"

        return result

    except Exception as e:
        _ensure_logout()
        return {"error": f"获取偿债能力指标异常: {str(e)}"}


# ---------- 工具 6：成长能力指标 ----------
def get_growth_indicators(symbol: str, year: int = None, quarter: int = None) -> dict:
    """
    获取成长能力指标。
    :param symbol: 股票代码
    :param year: 报告年份
    :param quarter: 报告季度
    :return: 成长能力指标 dict
    """
    print(f"获取成长能力指标... {symbol}")
    code = _normalize_symbol(symbol)

    if year is None:
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month < 5:
            year = current_year - 2
        else:
            year = current_year - 1
    if quarter is None:
        quarter = 4

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        rs = bs.query_growth_data(
            code=code,
            year=year,
            quarter=quarter,
        )
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未获取到 {symbol} {year}年Q{quarter}的成长能力数据。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        record = df.iloc[0].to_dict()

        result = {
            "symbol": symbol,
            "code": code,
            "report_date": f"{year}-Q{quarter}",
            "pub_date": record.get("pubDate"),
            "stat_date": record.get("statDate"),
        }

        # baostock growth_data 返回字段: YOYEquity, YOYAsset, YOYNI, YOYEPSBasic, YOYPNI
        # YOYEquity: 净资产同比增长率
        # YOYAsset: 总资产同比增长率
        # YOYNI: 净利润同比增长率
        # YOYEPSBasic: 基本每股收益同比增长率
        # YOYPNI: 归属母公司股东净利润同比增长率
        # 注意：这些字段是小数形式，需要转换为百分比
        yoy_equity = _safe_float(record.get("YOYEquity"))
        yoy_asset = _safe_float(record.get("YOYAsset"))
        yoy_ni = _safe_float(record.get("YOYNI"))
        yoy_eps = _safe_float(record.get("YOYEPSBasic"))
        yoy_pni = _safe_float(record.get("YOYPNI"))

        indicators = [
            ("yoy_equity", "净资产同比增长率(%)", yoy_equity * 100 if yoy_equity is not None else None),
            ("yoy_asset", "总资产同比增长率(%)", yoy_asset * 100 if yoy_asset is not None else None),
            ("yoy_net_profit", "净利润同比增长率(%)", yoy_ni * 100 if yoy_ni is not None else None),
            ("yoy_eps", "每股收益同比增长率(%)", yoy_eps * 100 if yoy_eps is not None else None),
            ("yoy_parent_net_profit", "归属母公司净利润同比增长率(%)", yoy_pni * 100 if yoy_pni is not None else None),
        ]

        result["indicators"] = []
        for key, name, value in indicators:
            result["indicators"].append({
                "key": key,
                "name": name,
                "value": value,
            })

        # 成长性评估
        asset_growth = _safe_float(record.get("YOYAsset"))
        profit_growth = _safe_float(record.get("YOYNI"))

        result["evaluation"] = {}

        if profit_growth is not None:
            profit_pct = profit_growth * 100  # 转换为百分比
            if profit_pct > 30:
                result["evaluation"]["profit_growth"] = f"净利润增长{profit_pct:.2f}%，高速增长"
            elif profit_pct > 10:
                result["evaluation"]["profit_growth"] = f"净利润增长{profit_pct:.2f}%，稳定增长"
            elif profit_pct > 0:
                result["evaluation"]["profit_growth"] = f"净利润增长{profit_pct:.2f}%，低速增长"
            else:
                result["evaluation"]["profit_growth"] = f"净利润增长{profit_pct:.2f}%，负增长需关注"

        if asset_growth is not None:
            asset_pct = asset_growth * 100
            if asset_pct > 20:
                result["evaluation"]["asset_growth"] = f"总资产增长{asset_pct:.2f}%，扩张较快"
            elif asset_pct > 0:
                result["evaluation"]["asset_growth"] = f"总资产增长{asset_pct:.2f}%，稳健扩张"
            else:
                result["evaluation"]["asset_growth"] = f"总资产增长{asset_pct:.2f}%，资产收缩"

        # 综合成长性评价
        if profit_growth is not None and asset_growth is not None:
            if profit_growth > 0.2 and asset_growth > 0.1:
                result["evaluation"]["overall"] = "高速成长期，利润和资产双增长"
            elif profit_growth > 0.1:
                result["evaluation"]["overall"] = "稳定成长期，利润增长良好"
            elif profit_growth > 0:
                result["evaluation"]["overall"] = "低速成长期，利润正增长"
            else:
                result["evaluation"]["overall"] = "需关注成长性，利润负增长"

        return result

    except Exception as e:
        _ensure_logout()
        return {"error": f"获取成长能力指标异常: {str(e)}"}


# ---------- 工具 7：杜邦分析 ----------
def get_dupont_analysis(symbol: str, year: int = None, quarter: int = None) -> dict:
    """
    杜邦分析：将ROE分解为净利率、资产周转率和权益乘数。
    :param symbol: 股票代码
    :param year: 报告年份
    :param quarter: 报告季度
    :return: 杜邦分析结果 dict
    """
    print(f"杜邦分析... {symbol}")
    code = _normalize_symbol(symbol)

    if year is None:
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month < 5:
            year = current_year - 2
        else:
            year = current_year - 1
    if quarter is None:
        quarter = 4

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    try:
        rs = bs.query_dupont_data(
            code=code,
            year=year,
            quarter=quarter,
        )
        if rs.error_code != "0":
            _ensure_logout()
            return {"error": f"查询失败: {rs.error_msg}"}

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        _ensure_logout()

        if not rows:
            return {"error": f"未获取到 {symbol} {year}年Q{quarter}的杜邦分析数据。"}

        df = pd.DataFrame(rows, columns=rs.fields)
        record = df.iloc[0].to_dict()

        result = {
            "symbol": symbol,
            "code": code,
            "report_date": f"{year}-Q{quarter}",
            "pub_date": record.get("pubDate"),
            "stat_date": record.get("statDate"),
        }

        # baostock dupont_data 返回字段:
        # dupontROE, dupontAssetStoEquity, dupontAssetTurn, dupontPnitoni, dupontNitogr, dupontTaxBurden, dupontIntburden, dupontEbittogr
        dupont_metrics = [
            ("roe", "净资产收益率ROE(%)", _safe_float(record.get("dupontROE"))),
            ("equity_multiplier", "权益乘数", _safe_float(record.get("dupontAssetStoEquity"))),
            ("asset_turnover", "总资产周转率(次)", _safe_float(record.get("dupontAssetTurn"))),
            ("net_profit_to_net_income", "归属母公司净利润/净利润", _safe_float(record.get("dupontPnitoni"))),
            ("net_profit_to_revenue", "销售净利率(%)", _safe_float(record.get("dupontNitogr"))),
            ("tax_burden", "税收负担比率", _safe_float(record.get("dupontTaxBurden"))),
        ]

        result["dupont_metrics"] = []
        for key, name, value in dupont_metrics:
            result["dupont_metrics"].append({
                "key": key,
                "name": name,
                "value": value,
            })

        # 杜邦分解说明
        roe = _safe_float(record.get("dupontROE"))
        em = _safe_float(record.get("dupontAssetStoEquity"))
        at = _safe_float(record.get("dupontAssetTurn"))
        npm = _safe_float(record.get("dupontNitogr"))

        result["analysis"] = {
            "formula": "ROE = 净利率 × 总资产周转率 × 权益乘数",
        }

        if all(v is not None for v in [roe, em, at, npm]):
            roe_pct = roe * 100 if roe < 1 else roe
            npm_pct = npm * 100 if npm < 1 else npm
            result["analysis"]["calculation"] = f"ROE={roe_pct:.2f}% ≈ 净利率{npm_pct:.2f}% × 资产周转率{at:.4f} × 权益乘数{em:.2f}"

            # 驱动因素分析
            drivers = []
            if npm_pct >= 15:
                drivers.append("净利率较高，盈利能力强")
            elif npm_pct >= 10:
                drivers.append("净利率适中")
            elif npm_pct >= 5:
                drivers.append("净利率偏低，需关注成本费用控制")
            else:
                drivers.append("净利率过低，盈利能力堪忧")

            if at >= 1:
                drivers.append("资产周转效率良好")
            elif at >= 0.5:
                drivers.append("资产周转效率一般")
            else:
                drivers.append("资产周转效率较低，资产利用有待提升")

            if em >= 3:
                drivers.append("财务杠杆较高，需关注财务风险")
            elif em >= 2:
                drivers.append("财务杠杆适中")
            else:
                drivers.append("财务杠杆较低，可适度利用杠杆提升ROE")

            result["analysis"]["drivers"] = drivers

        return result

    except Exception as e:
        _ensure_logout()
        return {"error": f"杜邦分析异常: {str(e)}"}


# ---------- 工具 8：多期财报对比 ----------
def get_financial_report_history(symbol: str, years: int = 3, with_plots: bool = False) -> dict:
    """
    获取多期财报数据对比。
    :param symbol: 股票代码
    :param years: 对比年数，默认3年
    :param with_plots: 是否生成趋势图
    :return: 多期财报对比数据
    """
    print(f"获取多期财报对比... {symbol}")
    code = _normalize_symbol(symbol)

    if not _ensure_login():
        return {"error": "数据服务登录失败，请稍后重试。"}

    current_year = datetime.now().year
    start_year = current_year - years

    result = {
        "symbol": symbol,
        "code": code,
        "years": list(range(start_year, current_year)),
        "reports": [],
    }

    try:
        for year in range(start_year, current_year):
            # 获取盈利能力
            rs_profit = bs.query_profit_data(code=code, year=year, quarter=4)
            profit_data = {}
            if rs_profit.error_code == "0":
                rows = []
                while rs_profit.next():
                    rows.append(rs_profit.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=rs_profit.fields)
                    profit_data = df.iloc[0].to_dict()

            # 获取成长能力
            rs_growth = bs.query_growth_data(code=code, year=year, quarter=4)
            growth_data = {}
            if rs_growth.error_code == "0":
                rows = []
                while rs_growth.next():
                    rows.append(rs_growth.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=rs_growth.fields)
                    growth_data = df.iloc[0].to_dict()

            # 获取偿债能力 - query_balance_data 返回 currentRatio, liabilityToAsset, assetToEquity
            rs_balance = bs.query_balance_data(code=code, year=year, quarter=4)
            balance_data = {}
            if rs_balance.error_code == "0":
                rows = []
                while rs_balance.next():
                    rows.append(rs_balance.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=rs_balance.fields)
                    balance_data = df.iloc[0].to_dict()

            # 转换为百分比显示
            roe = _safe_float(profit_data.get("roeAvg"))
            np_margin = _safe_float(profit_data.get("npMargin"))
            gp_margin = _safe_float(profit_data.get("gpMargin"))
            profit_growth = _safe_float(growth_data.get("YOYNI"))
            asset_growth = _safe_float(growth_data.get("YOYAsset"))
            current_ratio = _safe_float(balance_data.get("currentRatio"))
            debt_ratio = _safe_float(balance_data.get("liabilityToAsset"))

            report = {
                "year": year,
                "roe": roe * 100 if roe is not None and roe < 1 else roe,  # 转为百分比
                "net_profit_margin": np_margin * 100 if np_margin is not None and np_margin < 1 else np_margin,
                "gross_profit_margin": gp_margin * 100 if gp_margin is not None and gp_margin < 1 else gp_margin,
                "profit_growth": profit_growth * 100 if profit_growth is not None and profit_growth < 1 else profit_growth,
                "asset_growth": asset_growth * 100 if asset_growth is not None and asset_growth < 1 else asset_growth,
                "current_ratio": current_ratio,
                "debt_ratio": debt_ratio * 100 if debt_ratio is not None and debt_ratio < 1 else debt_ratio,
            }
            result["reports"].append(report)

        _ensure_logout()

        if with_plots and result["reports"]:
            plot_result = _build_financial_trend_plot(symbol, result["reports"])
            result["plot"] = plot_result

        return result

    except Exception as e:
        _ensure_logout()
        return {"error": f"获取多期财报对比异常: {str(e)}"}


def _build_financial_trend_plot(symbol: str, reports: list) -> dict:
    """生成财务指标趋势图。"""
    if not reports:
        return {"error": "无数据可绘图"}

    df = pd.DataFrame(reports)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("ROE趋势", "利润率趋势", "成长性趋势", "偿债能力趋势"),
    )

    years = df["year"].tolist()

    # ROE趋势
    if "roe" in df.columns:
        fig.add_trace(
            go.Scatter(x=years, y=df["roe"], mode="lines+markers", name="ROE(%)", line={"color": "#1f77b4"}),
            row=1, col=1,
        )

    # 利润率趋势
    if "net_profit_margin" in df.columns:
        fig.add_trace(
            go.Scatter(x=years, y=df["net_profit_margin"], mode="lines+markers", name="净利率(%)", line={"color": "#ff7f0e"}),
            row=1, col=2,
        )
    if "gross_profit_margin" in df.columns:
        fig.add_trace(
            go.Scatter(x=years, y=df["gross_profit_margin"], mode="lines+markers", name="毛利率(%)", line={"color": "#2ca02c"}),
            row=1, col=2,
        )

    # 成长性趋势
    if "revenue_growth" in df.columns:
        fig.add_trace(
            go.Scatter(x=years, y=df["revenue_growth"], mode="lines+markers", name="营收增长(%)", line={"color": "#d62728"}),
            row=2, col=1,
        )
    if "profit_growth" in df.columns:
        fig.add_trace(
            go.Scatter(x=years, y=df["profit_growth"], mode="lines+markers", name="利润增长(%)", line={"color": "#9467bd"}),
            row=2, col=1,
        )

    # 偿债能力趋势
    if "current_ratio" in df.columns:
        fig.add_trace(
            go.Scatter(x=years, y=df["current_ratio"], mode="lines+markers", name="流动比率", line={"color": "#8c564b"}),
            row=2, col=2,
        )
    if "debt_ratio" in df.columns:
        fig.add_trace(
            go.Scatter(x=years, y=df["debt_ratio"], mode="lines+markers", name="资产负债率(%)", line={"color": "#e377c2"}),
            row=2, col=2,
        )

    fig.update_layout(
        title=f"{symbol} 财务指标趋势图",
        height=720,
        showlegend=True,
        legend={"orientation": "h", "y": 1.02},
    )

    path = _save_plotly_html(fig, symbol=symbol, chart_name="financial_trend")
    return {"chart_type": "financial_trend", "path": path}


# ---------- 工具 9：综合财报分析 ----------
def analyze_financial_report(symbol: str, year: int = None, with_plots: bool = False) -> dict:
    """
    综合财报分析：整合资产负债表、利润表、现金流量表，给出综合评估。
    :param symbol: 股票代码
    :param year: 报告年份，默认最近一年
    :param with_plots: 是否生成图表
    :return: 综合财报分析结果
    """
    print(f"综合财报分析... {symbol}")

    if year is None:
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month < 5:
            year = current_year - 2
        else:
            year = current_year - 1

    result = {
        "symbol": symbol,
        "year": year,
        "balance_sheet": {},
        "income_statement": {},
        "cash_flow": {},
        "profitability": {},
        "solvency": {},
        "growth": {},
        "dupont": {},
        "overall_evaluation": {},
    }

    # 获取各部分数据
    result["balance_sheet"] = get_balance_sheet(symbol, year, 4)
    result["income_statement"] = get_income_statement(symbol, year, 4)
    result["cash_flow"] = get_cash_flow_statement(symbol, year, 4)
    result["profitability"] = get_profitability_indicators(symbol, year, 4)
    result["solvency"] = get_solvency_indicators(symbol, year, 4)
    result["growth"] = get_growth_indicators(symbol, year, 4)
    result["dupont"] = get_dupont_analysis(symbol, year, 4)

    # 综合评估
    evaluations = []

    # 盈利能力评估
    if "roe_evaluation" in result["profitability"]:
        evaluations.append(("盈利能力", result["profitability"]["roe_evaluation"]))

    # 成长性评估
    if "evaluation" in result["growth"]:
        for key, val in result["growth"]["evaluation"].items():
            evaluations.append((f"成长性-{key}", val))

    # 偿债能力评估
    if "evaluation" in result["solvency"]:
        for key, val in result["solvency"]["evaluation"].items():
            evaluations.append((f"偿债能力-{key}", val))

    # 现金流评估
    if "analysis" in result["cash_flow"]:
        for key, val in result["cash_flow"]["analysis"].items():
            evaluations.append((f"现金流-{key}", val))

    # 杜邦分析驱动因素
    if "analysis" in result["dupont"] and "drivers" in result["dupont"]["analysis"]:
        for driver in result["dupont"]["analysis"]["drivers"]:
            evaluations.append(("杜邦分析", driver))

    result["overall_evaluation"]["items"] = evaluations

    # 综合评分（简化版）
    score = 0
    max_score = 100

    # ROE 评分 (25分) - indicators 中的值已经是百分比形式
    roe = None
    if "indicators" in result["profitability"]:
        for ind in result["profitability"]["indicators"]:
            if ind["key"] == "roe":
                roe = ind["value"]
                break
    if roe is not None:
        if roe >= 20:
            score += 25
        elif roe >= 15:
            score += 20
        elif roe >= 10:
            score += 15
        elif roe >= 5:
            score += 10
        else:
            score += 5

    # 成长性评分 (25分) - indicators 中的值已经是百分比形式
    profit_growth = None
    if "indicators" in result["growth"]:
        for ind in result["growth"]["indicators"]:
            if ind["key"] == "yoy_net_profit":
                profit_growth = ind["value"]
                break
    if profit_growth is not None:
        if profit_growth >= 30:
            score += 25
        elif profit_growth >= 20:
            score += 20
        elif profit_growth >= 10:
            score += 15
        elif profit_growth >= 0:
            score += 10
        else:
            score += 5

    # 偿债能力评分 (25分) - liabilityToAsset 是小数形式
    debt_ratio = None
    if "indicators" in result["solvency"]:
        for ind in result["solvency"]["indicators"]:
            if ind["key"] == "asset_liability_ratio":
                debt_ratio = ind["value"]
                break
    if debt_ratio is not None:
        debt_pct = debt_ratio * 100 if debt_ratio < 1 else debt_ratio
        if debt_pct <= 40:
            score += 25
        elif debt_pct <= 50:
            score += 20
        elif debt_pct <= 60:
            score += 15
        elif debt_pct <= 70:
            score += 10
        else:
            score += 5

    # 现金流评分 (25分) - 基于 CFOToNP
    cfo_to_np = None
    if "metrics" in result["cash_flow"]:
        for m in result["cash_flow"]["metrics"]:
            if m["key"] == "cfo_to_np":
                cfo_to_np = m["value"]
                break
    if cfo_to_np is not None:
        if cfo_to_np > 1:
            score += 25  # 经营现金流大于净利润，盈利质量高
        elif cfo_to_np > 0.5:
            score += 20
        elif cfo_to_np > 0:
            score += 15
        else:
            score += 5

    result["overall_evaluation"]["score"] = min(score, max_score)

    if score >= 80:
        result["overall_evaluation"]["rating"] = "优秀"
        result["overall_evaluation"]["summary"] = "财务状况优秀，盈利能力强、成长性好、财务健康"
    elif score >= 60:
        result["overall_evaluation"]["rating"] = "良好"
        result["overall_evaluation"]["summary"] = "财务状况良好，大部分指标表现正常"
    elif score >= 40:
        result["overall_evaluation"]["rating"] = "一般"
        result["overall_evaluation"]["summary"] = "财务状况一般，部分指标需关注"
    else:
        result["overall_evaluation"]["rating"] = "较弱"
        result["overall_evaluation"]["summary"] = "财务状况较弱，建议谨慎投资"

    if with_plots:
        history_result = get_financial_report_history(symbol, years=3, with_plots=True)
        if "plot" in history_result:
            result["plot"] = history_result["plot"]

    return result


if __name__ == "__main__":
    # 测试
    print("测试财报分析工具...")

    # 测试资产负债表
    print("\n=== 资产负债表 ===")
    result = get_balance_sheet("600000")
    print(result)

    # 测试利润表
    print("\n=== 利润表 ===")
    result = get_income_statement("600000")
    print(result)

    # 测试现金流量表
    print("\n=== 现金流量表 ===")
    result = get_cash_flow_statement("600000")
    print(result)

    # 测试综合分析
    print("\n=== 综合财报分析 ===")
    result = analyze_financial_report("600000")
    print(result.get("overall_evaluation"))
