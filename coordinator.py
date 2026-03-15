# -*- coding: utf-8 -*-
"""
Agent 协调器：管理 StockAgent、FinancialAgent、TradeAgent 的协作。
支持智能路由、顺序管道、预定义工作流。

协作核心：
1. 预获取数据：在流程开始时统一获取基础数据，避免重复调用
2. 共享上下文：存储跨 Agent 共享的结构化信息
3. 上下文注入：为后续 Agent 构建包含前序分析结果的提示
"""
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from base_agent import BaseAgentConfig
from agent import StockAgent, StockAgentConfig
from FinancialAgent import FinancialAgent, FinancialAgentConfig
from TradeAgent import TradeAgent, TradeAgentConfig
from tools import stock_tools, financial_tools


class CoordinatorConfig:
    """协调器配置"""
    def __init__(
        self,
        token=None,
        endpoint="https://models.github.ai/inference",
        model="openai/gpt-5",
        # 交易相关配置
        default_capital: float = 1000000,
        max_position_pct: float = 0.3,
        max_sector_pct: float = 0.5,
        risk_free_rate: float = 0.03,
    ):
        self.stock_config = StockAgentConfig(
            token=token, endpoint=endpoint, model=model
        )
        self.financial_config = FinancialAgentConfig(
            token=token, endpoint=endpoint, model=model
        )
        self.trade_config = TradeAgentConfig(
            token=token,
            endpoint=endpoint,
            model=model,
            default_capital=default_capital,
            max_position_pct=max_position_pct,
            max_sector_pct=max_sector_pct,
            risk_free_rate=risk_free_rate,
        )


class Coordinator:
    """Agent 协调器：管理多 Agent 协作

    协作机制：
    1. shared_context: 存储跨 Agent 共享的结构化信息
    2. _build_collaborative_prompt: 为后续 Agent 构建包含前序分析结果的提示
    3. 每个工作流都会将前序 Agent 的关键结论传递给后续 Agent
    """

    ROUTER_KEYWORDS = {
        "stock": ["技术", "行情", "K线", "指标", "MA", "MACD", "RSI", "选股", "回测", "均线", "趋势", "波动"],
        "financial": ["财务", "财报", "利润", "资产", "负债", "ROE", "杜邦", "现金流", "营收", "毛利率", "净利率"],
        "trade": ["交易", "仓位", "止损", "止盈", "组合", "风险", "VaR", "建仓", "执行", "持仓", "优化"],
    }

    def __init__(self, config: CoordinatorConfig):
        self.stock_agent = StockAgent(config.stock_config)
        self.financial_agent = FinancialAgent(config.financial_config)
        self.trade_agent = TradeAgent(config.trade_config)

        # 共享上下文：存储跨 Agent 的结构化信息（扩展版）
        self.shared_context: Dict[str, Any] = {
            # 基础信息
            "symbol": None,
            "capital": None,
            "timestamp": None,

            # 预获取的原始数据（避免重复调用）
            "raw_data": {
                "stock_quote": None,        # 实时行情
                "stock_basic": None,        # 基本信息
                "market_overview": None,    # 大盘概况
                "technical_indicators": None,  # 技术指标
                "balance_sheet": None,      # 资产负债表
                "income_statement": None,   # 利润表
                "cash_flow": None,          # 现金流量表
                "profitability": None,      # 盈利能力
                "solvency": None,           # 偿债能力
                "growth": None,             # 成长能力
                "dupont": None,             # 杜邦分析
            },

            # 结构化分析结果（Agent 输出）
            "stock_analysis": {
                "symbol": None,
                "current_price": None,
                "change_pct": None,
                "ma5": None, "ma10": None, "ma20": None, "ma60": None,
                "macd_dif": None, "macd_dea": None, "macd_hist": None,
                "rsi": None,
                "boll_upper": None, "boll_mid": None, "boll_lower": None,
                "support_level": None,
                "resistance_level": None,
                "trend": None,  # 上涨/下跌/震荡
                "recommendation": None,  # 买入/卖出/观望
                "score": None,
                "risk_warning": None,
                "raw_response": None,
            },
            "financial_analysis": {
                "symbol": None,
                "roe": None,
                "roa": None,
                "gross_margin": None,
                "net_margin": None,
                "revenue_growth": None,
                "profit_growth": None,
                "current_ratio": None,
                "debt_ratio": None,
                "operating_cash_flow": None,
                "recommendation": None,
                "score": None,
                "financial_risk": None,
                "raw_response": None,
            },
            "trade_plan": None,

            # 选股候选列表
            "candidates": [],

            # 已获取数据的标记
            "fetched_data": set(),
        }
        self.execution_log: List[Dict[str, str]] = []

    def _log(self, agent: str, action: str, input_text: str, output_text: str):
        """记录执行日志"""
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "input": input_text[:200],
            "output": output_text[:200],
        })

    # ===== 数据预获取方法（避免重复工具调用）=====

    def _fetch_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        预获取股票基础数据，供后续 Agent 使用，避免重复调用。
        """
        print(f"  [预获取] 获取 {symbol} 基础数据...")
        data = {}

        try:
            # 实时行情
            quote = stock_tools.get_stock_quote(symbol)
            data["stock_quote"] = quote
            self.shared_context["fetched_data"].add("stock_quote")
        except Exception as e:
            print(f"    警告: 获取行情失败 - {e}")

        try:
            # 基本信息
            basic = stock_tools.get_stock_basic(symbol)
            data["stock_basic"] = basic
            self.shared_context["fetched_data"].add("stock_basic")
        except Exception as e:
            print(f"    警告: 获取基本信息失败 - {e}")

        try:
            # 大盘概况
            market = stock_tools.get_market_overview()
            data["market_overview"] = market
            self.shared_context["fetched_data"].add("market_overview")
        except Exception as e:
            print(f"    警告: 获取大盘概况失败 - {e}")

        # 更新共享上下文
        self.shared_context["raw_data"].update(data)

        return data

    def _fetch_technical_data(self, symbol: str, days: int = 90) -> Dict[str, Any]:
        """
        预获取技术分析数据。
        """
        print(f"  [预获取] 获取 {symbol} 技术指标...")
        data = {}

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            tech = stock_tools.get_technical_analysis(symbol, start_date, end_date)
            data["technical_indicators"] = tech
            self.shared_context["fetched_data"].add("technical_indicators")
        except Exception as e:
            print(f"    警告: 获取技术指标失败 - {e}")

        self.shared_context["raw_data"].update(data)
        return data

    def _fetch_financial_data(self, symbol: str) -> Dict[str, Any]:
        """
        预获取财务数据。
        """
        print(f"  [预获取] 获取 {symbol} 财务数据...")
        data = {}

        try:
            data["balance_sheet"] = financial_tools.get_balance_sheet(symbol)
            self.shared_context["fetched_data"].add("balance_sheet")
        except Exception as e:
            print(f"    警告: 获取资产负债表失败 - {e}")

        try:
            data["income_statement"] = financial_tools.get_income_statement(symbol)
            self.shared_context["fetched_data"].add("income_statement")
        except Exception as e:
            print(f"    警告: 获取利润表失败 - {e}")

        try:
            data["cash_flow"] = financial_tools.get_cash_flow_statement(symbol)
            self.shared_context["fetched_data"].add("cash_flow")
        except Exception as e:
            print(f"    警告: 获取现金流量表失败 - {e}")

        try:
            data["profitability"] = financial_tools.get_profitability_indicators(symbol)
            self.shared_context["fetched_data"].add("profitability")
        except Exception as e:
            print(f"    警告: 获取盈利能力失败 - {e}")

        try:
            data["solvency"] = financial_tools.get_solvency_indicators(symbol)
            self.shared_context["fetched_data"].add("solvency")
        except Exception as e:
            print(f"    警告: 获取偿债能力失败 - {e}")

        try:
            data["growth"] = financial_tools.get_growth_indicators(symbol)
            self.shared_context["fetched_data"].add("growth")
        except Exception as e:
            print(f"    警告: 获取成长能力失败 - {e}")

        try:
            data["dupont"] = financial_tools.get_dupont_analysis(symbol)
            self.shared_context["fetched_data"].add("dupont")
        except Exception as e:
            print(f"    警告: 获取杜邦分析失败 - {e}")

        self.shared_context["raw_data"].update(data)
        return data

    def _build_data_summary(self) -> str:
        """
        构建已获取数据的摘要，告知 Agent 哪些数据已可用。
        """
        raw = self.shared_context.get("raw_data", {})
        fetched = self.shared_context.get("fetched_data", set())

        if not fetched:
            return ""

        parts = ["\n【已获取的数据（可直接使用，无需重复调用工具）】"]

        # 行情数据
        if "stock_quote" in fetched and raw.get("stock_quote"):
            quote = raw["stock_quote"]
            if isinstance(quote, dict) and not quote.get("error"):
                price = quote.get('close', 'N/A')
                pct = quote.get('pctChg', 'N/A')
                amount = quote.get('amount', 'N/A')
                if amount and isinstance(amount, (int, float)):
                    amount = f"{amount/1e8:.2f}亿"
                parts.append(f"- 实时行情: 价格={price}, 涨跌幅={pct}%, 成交额={amount}")

        # 技术指标（注意：返回字段是大写的 MA5, MA10 等）
        if "technical_indicators" in fetched and raw.get("technical_indicators"):
            tech = raw["technical_indicators"]
            if isinstance(tech, dict) and not tech.get("error"):
                ma5 = tech.get('MA5') or tech.get('ma5', 'N/A')
                ma10 = tech.get('MA10') or tech.get('ma10', 'N/A')
                ma20 = tech.get('MA20') or tech.get('ma20', 'N/A')
                rsi = tech.get('RSI') or tech.get('rsi', 'N/A')
                macd_dif = tech.get('MACD_DIF') or tech.get('macd_dif', 'N/A')
                macd_dea = tech.get('MACD_DEA') or tech.get('macd_dea', 'N/A')
                bb_upper = tech.get('BB_UPPER') or tech.get('boll_upper', 'N/A')
                bb_mid = tech.get('BB_MID') or tech.get('boll_mid', 'N/A')
                bb_lower = tech.get('BB_LOWER') or tech.get('boll_lower', 'N/A')

                parts.append(f"- 技术指标: MA5={ma5}, MA10={ma10}, MA20={ma20}, RSI={rsi}")
                parts.append(f"  MACD: DIF={macd_dif}, DEA={macd_dea}")
                parts.append(f"  布林带: 上轨={bb_upper}, 中轨={bb_mid}, 下轨={bb_lower}")

        # 大盘概况（注意：返回格式是 {"indices": [...]}）
        if "market_overview" in fetched and raw.get("market_overview"):
            market = raw["market_overview"]
            if isinstance(market, dict) and not market.get("error"):
                indices = market.get("indices", [])
                if indices:
                    market_info = []
                    for idx in indices:
                        name = idx.get("name", "")
                        close = idx.get("close", "N/A")
                        pct = idx.get("pctChg", "N/A")
                        market_info.append(f"{name}={close}({pct}%)")
                    parts.append(f"- 大盘概况: {', '.join(market_info)}")

        # 财务数据（注意：返回格式是 {"indicators": [...]}）
        def _get_indicator_value(data: dict, key: str) -> any:
            """从 indicators 列表中提取指定 key 的值"""
            indicators = data.get("indicators", [])
            for item in indicators:
                if item.get("key") == key:
                    return item.get("value")
            return "N/A"

        if "profitability" in fetched and raw.get("profitability"):
            prof = raw["profitability"]
            if isinstance(prof, dict) and not prof.get("error"):
                roe = _get_indicator_value(prof, "roe")
                gross = _get_indicator_value(prof, "gross_profit_margin")
                net = _get_indicator_value(prof, "net_profit_margin")
                parts.append(f"- 盈利能力: ROE={roe}%, 毛利率={gross}%, 净利率={net}%")

        if "solvency" in fetched and raw.get("solvency"):
            solv = raw["solvency"]
            if isinstance(solv, dict) and not solv.get("error"):
                current = _get_indicator_value(solv, "current_ratio")
                debt = _get_indicator_value(solv, "debt_ratio")
                parts.append(f"- 偿债能力: 流动比率={current}, 资产负债率={debt}%")

        if "growth" in fetched and raw.get("growth"):
            growth = raw["growth"]
            if isinstance(growth, dict) and not growth.get("error"):
                yoy_sales = _get_indicator_value(growth, "yoy_sales")
                yoy_profit = _get_indicator_value(growth, "yoy_profit")
                parts.append(f"- 成长能力: 营收增长={yoy_sales}%, 利润增长={yoy_profit}%")

        # 检查是否有有效数据
        if len(parts) == 1:
            return ""

        return "\n".join(parts)

    def _detect_intent(self, user_input: str) -> str:
        """检测用户意图"""
        scores = {"stock": 0, "financial": 0, "trade": 0}
        for agent_type, keywords in self.ROUTER_KEYWORDS.items():
            for kw in keywords:
                if kw in user_input:
                    scores[agent_type] += 1
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "stock"

    def _build_collaborative_prompt(self, base_prompt: str, context_keys: List[str], include_raw_data: bool = True) -> str:
        """
        构建协作提示：将前序 Agent 的分析结果注入到后续 Agent 的提示中。

        :param base_prompt: 基础提示
        :param context_keys: 需要注入的上下文键列表
        :param include_raw_data: 是否包含已获取的原始数据摘要
        :return: 包含上下文的完整提示
        """
        context_parts = []

        # 1. 注入已获取的原始数据摘要
        if include_raw_data:
            data_summary = self._build_data_summary()
            if data_summary:
                context_parts.append(data_summary)

        # 2. 注入结构化分析结果
        for key in context_keys:
            value = self.shared_context.get(key)
            if not value:
                continue

            if key == "stock_analysis" and isinstance(value, dict):
                # 技术分析结果（详细版）
                stock_info = []
                stock_info.append("\n【技术分析结果】")
                stock_info.append(f"- 股票代码: {value.get('symbol', '未知')}")

                if value.get("current_price"):
                    stock_info.append(f"- 当前价格: {value.get('current_price')} 元, 涨跌幅: {value.get('change_pct', 'N/A')}%")

                # 均线系统
                ma_parts = []
                for ma in ["ma5", "ma10", "ma20", "ma60"]:
                    if value.get(ma):
                        ma_parts.append(f"{ma.upper()}={value.get(ma)}")
                if ma_parts:
                    stock_info.append(f"- 均线系统: {', '.join(ma_parts)}")

                # MACD
                if value.get("macd_dif"):
                    stock_info.append(f"- MACD: DIF={value.get('macd_dif')}, DEA={value.get('macd_dea')}, HIST={value.get('macd_hist')}")

                # RSI
                if value.get("rsi"):
                    stock_info.append(f"- RSI: {value.get('rsi')}")

                # 布林带
                if value.get("boll_upper"):
                    stock_info.append(f"- 布林带: 上轨={value.get('boll_upper')}, 中轨={value.get('boll_mid')}, 下轨={value.get('boll_lower')}")

                # 支撑压力位
                if value.get("support_level"):
                    stock_info.append(f"- 支撑位: {value.get('support_level')}")
                if value.get("resistance_level"):
                    stock_info.append(f"- 压力位: {value.get('resistance_level')}")

                # 趋势和建议
                if value.get("trend"):
                    stock_info.append(f"- 趋势判断: {value.get('trend')}")
                if value.get("recommendation"):
                    stock_info.append(f"- 技术建议: {value.get('recommendation')}")
                if value.get("score"):
                    stock_info.append(f"- 技术评分: {value.get('score')}")
                if value.get("risk_warning"):
                    stock_info.append(f"- 风险提示: {value.get('risk_warning')}")

                context_parts.append("\n".join(stock_info))

            elif key == "financial_analysis" and isinstance(value, dict):
                # 财务分析结果（详细版）
                fin_info = []
                fin_info.append("\n【财务分析结果】")
                fin_info.append(f"- 股票代码: {value.get('symbol', '未知')}")

                # 盈利能力
                profit_parts = []
                for item in [("ROE", "roe"), ("ROA", "roa"), ("毛利率", "gross_margin"), ("净利率", "net_margin")]:
                    if value.get(item[1]):
                        profit_parts.append(f"{item[0]}={value.get(item[1])}")
                if profit_parts:
                    fin_info.append(f"- 盈利能力: {', '.join(profit_parts)}")

                # 成长能力
                if value.get("revenue_growth"):
                    fin_info.append(f"- 营收增长: {value.get('revenue_growth')}")
                if value.get("profit_growth"):
                    fin_info.append(f"- 利润增长: {value.get('profit_growth')}")

                # 偿债能力
                debt_parts = []
                if value.get("current_ratio"):
                    debt_parts.append(f"流动比率={value.get('current_ratio')}")
                if value.get("debt_ratio"):
                    debt_parts.append(f"资产负债率={value.get('debt_ratio')}")
                if debt_parts:
                    fin_info.append(f"- 偿债能力: {', '.join(debt_parts)}")

                # 现金流
                if value.get("operating_cash_flow"):
                    fin_info.append(f"- 经营现金流: {value.get('operating_cash_flow')}")

                # 建议
                if value.get("recommendation"):
                    fin_info.append(f"- 投资建议: {value.get('recommendation')}")
                if value.get("score"):
                    fin_info.append(f"- 财务评分: {value.get('score')}")
                if value.get("financial_risk"):
                    fin_info.append(f"- 财务风险: {value.get('financial_risk')}")

                context_parts.append("\n".join(fin_info))

            elif key == "candidates" and isinstance(value, list) and value:
                candidates_info = ["\n【选股候选】"]
                for i, c in enumerate(value[:5], 1):
                    candidates_info.append(f"  {i}. {c.get('symbol', '')} - {c.get('reason', '因子选股')}")
                context_parts.append("\n".join(candidates_info))

            elif isinstance(value, str) and value:
                context_parts.append(f"\n【{key}】\n{value[:800]}")

        if context_parts:
            context_block = "\n".join(context_parts)
            return f"{base_prompt}\n\n{context_block}\n\n请充分利用上述已获取的数据和分析结果，避免重复调用工具获取相同数据。"
        return base_prompt

    def _extract_structured_result(self, agent_type: str, response: str, symbol: str = None) -> Dict[str, Any]:
        """
        从 Agent 响应中提取结构化信息。
        改进版：提取更多关键字段，兼容大小写字段名。
        """
        result = {
            "raw_response": response,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
        }

        if not response:
            return result

        # 从已获取的原始数据中提取
        raw = self.shared_context.get("raw_data", {})

        if agent_type == "stock":
            # 从原始数据提取技术指标（兼容大小写字段名）
            tech = raw.get("technical_indicators", {})
            if isinstance(tech, dict) and not tech.get("error"):
                result["ma5"] = tech.get("MA5") or tech.get("ma5")
                result["ma10"] = tech.get("MA10") or tech.get("ma10")
                result["ma20"] = tech.get("MA20") or tech.get("ma20")
                result["ma60"] = tech.get("MA60") or tech.get("ma60")
                result["macd_dif"] = tech.get("MACD_DIF") or tech.get("macd_dif")
                result["macd_dea"] = tech.get("MACD_DEA") or tech.get("macd_dea")
                result["macd_hist"] = tech.get("MACD_HIST") or tech.get("macd_hist")
                result["rsi"] = tech.get("RSI") or tech.get("rsi")
                result["boll_upper"] = tech.get("BB_UPPER") or tech.get("boll_upper")
                result["boll_mid"] = tech.get("BB_MID") or tech.get("boll_mid")
                result["boll_lower"] = tech.get("BB_LOWER") or tech.get("boll_lower")

            # 从行情数据提取
            quote = raw.get("stock_quote", {})
            if isinstance(quote, dict) and not quote.get("error"):
                result["current_price"] = quote.get("close")
                result["change_pct"] = quote.get("pctChg")

            # 从响应文本提取建议
            if "买入" in response or "建议买入" in response:
                result["recommendation"] = "买入"
            elif "卖出" in response or "建议卖出" in response:
                result["recommendation"] = "卖出"
            elif "观望" in response or "等待" in response:
                result["recommendation"] = "观望"
            else:
                result["recommendation"] = "观望"

            # 提取趋势判断
            if "上涨趋势" in response or "上升趋势" in response:
                result["trend"] = "上涨"
            elif "下跌趋势" in response or "下降趋势" in response:
                result["trend"] = "下跌"
            elif "震荡" in response or "横盘" in response:
                result["trend"] = "震荡"

            # 提取支撑压力位（正则匹配）
            support_match = re.search(r"支撑[位位]?[：:]\s*([\d.]+)", response)
            if support_match:
                result["support_level"] = support_match.group(1)

            resistance_match = re.search(r"压力[位位]?[：:]\s*([\d.]+)", response)
            if resistance_match:
                result["resistance_level"] = resistance_match.group(1)

            # 提取评分
            score_match = re.search(r"(?:技术)?评[分价][：:]\s*(\d+)", response)
            if score_match:
                result["score"] = score_match.group(1)

            # 提取风险提示
            risk_match = re.search(r"风险[提示]?[：:]\s*([^\n]+)", response)
            if risk_match:
                result["risk_warning"] = risk_match.group(1).strip()

        elif agent_type == "financial":
            # 从原始数据提取财务指标（indicators 列表格式）
            def _get_ind_val(data: dict, key: str) -> any:
                indicators = data.get("indicators", [])
                for item in indicators:
                    if item.get("key") == key:
                        return item.get("value")
                return None

            prof = raw.get("profitability", {})
            if isinstance(prof, dict) and not prof.get("error"):
                result["roe"] = _get_ind_val(prof, "roe")
                result["net_margin"] = _get_ind_val(prof, "net_profit_margin")
                result["gross_margin"] = _get_ind_val(prof, "gross_profit_margin")

            solv = raw.get("solvency", {})
            if isinstance(solv, dict) and not solv.get("error"):
                result["current_ratio"] = _get_ind_val(solv, "current_ratio")
                result["debt_ratio"] = _get_ind_val(solv, "debt_ratio")

            growth = raw.get("growth", {})
            if isinstance(growth, dict) and not growth.get("error"):
                result["revenue_growth"] = _get_ind_val(growth, "yoy_sales")
                result["profit_growth"] = _get_ind_val(growth, "yoy_profit")

            cf = raw.get("cash_flow", {})
            if isinstance(cf, dict) and not cf.get("error"):
                result["operating_cash_flow"] = _get_ind_val(cf, "operating_cash_flow")

            # 从响应文本提取建议
            if "推荐" in response or "建议买入" in response:
                result["recommendation"] = "推荐"
            elif "谨慎" in response or "风险" in response:
                result["recommendation"] = "谨慎"
            elif "回避" in response or "不推荐" in response:
                result["recommendation"] = "回避"
            else:
                result["recommendation"] = "中性"

            # 提取评分
            score_match = re.search(r"(?:财务)?评[分价][：:]\s*(\d+)", response)
            if score_match:
                result["score"] = score_match.group(1)

            # 提取财务风险
            risk_match = re.search(r"(?:财务)?风险[：:]\s*([^\n]+)", response)
            if risk_match:
                result["financial_risk"] = risk_match.group(1).strip()

        elif agent_type == "trade":
            result["action"] = "交易计划已生成"

        return result

    # ===== 核心方法 =====

    def route(self, user_input: str) -> tuple:
        """智能路由：根据意图选择 Agent"""
        intent = self._detect_intent(user_input)

        if intent == "stock":
            response = self.stock_agent.chat(user_input)
        elif intent == "financial":
            response = self.financial_agent.chat(user_input)
        else:
            response = self.trade_agent.chat(user_input)

        self._log(intent, "route", user_input, response)
        return intent, response

    def pipeline(self, steps: list) -> list:
        """
        顺序执行多 Agent 任务。
        :param steps: [{"agent": "stock", "input": "..."}, ...]
        :return: 最终响应列表
        """
        results = []
        for step in steps:
            agent_name = step["agent"]
            prompt = step["input"]

            if agent_name == "stock":
                response = self.stock_agent.chat(prompt)
            elif agent_name == "financial":
                response = self.financial_agent.chat(prompt)
            elif agent_name == "trade":
                response = self.trade_agent.chat(prompt)
            else:
                response = f"未知 Agent: {agent_name}"

            results.append({"agent": agent_name, "response": response})
            self._log(agent_name, "pipeline", prompt, response)

        return results

    # ===== 预定义工作流 =====

    def analyze_and_trade(self, symbol: str, capital: float = None) -> dict:
        """
        完整分析流程：预获取数据 → 技术分析 → 财务验证 → 交易规划

        优化：
        1. 预获取所有基础数据，避免各 Agent 重复调用
        2. 通过上下文传递前序分析结果
        3. 后续 Agent 直接使用已获取数据
        """
        if capital:
            self.trade_agent.default_capital = capital

        # 初始化共享上下文
        self.shared_context["symbol"] = symbol
        self.shared_context["capital"] = capital or self.trade_agent.default_capital
        self.shared_context["timestamp"] = datetime.now().isoformat()

        results = {}

        # ===== Step 0: 预获取所有基础数据 =====
        print(f"\n[Step 0] 预获取 {symbol} 的基础数据...")
        self._fetch_stock_data(symbol)
        self._fetch_technical_data(symbol)
        self._fetch_financial_data(symbol)

        # ===== Step 1: 技术分析 =====
        print(f"\n[Step 1/3] StockAgent 正在分析 {symbol} 技术面...")
        stock_prompt = f"""分析 {symbol} 的技术面。

【重要提示】
以下数据已预先获取，请直接使用，不要重复调用工具：
- 行情数据和技术指标已获取
- 大盘概况已获取

请基于已有数据进行分析，提供：
1. 当前价格和技术指标解读（MA、MACD、RSI、布林带等）
2. 支撑位和压力位判断
3. 趋势判断（上涨/下跌/震荡）
4. 明确的买入/卖出/观望建议
5. 技术面风险提示

请以结构化方式输出，便于后续财务验证和交易规划参考。"""

        stock_prompt = self._build_collaborative_prompt(
            stock_prompt,
            context_keys=[],
            include_raw_data=True
        )
        stock_response = self.stock_agent.chat(stock_prompt)
        results["stock_analysis"] = stock_response
        self.shared_context["stock_analysis"] = self._extract_structured_result("stock", stock_response, symbol)
        self._log("stock", "analyze_and_trade", stock_prompt, stock_response)

        # ===== Step 2: 财务验证 =====
        print(f"[Step 2/3] FinancialAgent 正在验证 {symbol} 财务状况...")

        financial_base = f"""分析 {symbol} 的财务状况，验证投资价值。

【关键说明】
以下是已获取的财务数据，请**直接引用分析**，**禁止再次调用工具获取相同数据**：
- 资产负债表、利润表、现金流量表已完整获取
- 盈利能力指标（ROE、毛利率、净利率）已获取
- 偿债能力指标（流动比率、资产负债率）已获取
- 成长能力指标（营收增长、利润增长）已获取
- 杜邦分析数据已获取

请结合技术分析结果，提供：
1. 核心财务指标解读（引用上述数据）
2. 财务健康度评估
3. 与技术面分析的综合判断
4. 明确的投资建议（推荐/谨慎/回避）

输出要求：直接使用已提供数据进行分析，不要调用任何财报查询工具。"""

        financial_prompt = self._build_collaborative_prompt(
            financial_base,
            context_keys=["stock_analysis"],
            include_raw_data=True
        )
        financial_response = self.financial_agent.chat(financial_prompt)
        results["financial_analysis"] = financial_response
        self.shared_context["financial_analysis"] = self._extract_structured_result("financial", financial_response, symbol)
        self._log("financial", "analyze_and_trade", financial_prompt, financial_response)

        # ===== Step 3: 交易规划 =====
        print(f"[Step 3/3] TradeAgent 正在制定交易计划...")

        trade_base = f"""为 {symbol} 制定交易计划。

【重要提示】
以下数据已预先获取，请直接使用，不要重复调用工具：
- 行情数据、技术指标已获取
- 大盘概况已获取
- 财务数据已获取

请综合技术分析和财务分析结果，提供：
1. 综合投资决策（买入/观望/回避）
2. 建议仓位比例和金额
3. 止损位和止盈位（参考技术分析的支撑压力位）
4. 分批建仓/减仓计划
5. 风险提示和应对措施

总资金: {self.shared_context['capital']:,.0f} 元
请以结构化方式输出可执行的交易计划。"""

        trade_prompt = self._build_collaborative_prompt(
            trade_base,
            context_keys=["stock_analysis", "financial_analysis"],
            include_raw_data=True
        )
        trade_response = self.trade_agent.chat(trade_prompt)
        results["trade_plan"] = trade_response
        self.shared_context["trade_plan"] = trade_response
        self._log("trade", "analyze_and_trade", trade_prompt, trade_response)

        print(f"\n[完成] {symbol} 完整分析流程结束")
        return results

    def quick_screen(self, capital: float = None) -> dict:
        """
        快速选股流程：技术选股 → 交易规划（带上下文传递）

        协作流程：
        1. StockAgent: 因子选股，输出候选列表
        2. TradeAgent: 基于候选列表制定交易计划
        """
        if capital:
            self.trade_agent.default_capital = capital

        self.shared_context["capital"] = capital or self.trade_agent.default_capital
        self.shared_context["timestamp"] = datetime.now().isoformat()

        results = {}

        # ===== Step 1: 技术选股 =====
        print("\n[Step 1/2] StockAgent 正在进行因子选股...")
        stock_prompt = """进行因子选股，请提供：
1. Top 5 候选股票代码和名称
2. 每只股票的入选理由（技术指标支撑）
3. 动量、趋势、波动率等因子评分
4. 流动性评估

请以结构化方式输出，便于后续制定交易计划。"""
        stock_response = self.stock_agent.chat(stock_prompt)
        results["stock_selection"] = stock_response

        # 提取候选列表（简化实现）
        candidates = []
        for word in stock_response.split():
            if word.isdigit() and len(word) == 6:
                candidates.append({"symbol": word, "reason": "因子选股"})
        self.shared_context["candidates"] = candidates
        self._log("stock", "quick_screen", stock_prompt, stock_response)

        # ===== Step 2: 交易规划 =====
        print("[Step 2/2] TradeAgent 正在制定交易计划...")
        trade_base = f"""基于选股结果制定交易计划。请提供：
1. 每只候选股票的建议仓位
2. 组合风险预算分配
3. 整体止损策略
4. 分批执行建议

总资金: {self.shared_context['capital']:,.0f} 元
请以结构化方式输出可执行的交易计划。"""

        trade_prompt = self._build_collaborative_prompt(
            trade_base,
            context_keys=["candidates", "stock_analysis"]
        )
        trade_response = self.trade_agent.chat(trade_prompt)
        results["trade_plan"] = trade_response
        self.shared_context["trade_plan"] = trade_response
        self._log("trade", "quick_screen", trade_prompt, trade_response)

        print("\n[完成] 快速选股流程结束")
        return results

    def financial_check(self, symbol: str, capital: float = None) -> dict:
        """
        财务验证流程：预获取数据 → 财务分析 → 交易规划

        优化：预获取财务相关数据，避免重复调用。
        """
        if capital:
            self.trade_agent.default_capital = capital

        self.shared_context["symbol"] = symbol
        self.shared_context["capital"] = capital or self.trade_agent.default_capital
        self.shared_context["timestamp"] = datetime.now().isoformat()

        results = {}

        # ===== Step 0: 预获取数据 =====
        print(f"\n[Step 0] 预获取 {symbol} 的财务相关数据...")
        self._fetch_stock_data(symbol)
        self._fetch_financial_data(symbol)

        # ===== Step 1: 财务分析 =====
        print(f"\n[Step 1/2] FinancialAgent 正在分析 {symbol} 财务状况...")

        financial_base = f"""全面分析 {symbol} 的财务状况。

【重要提示】
以下数据已预先获取，请直接使用，不要重复调用工具：
- 资产负债表、利润表、现金流量表已获取
- 盈利能力、偿债能力、成长能力指标已获取
- 杜邦分析数据已获取

请提供：
1. 三大报表核心数据解读（资产、营收、净利润、现金流）
2. 盈利能力指标解读（ROE、毛利率、净利率）
3. 偿债能力指标解读（流动比率、资产负债率）
4. 成长能力指标解读（营收增长、利润增长）
5. 杜邦分析结果
6. 综合财务评分和投资建议

请以结构化方式输出。"""

        financial_prompt = self._build_collaborative_prompt(
            financial_base,
            context_keys=[],
            include_raw_data=True
        )
        financial_response = self.financial_agent.chat(financial_prompt)
        results["financial_analysis"] = financial_response
        self.shared_context["financial_analysis"] = self._extract_structured_result("financial", financial_response, symbol)
        self._log("financial", "financial_check", financial_prompt, financial_response)

        # ===== Step 2: 交易规划 =====
        print(f"[Step 2/2] TradeAgent 正在制定交易计划...")

        trade_base = f"""基于财务分析结果为 {symbol} 制定交易计划。

【重要提示】
- 行情数据已获取，请直接使用
- 财务数据已获取，请直接使用

请提供：
1. 基于财务状况的投资决策
2. 建议仓位和风险预算
3. 止损止盈设置
4. 财务风险应对措施

总资金: {self.shared_context['capital']:,.0f} 元
请以结构化方式输出。"""

        trade_prompt = self._build_collaborative_prompt(
            trade_base,
            context_keys=["financial_analysis"],
            include_raw_data=True
        )
        trade_response = self.trade_agent.chat(trade_prompt)
        results["trade_plan"] = trade_response
        self.shared_context["trade_plan"] = trade_response
        self._log("trade", "financial_check", trade_prompt, trade_response)

        print(f"\n[完成] {symbol} 财务验证流程结束")
        return results

    # ===== 自定义协作流程 =====

    def custom_pipeline(self, steps: List[Dict[str, str]], symbol: str = None) -> Dict[str, Any]:
        """
        自定义协作流程：支持用户定义的 Agent 执行顺序

        :param steps: 执行步骤列表，如:
            [
                {"agent": "stock", "task": "分析技术面"},
                {"agent": "financial", "task": "验证财务"},
                {"agent": "trade", "task": "制定计划"}
            ]
        :param symbol: 股票代码（可选）
        :return: 执行结果
        """
        if symbol:
            self.shared_context["symbol"] = symbol

        results = {}
        context_keys = []  # 累积需要传递的上下文键

        for i, step in enumerate(steps):
            agent_name = step.get("agent", "stock")
            task = step.get("task", "")

            print(f"\n[Step {i+1}/{len(steps)}] {agent_name}Agent 正在执行: {task}")

            # 构建包含前序上下文的提示
            base_prompt = f"{task}"
            if symbol:
                base_prompt = f"{task}（股票: {symbol}）"

            prompt = self._build_collaborative_prompt(base_prompt, context_keys)

            # 执行对应 Agent
            if agent_name == "stock":
                response = self.stock_agent.chat(prompt)
                self.shared_context["stock_analysis"] = self._extract_structured_result("stock", response, symbol)
                context_keys.append("stock_analysis")
            elif agent_name == "financial":
                response = self.financial_agent.chat(prompt)
                self.shared_context["financial_analysis"] = self._extract_structured_result("financial", response, symbol)
                context_keys.append("financial_analysis")
            elif agent_name == "trade":
                response = self.trade_agent.chat(prompt)
                self.shared_context["trade_plan"] = response
            else:
                response = f"未知 Agent: {agent_name}"

            results[f"step_{i+1}_{agent_name}"] = response
            self._log(agent_name, "custom_pipeline", prompt, response)

        return results

    # ===== 工具方法 =====

    def get_shared_context(self) -> dict:
        """获取共享上下文"""
        return self.shared_context

    def get_summary(self) -> str:
        """
        获取当前协作状态的摘要报告
        """
        lines = ["=" * 50, "协作状态摘要", "=" * 50]

        if self.shared_context.get("symbol"):
            lines.append(f"股票代码: {self.shared_context['symbol']}")

        if self.shared_context.get("capital"):
            lines.append(f"资金规模: {self.shared_context['capital']:,.0f} 元")

        # 技术分析摘要
        sa = self.shared_context.get("stock_analysis", {})
        if isinstance(sa, dict) and sa.get("recommendation"):
            lines.append(f"\n技术分析:")
            lines.append(f"  当前价格: {sa.get('current_price', 'N/A')}")
            lines.append(f"  趋势判断: {sa.get('trend', 'N/A')}")
            lines.append(f"  技术建议: {sa.get('recommendation', 'N/A')}")

        # 财务分析摘要
        fa = self.shared_context.get("financial_analysis", {})
        if isinstance(fa, dict) and fa.get("recommendation"):
            lines.append(f"\n财务分析:")
            lines.append(f"  ROE: {fa.get('roe', 'N/A')}")
            lines.append(f"  投资建议: {fa.get('recommendation', 'N/A')}")

        # 已获取数据
        fetched = self.shared_context.get("fetched_data", set())
        if fetched:
            lines.append(f"\n已获取数据: {', '.join(fetched)}")

        if self.shared_context.get("timestamp"):
            lines.append(f"\n更新时间: {self.shared_context['timestamp']}")

        return "\n".join(lines)

    def reset_all(self):
        """重置所有 Agent 状态"""
        self.stock_agent.reset()
        self.financial_agent.reset()
        self.trade_agent.reset()
        self.shared_context = {
            "symbol": None,
            "capital": None,
            "timestamp": None,
            "raw_data": {
                "stock_quote": None,
                "stock_basic": None,
                "market_overview": None,
                "technical_indicators": None,
                "balance_sheet": None,
                "income_statement": None,
                "cash_flow": None,
                "profitability": None,
                "solvency": None,
                "growth": None,
                "dupont": None,
            },
            "stock_analysis": {
                "symbol": None,
                "current_price": None,
                "change_pct": None,
                "ma5": None, "ma10": None, "ma20": None, "ma60": None,
                "macd_dif": None, "macd_dea": None, "macd_hist": None,
                "rsi": None,
                "boll_upper": None, "boll_mid": None, "boll_lower": None,
                "support_level": None,
                "resistance_level": None,
                "trend": None,
                "recommendation": None,
                "score": None,
                "risk_warning": None,
                "raw_response": None,
            },
            "financial_analysis": {
                "symbol": None,
                "roe": None,
                "roa": None,
                "gross_margin": None,
                "net_margin": None,
                "revenue_growth": None,
                "profit_growth": None,
                "current_ratio": None,
                "debt_ratio": None,
                "operating_cash_flow": None,
                "recommendation": None,
                "score": None,
                "financial_risk": None,
                "raw_response": None,
            },
            "trade_plan": None,
            "candidates": [],
            "fetched_data": set(),
        }
        self.execution_log = []

    def get_execution_log(self) -> list:
        """获取执行日志"""
        return self.execution_log

    # ===== 直接调用方法 =====

    def call_stock(self, prompt: str) -> str:
        """直接调用 StockAgent"""
        response = self.stock_agent.chat(prompt)
        self._log("stock", "direct_call", prompt, response)
        return response

    def call_financial(self, prompt: str) -> str:
        """直接调用 FinancialAgent"""
        response = self.financial_agent.chat(prompt)
        self._log("financial", "direct_call", prompt, response)
        return response

    def call_trade(self, prompt: str) -> str:
        """直接调用 TradeAgent"""
        response = self.trade_agent.chat(prompt)
        self._log("trade", "direct_call", prompt, response)
        return response


def main():
    """主函数：启动协调器 REPL"""
    from dotenv import load_dotenv
    load_dotenv()

    config = CoordinatorConfig(
        token=os.getenv("LLM_API_KEY") or os.getenv("GITHUB_TOKEN"),
        endpoint=os.getenv("LLM_ENDPOINT", "https://models.github.ai/inference"),
        model=os.getenv("LLM_MODEL", "GLM-5"),
    )

    try:
        coordinator = Coordinator(config)
    except Exception as exc:
        print(f"初始化协调器失败: {exc}")
        raise

    print("=" * 60)
    print("Agent 协作模式已启动（支持上下文传递）")
    print("命令：")
    print("  /route <问题>      - 智能路由到对应 Agent")
    print("  /analyze <代码>    - 完整分析流程（技术→财务→交易）")
    print("  /quick <资金>      - 快速选股流程")
    print("  /check <代码>      - 财务验证流程")
    print("  /stock <问题>      - 直接调用 StockAgent")
    print("  /financial <问题>  - 直接调用 FinancialAgent")
    print("  /trade <问题>      - 直接调用 TradeAgent")
    print("  /summary           - 查看协作状态摘要")
    print("  /context           - 查看共享上下文")
    print("  /log               - 查看执行日志")
    print("  /reset             - 重置所有 Agent")
    print("  exit/quit          - 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("再见!")
            break

        # 命令解析
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd == "/route":
                intent, response = coordinator.route(args)
                print(f"\n[Routed to: {intent}]")
                print(f"Assistant: {response}")

            elif cmd == "/analyze":
                symbol = args.strip() or "600000"
                results = coordinator.analyze_and_trade(symbol)
                print("\n=== 完整分析结果 ===")
                for key, value in results.items():
                    print(f"\n[{key}]\n{value}")

            elif cmd == "/quick":
                capital = float(args) if args.strip() else None
                results = coordinator.quick_screen(capital)
                print("\n=== 快速选股结果 ===")
                for key, value in results.items():
                    print(f"\n[{key}]\n{value}")

            elif cmd == "/check":
                symbol = args.strip() or "600000"
                results = coordinator.financial_check(symbol)
                print("\n=== 财务验证结果 ===")
                for key, value in results.items():
                    print(f"\n[{key}]\n{value}")

            elif cmd == "/stock":
                response = coordinator.call_stock(args)
                print(f"\nAssistant: {response}")

            elif cmd == "/financial":
                response = coordinator.call_financial(args)
                print(f"\nAssistant: {response}")

            elif cmd == "/trade":
                response = coordinator.call_trade(args)
                print(f"\nAssistant: {response}")

            elif cmd == "/summary":
                print(coordinator.get_summary())

            elif cmd == "/context":
                context = coordinator.get_shared_context()
                print(f"\n共享上下文: {context}")

            elif cmd == "/log":
                log = coordinator.get_execution_log()
                for entry in log:
                    print(f"\n[{entry['timestamp']}] {entry['agent']}: {entry['action']}")

            elif cmd == "/reset":
                coordinator.reset_all()
                print("所有 Agent 已重置。")

            else:
                print(f"未知命令: {cmd}")

        else:
            # 默认使用智能路由
            intent, response = coordinator.route(user_input)
            print(f"\n[Agent: {intent}]")
            print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
