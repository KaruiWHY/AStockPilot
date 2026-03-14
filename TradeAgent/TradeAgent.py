# -*- coding: utf-8 -*-
"""
交易Agent：专注于A股市场的组合构建、交易执行与风险管理。
核心功能：组合优化、风险计算、执行规划、持仓管理。
"""
import os
import sys

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent, BaseAgentConfig
from tools import stock_tools
from tools import trade_tools


class TradeAgentConfig(BaseAgentConfig):
    """交易Agent配置，包含交易相关特有参数。"""
    def __init__(
        self,
        token=None,
        endpoint="https://models.github.ai/inference",
        model="openai/gpt-5",
        max_context_tokens=8000,
        max_recent_turns=10,
        default_capital: float = 1000000,
        max_position_pct: float = 0.3,
        max_sector_pct: float = 0.5,
        risk_free_rate: float = 0.03,
    ):
        super().__init__(
            token=token,
            endpoint=endpoint,
            model=model,
            max_context_tokens=max_context_tokens,
            max_recent_turns=max_recent_turns,
        )
        self.default_capital = default_capital
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.risk_free_rate = risk_free_rate


class TradeAgent(BaseAgent):
    """
    交易Agent：专注于A股市场的组合构建、交易执行与风险管理。

    核心职责：
    1. 组合构建：基于候选标的，进行组合优化与风险预算分配
    2. 执行规划：制定分批建仓策略、入场时机判断
    3. 持仓管理：监控持仓风险、触发风控信号、建议调仓
    """

    def __init__(self, config: TradeAgentConfig):
        # 保存特有配置
        self.default_capital = config.default_capital
        self.max_position_pct = config.max_position_pct
        self.max_sector_pct = config.max_sector_pct
        self.risk_free_rate = config.risk_free_rate
        # 调用父类初始化
        super().__init__(config)

    def _build_system_message(self) -> str:
        """构建系统消息。"""
        return (
            f"今天是{self._get_today_date()}。你是一名专业的股票交易员，专注于A股市场的组合构建、交易执行与风险管理。\n\n"
            "【核心职责】\n"
            "1. 组合构建：基于候选标的，进行组合优化与风险预算分配\n"
            "2. 执行规划：制定分批建仓策略、入场时机判断\n"
            "3. 持仓管理：监控持仓风险、触发风控信号、建议调仓\n\n"
            "【决策原则】\n"
            "- 风险优先：先算风险，再看收益\n"
            "- 纪律执行：止损止盈不犹豫\n"
            "- 顺势而为：根据市场状态调整执行节奏\n"
            "- 分散配置：控制集中度风险\n\n"
            "【可用工具】\n"
            "1. get_stock_quote: 获取单只股票最新行情\n"
            "2. get_market_overview: 获取大盘概况（判断市场状态）\n"
            "3. select_stocks_for_trader: 交易员选股（因子筛选+回测验证+仓位建议+止损止盈）\n"
            "4. optimize_portfolio: 组合优化（风险平价/均值方差）\n"
            "5. calculate_portfolio_risk: 风险计算（VaR、CVaR、最大回撤）\n"
            "6. plan_trade_execution: 执行规划（分批建仓计划）\n"
            "7. get_portfolio_snapshot: 持仓快照（市值、盈亏）\n\n"
            "【工作流程】\n"
            "1. 选股阶段：调用 select_stocks_for_trader 获取候选标的及建议仓位\n"
            "2. 组合构建：调用 optimize_portfolio 进行权重优化\n"
            "3. 风险评估：调用 calculate_portfolio_risk 计算 VaR 等风险指标\n"
            "4. 执行规划：调用 plan_trade_execution 制定分批建仓计划\n"
            "5. 持仓监控：调用 get_portfolio_snapshot 追踪盈亏\n\n"
            "【输出规范】\n"
            "每次决策必须包含：\n"
            "1. 明确建议：买入/卖出/观望，目标仓位\n"
            "2. 量化依据：风险指标、仓位权重支撑\n"
            "3. 执行方案：分批建仓计划、入场时机\n"
            "4. 风险提示：止损位、止盈位、风险等级\n\n"
            "股票代码可输入6位数字（如600000、000001），工具会自动区分沪/深。\n"
            f"默认资金：{self.default_capital:,.0f}元，单标的最大仓位：{self.max_position_pct*100:.0f}%"
        )

    def _define_tools(self) -> list:
        """定义工具 schema（精简为交易决策核心工具）。"""
        return [
            # 行情工具
            {
                "type": "function",
                "function": {
                    "name": "get_stock_quote",
                    "description": "获取单只股票最新行情（开盘、收盘、涨跌幅等）。交易决策前获取当前价格。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码，如 600000、000001"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_market_overview",
                    "description": "获取大盘主要股指（上证、深证、创业板）最新行情。判断市场状态（bull/bear/neutral）时调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            # 选股工具
            {
                "type": "function",
                "function": {
                    "name": "select_stocks_for_trader",
                    "description": "交易员选股：因子筛选+回测验证，输出Top候选及建议仓位、止损止盈。选股阶段核心工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "description": "开始日期，格式 yyyy-mm-dd"},
                            "end_date": {"type": "string", "description": "结束日期，格式 yyyy-mm-dd"},
                            "universe_codes": {"type": "string", "description": "候选股票池，逗号分隔。为空则自动构建。"},
                            "pick_n": {"type": "integer", "description": "输出数量", "default": 5},
                            "with_plots": {"type": "boolean", "description": "是否输出图表", "default": False},
                        },
                        "required": ["start_date", "end_date"],
                        "additionalProperties": False,
                    },
                },
            },
            # 交易工具
            {
                "type": "function",
                "function": {
                    "name": "optimize_portfolio",
                    "description": "组合优化：根据候选股票计算最优权重分配。支持风险平价、均值方差、等权重三种方法。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {"type": "string", "description": "股票代码，逗号分隔，如 600000,000001,300750"},
                            "total_capital": {"type": "number", "description": "总资金", "default": 1000000},
                            "method": {"type": "string", "enum": ["risk_parity", "mean_variance", "equal"], "description": "优化方法", "default": "risk_parity"},
                            "max_position_pct": {"type": "number", "description": "单标的最大仓位比例", "default": 0.3},
                            "with_plots": {"type": "boolean", "description": "是否输出图表", "default": False},
                        },
                        "required": ["symbols"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_portfolio_risk",
                    "description": "风险计算：计算组合VaR、CVaR、预期最大回撤等风险指标。风险评估阶段核心工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {"type": "string", "description": "股票代码，逗号分隔"},
                            "weights": {"type": "string", "description": "权重，逗号分隔，如 0.3,0.3,0.4。默认等权。"},
                            "total_capital": {"type": "number", "description": "总资金", "default": 1000000},
                            "confidence_level": {"type": "number", "description": "置信水平", "default": 0.95},
                            "holding_days": {"type": "integer", "description": "持仓天数", "default": 10},
                            "with_plots": {"type": "boolean", "description": "是否输出图表", "default": False},
                        },
                        "required": ["symbols"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "plan_trade_execution",
                    "description": "执行规划：制定分批建仓/减仓计划，给出止损止盈建议。执行阶段核心工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "target_position": {"type": "number", "description": "目标仓位金额"},
                            "total_capital": {"type": "number", "description": "总资金", "default": 1000000},
                            "execution_days": {"type": "integer", "description": "执行天数", "default": 5},
                            "market_state": {"type": "string", "enum": ["bull", "bear", "neutral"], "description": "市场状态", "default": "neutral"},
                        },
                        "required": ["symbol", "target_position"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_portfolio_snapshot",
                    "description": "持仓快照：计算当前市值、盈亏等。持仓监控核心工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {"type": "string", "description": "股票代码，逗号分隔"},
                            "shares": {"type": "string", "description": "持仓股数，逗号分隔"},
                            "cost_prices": {"type": "string", "description": "成本价，逗号分隔。可选。"},
                        },
                        "required": ["symbols", "shares"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _build_tool_registry(self) -> dict:
        """构建工具注册表（精简为交易决策核心工具）。"""
        return {
            # 行情工具
            "get_stock_quote": stock_tools.get_stock_quote,
            "get_market_overview": stock_tools.get_market_overview,
            # 选股工具
            "select_stocks_for_trader": stock_tools.select_stocks_for_trader,
            # 交易工具
            "optimize_portfolio": trade_tools.optimize_portfolio,
            "calculate_portfolio_risk": trade_tools.calculate_portfolio_risk,
            "plan_trade_execution": trade_tools.plan_trade_execution,
            "get_portfolio_snapshot": trade_tools.get_portfolio_snapshot,
        }


def main():
    """主函数：启动交易Agent REPL。"""
    from dotenv import load_dotenv
    load_dotenv()

    config = TradeAgentConfig(
        token=os.getenv("LLM_API_KEY") or os.getenv("GITHUB_TOKEN"),
        endpoint=os.getenv("LLM_ENDPOINT", "https://models.github.ai/inference"),
        model=os.getenv("LLM_MODEL", "GLM-5"),
        default_capital=1000000,
        max_position_pct=0.3,
    )

    try:
        agent = TradeAgent(config)
    except Exception as exc:
        print(f"初始化交易Agent失败: {exc}")
        raise

    print("=" * 60)
    print("交易Agent 已启动")
    print("核心功能：组合优化、风险计算、执行规划、持仓管理")
    print("工作流程：选股 → 组合优化 → 风险评估 → 执行规划 → 持仓监控")
    print("输入 'exit' 或 'quit' 退出，输入 'reset' 重置对话")
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

        if user_input.lower() == "reset":
            agent.reset()
            print("对话已重置。")
            continue

        try:
            response = agent.chat(user_input)
            print(f"\nAssistant: {response}")
        except Exception as exc:
            print(f"请求失败: {exc}")


if __name__ == "__main__":
    main()
