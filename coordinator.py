# -*- coding: utf-8 -*-
"""
Agent 协调器：管理 StockAgent、FinancialAgent、TradeAgent 的协作。
支持智能路由、顺序管道、预定义工作流。
"""
import os
from datetime import datetime
from typing import Optional

from base_agent import BaseAgentConfig
from agent import StockAgent, StockAgentConfig
from FinancialAgent import FinancialAgent, FinancialAgentConfig
from TradeAgent import TradeAgent, TradeAgentConfig


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
    """Agent 协调器：管理多 Agent 协作"""

    ROUTER_KEYWORDS = {
        "stock": ["技术", "行情", "K线", "指标", "MA", "MACD", "RSI", "选股", "回测", "均线", "趋势", "波动"],
        "financial": ["财务", "财报", "利润", "资产", "负债", "ROE", "杜邦", "现金流", "营收", "毛利率", "净利率"],
        "trade": ["交易", "仓位", "止损", "止盈", "组合", "风险", "VaR", "建仓", "执行", "持仓", "优化"],
    }

    def __init__(self, config: CoordinatorConfig):
        self.stock_agent = StockAgent(config.stock_config)
        self.financial_agent = FinancialAgent(config.financial_config)
        self.trade_agent = TradeAgent(config.trade_config)

        self.shared_context = {}
        self.execution_log = []

    def _log(self, agent: str, action: str, input_text: str, output_text: str):
        """记录执行日志"""
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "input": input_text[:200],  # 截断
            "output": output_text[:200],
        })

    def _detect_intent(self, user_input: str) -> str:
        """检测用户意图"""
        scores = {"stock": 0, "financial": 0, "trade": 0}
        for agent_type, keywords in self.ROUTER_KEYWORDS.items():
            for kw in keywords:
                if kw in user_input:
                    scores[agent_type] += 1
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "stock"

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
        完整分析流程：技术分析 → 财务验证 → 交易规划
        """
        if capital:
            self.trade_agent.default_capital = capital

        results = {}

        # Step 1: 技术分析
        stock_prompt = f"分析 {symbol} 的技术面，给出买入/卖出/观望建议"
        results["stock_analysis"] = self.stock_agent.chat(stock_prompt)
        self._log("stock", "analyze_and_trade", stock_prompt, results["stock_analysis"])

        # Step 2: 财务验证
        financial_prompt = f"分析 {symbol} 的财务状况，验证投资价值"
        results["financial_analysis"] = self.financial_agent.chat(financial_prompt)
        self._log("financial", "analyze_and_trade", financial_prompt, results["financial_analysis"])

        # Step 3: 交易规划
        trade_prompt = f"基于以上分析，为 {symbol} 制定交易计划，包括仓位和止损止盈"
        results["trade_plan"] = self.trade_agent.chat(trade_prompt)
        self._log("trade", "analyze_and_trade", trade_prompt, results["trade_plan"])

        # 更新共享上下文
        self.shared_context = {
            "symbol": symbol,
            "capital": capital or self.trade_agent.default_capital,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        return results

    def quick_screen(self, capital: float = None) -> dict:
        """
        快速选股流程：技术选股 → 交易规划
        """
        if capital:
            self.trade_agent.default_capital = capital

        results = {}

        # Step 1: 技术选股
        stock_prompt = "进行选股，给出今日推荐标的列表"
        results["stock_selection"] = self.stock_agent.chat(stock_prompt)
        self._log("stock", "quick_screen", stock_prompt, results["stock_selection"])

        # Step 2: 交易规划
        trade_prompt = "基于以上选股结果，制定交易计划"
        results["trade_plan"] = self.trade_agent.chat(trade_prompt)
        self._log("trade", "quick_screen", trade_prompt, results["trade_plan"])

        self.shared_context = {
            "capital": capital or self.trade_agent.default_capital,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        return results

    def financial_check(self, symbol: str, capital: float = None) -> dict:
        """
        财务验证流程：财务分析 → 交易规划
        """
        if capital:
            self.trade_agent.default_capital = capital

        results = {}

        # Step 1: 财务分析
        financial_prompt = f"全面分析 {symbol} 的财务状况，评估投资价值"
        results["financial_analysis"] = self.financial_agent.chat(financial_prompt)
        self._log("financial", "financial_check", financial_prompt, results["financial_analysis"])

        # Step 2: 交易规划
        trade_prompt = f"基于财务分析结果，为 {symbol} 制定交易计划"
        results["trade_plan"] = self.trade_agent.chat(trade_prompt)
        self._log("trade", "financial_check", trade_prompt, results["trade_plan"])

        self.shared_context = {
            "symbol": symbol,
            "capital": capital or self.trade_agent.default_capital,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        return results

    # ===== 工具方法 =====

    def get_shared_context(self) -> dict:
        """获取共享上下文"""
        return self.shared_context

    def reset_all(self):
        """重置所有 Agent 状态"""
        self.stock_agent.reset()
        self.financial_agent.reset()
        self.trade_agent.reset()
        self.shared_context = {}
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
    print("Agent 协作模式已启动")
    print("命令：")
    print("  /route <问题>      - 智能路由到对应 Agent")
    print("  /analyze <代码>    - 完整分析流程（技术→财务→交易）")
    print("  /quick <资金>      - 快速选股流程")
    print("  /check <代码>      - 财务验证流程")
    print("  /stock <问题>      - 直接调用 StockAgent")
    print("  /financial <问题>  - 直接调用 FinancialAgent")
    print("  /trade <问题>      - 直接调用 TradeAgent")
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
