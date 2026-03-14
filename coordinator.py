# -*- coding: utf-8 -*-
"""
Agent 协调器：管理 StockAgent、FinancialAgent、TradeAgent 的协作。
支持智能路由、顺序管道、预定义工作流。

协作核心：通过共享上下文实现 Agent 间的信息传递，后续 Agent 可获取前序 Agent 的分析结果。
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

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

        # 共享上下文：存储跨 Agent 的结构化信息
        self.shared_context: Dict[str, Any] = {
            "symbol": None,
            "capital": None,
            "stock_analysis": None,      # StockAgent 分析结果
            "financial_analysis": None,  # FinancialAgent 分析结果
            "trade_plan": None,          # TradeAgent 交易计划
            "candidates": [],            # 选股候选列表
            "timestamp": None,
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

    def _detect_intent(self, user_input: str) -> str:
        """检测用户意图"""
        scores = {"stock": 0, "financial": 0, "trade": 0}
        for agent_type, keywords in self.ROUTER_KEYWORDS.items():
            for kw in keywords:
                if kw in user_input:
                    scores[agent_type] += 1
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "stock"

    def _build_collaborative_prompt(self, base_prompt: str, context_keys: List[str]) -> str:
        """
        构建协作提示：将前序 Agent 的分析结果注入到后续 Agent 的提示中。

        :param base_prompt: 基础提示
        :param context_keys: 需要注入的上下文键列表
        :return: 包含上下文的完整提示
        """
        context_parts = []

        for key in context_keys:
            value = self.shared_context.get(key)
            if value:
                if key == "stock_analysis" and isinstance(value, dict):
                    context_parts.append(f"""
【技术分析结果】
- 股票代码: {value.get('symbol', '未知')}
- 技术评分: {value.get('score', '未评估')}
- 技术建议: {value.get('recommendation', '无')}
- 关键指标: {value.get('indicators', '无')}
- 风险提示: {value.get('risk_warning', '无')}
""")
                elif key == "financial_analysis" and isinstance(value, dict):
                    context_parts.append(f"""
【财务分析结果】
- 股票代码: {value.get('symbol', '未知')}
- 财务评分: {value.get('score', '未评估')}
- 投资建议: {value.get('recommendation', '无')}
- 核心指标: ROE={value.get('roe', 'N/A')}, 营收增长={value.get('revenue_growth', 'N/A')}
- 财务风险: {value.get('financial_risk', '无')}
""")
                elif key == "candidates" and isinstance(value, list):
                    candidates_str = ", ".join([c.get('symbol', '') for c in value[:5]])
                    context_parts.append(f"""
【选股候选】
- 候选股票: {candidates_str}
- 选股依据: {value[0].get('reason', '因子评分') if value else '无'}
""")
                elif isinstance(value, str):
                    # 字符串类型的上下文直接附加
                    context_parts.append(f"【{key}】\n{value[:500]}")

        if context_parts:
            context_block = "\n".join(context_parts)
            return f"{base_prompt}\n\n以下是前序分析结果，请结合这些信息进行分析：\n{context_block}"
        return base_prompt

    def _extract_structured_result(self, agent_type: str, response: str, symbol: str = None) -> Dict[str, Any]:
        """
        从 Agent 响应中提取结构化信息。
        这是一个简化的实现，实际可以使用 LLM 进行更精确的提取。
        """
        result = {
            "raw_response": response,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
        }

        # 简单的关键词提取
        response_lower = response.lower() if response else ""

        if agent_type == "stock":
            if "买入" in response:
                result["recommendation"] = "买入"
            elif "卖出" in response:
                result["recommendation"] = "卖出"
            else:
                result["recommendation"] = "观望"
            result["score"] = "待评估"

        elif agent_type == "financial":
            if "买入" in response or "推荐" in response:
                result["recommendation"] = "推荐"
            elif "风险" in response or "谨慎" in response:
                result["recommendation"] = "谨慎"
            else:
                result["recommendation"] = "中性"
            result["score"] = "待评估"

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
        完整分析流程：技术分析 → 财务验证 → 交易规划（带上下文传递）

        协作流程：
        1. StockAgent: 独立进行技术分析
        2. FinancialAgent: 基于技术分析结果进行财务验证
        3. TradeAgent: 综合技术+财务分析结果制定交易计划
        """
        if capital:
            self.trade_agent.default_capital = capital

        # 初始化共享上下文
        self.shared_context["symbol"] = symbol
        self.shared_context["capital"] = capital or self.trade_agent.default_capital
        self.shared_context["timestamp"] = datetime.now().isoformat()

        results = {}

        # ===== Step 1: 技术分析（独立执行）=====
        print(f"\n[Step 1/3] StockAgent 正在分析 {symbol} 技术面...")
        stock_prompt = f"""分析 {symbol} 的技术面，请提供：
1. 当前价格和技术指标（MA、MACD、RSI等）
2. 支撑位和压力位
3. 明确的买入/卖出/观望建议
4. 技术面风险提示

请以结构化方式输出，便于后续分析参考。"""
        stock_response = self.stock_agent.chat(stock_prompt)
        results["stock_analysis"] = stock_response
        self.shared_context["stock_analysis"] = self._extract_structured_result("stock", stock_response, symbol)
        self._log("stock", "analyze_and_trade", stock_prompt, stock_response)

        # ===== Step 2: 财务验证（基于技术分析结果）=====
        print(f"[Step 2/3] FinancialAgent 正在验证 {symbol} 财务状况...")
        # 构建包含技术分析结果的协作提示
        financial_base = f"""分析 {symbol} 的财务状况，验证投资价值。请提供：
1. 核心财务指标（ROE、营收增长、净利润、现金流等）
2. 财务健康度评估
3. 与技术面分析的综合判断
4. 明确的投资建议（推荐/谨慎/回避）

请以结构化方式输出。"""

        financial_prompt = self._build_collaborative_prompt(
            financial_base,
            context_keys=["stock_analysis"]
        )
        financial_response = self.financial_agent.chat(financial_prompt)
        results["financial_analysis"] = financial_response
        self.shared_context["financial_analysis"] = self._extract_structured_result("financial", financial_response, symbol)
        self._log("financial", "analyze_and_trade", financial_prompt, financial_response)

        # ===== Step 3: 交易规划（综合技术+财务分析）=====
        print(f"[Step 3/3] TradeAgent 正在制定交易计划...")
        # 构建包含前序分析结果的协作提示
        trade_base = f"""为 {symbol} 制定交易计划。请提供：
1. 综合投资决策（买入/观望/回避）
2. 建议仓位比例和金额
3. 止损位和止盈位
4. 分批建仓/减仓计划
5. 风险提示和应对措施

总资金: {self.shared_context['capital']:,.0f} 元
请以结构化方式输出可执行的交易计划。"""

        trade_prompt = self._build_collaborative_prompt(
            trade_base,
            context_keys=["stock_analysis", "financial_analysis"]
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
        财务验证流程：财务分析 → 交易规划（带上下文传递）

        协作流程：
        1. FinancialAgent: 全面财务分析
        2. TradeAgent: 基于财务结果制定交易计划
        """
        if capital:
            self.trade_agent.default_capital = capital

        self.shared_context["symbol"] = symbol
        self.shared_context["capital"] = capital or self.trade_agent.default_capital
        self.shared_context["timestamp"] = datetime.now().isoformat()

        results = {}

        # ===== Step 1: 财务分析 =====
        print(f"\n[Step 1/2] FinancialAgent 正在分析 {symbol} 财务状况...")
        financial_prompt = f"""全面分析 {symbol} 的财务状况。请提供：
1. 三大报表核心数据（资产、营收、净利润、现金流）
2. 盈利能力指标（ROE、毛利率、净利率）
3. 偿债能力指标（流动比率、资产负债率）
4. 成长能力指标（营收增长、利润增长）
5. 杜邦分析结果
6. 综合财务评分和投资建议

请以结构化方式输出。"""
        financial_response = self.financial_agent.chat(financial_prompt)
        results["financial_analysis"] = financial_response
        self.shared_context["financial_analysis"] = self._extract_structured_result("financial", financial_response, symbol)
        self._log("financial", "financial_check", financial_prompt, financial_response)

        # ===== Step 2: 交易规划 =====
        print(f"[Step 2/2] TradeAgent 正在制定交易计划...")
        trade_base = f"""基于财务分析结果为 {symbol} 制定交易计划。请提供：
1. 基于财务状况的投资决策
2. 建议仓位和风险预算
3. 止损止盈设置
4. 财务风险应对措施

总资金: {self.shared_context['capital']:,.0f} 元
请以结构化方式输出。"""

        trade_prompt = self._build_collaborative_prompt(
            trade_base,
            context_keys=["financial_analysis"]
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

        if self.shared_context.get("stock_analysis"):
            sa = self.shared_context["stock_analysis"]
            if isinstance(sa, dict):
                lines.append(f"\n技术分析: {sa.get('recommendation', '未评估')}")

        if self.shared_context.get("financial_analysis"):
            fa = self.shared_context["financial_analysis"]
            if isinstance(fa, dict):
                lines.append(f"财务分析: {fa.get('recommendation', '未评估')}")

        if self.shared_context.get("candidates"):
            lines.append(f"\n候选股票: {len(self.shared_context['candidates'])} 只")

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
            "stock_analysis": None,
            "financial_analysis": None,
            "trade_plan": None,
            "candidates": [],
            "timestamp": None,
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
