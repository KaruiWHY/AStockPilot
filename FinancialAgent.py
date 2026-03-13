# -*- coding: utf-8 -*-
"""
财报分析 Agent：专注于 A 股上市公司财务报表分析。
支持资产负债表、利润表、现金流量表查询，财务指标计算，成长性分析与杜邦分析。
"""
import os
import json
import openai

from tools import stock_tools
from tools import financial_tools


class FinancialAgentConfig:
    """财报分析 Agent 配置。"""
    def __init__(
        self,
        token=None,
        endpoint="https://models.github.ai/inference",
        model="openai/gpt-5",
        max_context_tokens=8000,
        max_recent_turns=10,
    ):
        self.token = token if token is not None else os.getenv("GITHUB_TOKEN") or os.getenv("LLM_API_KEY")
        self.endpoint = endpoint
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_recent_turns = max_recent_turns


class FinancialAgent:
    """
    财报分析 Agent：专注于上市公司财务报表分析。
    """

    def __init__(self, config: FinancialAgentConfig):
        self.api_key = config.token
        if not self.api_key:
            raise ValueError("请设置 GITHUB_TOKEN 或 LLM_API_KEY 环境变量。")
        self.endpoint = config.endpoint
        self.model = config.model
        self.max_context_tokens = config.max_context_tokens
        self.max_recent_turns = config.max_recent_turns

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.endpoint)

        self.system_messages = [
            {
                "role": "system",
                "content": (
                    "今天是{today_date}。".format(today_date=self._get_today_date()) +
                    "你是一名专业的财务分析师，专注于 A 股上市公司财务报表分析。\n\n"
                    "你可以通过工具获取以下财务数据：\n"
                    "1. 资产负债表：资产、负债、所有者权益等核心指标\n"
                    "2. 利润表：营业收入、净利润、毛利率、净利率等盈利指标\n"
                    "3. 现金流量表：经营/投资/筹资活动现金流\n"
                    "4. 盈利能力指标：ROE、ROA、毛利率、净利率等\n"
                    "5. 偿债能力指标：流动比率、速动比率、资产负债率等\n"
                    "6. 成长能力指标：营收增长率、净利润增长率等\n"
                    "7. 杜邦分析：ROE分解为净利率×资产周转率×权益乘数\n"
                    "8. 多期财报对比：近3年财务指标趋势\n"
                    "9. 综合财报分析：整合三大报表，给出综合评估\n\n"
                    "分析原则：\n"
                    "- 分析具体公司时，先调用综合财报分析工具获取整体概况\n"
                    "- 关注ROE、营收增长、现金流等核心指标\n"
                    "- 结合杜邦分析理解ROE驱动因素\n"
                    "- 对比历史趋势判断公司发展方向\n"
                    "- 给出明确的投资建议（买入/持有/卖出/观望）并说明理由\n"
                    "- 提示潜在风险点\n\n"
                    "股票代码可输入6位数字（如600000、000001），工具会自动区分沪/深。"
                ),
            }
        ]

        self.conversation_history = []
        self.summary = ""

        self.tools = self._define_financial_tools()
        self.tool_registry = {
            # 原有股票工具
            "get_stock_quote": stock_tools.get_stock_quote,
            "get_stock_basic": stock_tools.get_stock_basic,
            "search_stock": stock_tools.search_stock,
            "get_market_overview": stock_tools.get_market_overview,
            # 财报工具
            "get_balance_sheet": financial_tools.get_balance_sheet,
            "get_income_statement": financial_tools.get_income_statement,
            "get_cash_flow_statement": financial_tools.get_cash_flow_statement,
            "get_profitability_indicators": financial_tools.get_profitability_indicators,
            "get_solvency_indicators": financial_tools.get_solvency_indicators,
            "get_growth_indicators": financial_tools.get_growth_indicators,
            "get_dupont_analysis": financial_tools.get_dupont_analysis,
            "get_financial_report_history": financial_tools.get_financial_report_history,
            "analyze_financial_report": financial_tools.analyze_financial_report,
        }

    def _get_today_date(self) -> str:
        """获取当前日期。"""
        from datetime import date
        today = date.today()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return f"{today.strftime('%Y-%m-%d')} {weekdays[today.weekday()]}"

    def reset(self):
        """重置对话历史。"""
        self.conversation_history = []
        self.summary = ""

    def _define_financial_tools(self) -> list:
        """定义财报分析工具的 schema。"""
        return [
            # 原有工具
            {
                "type": "function",
                "function": {
                    "name": "get_stock_quote",
                    "description": "获取单只股票最新行情（开盘、收盘、涨跌幅等）。分析财报前可先获取当前股价。",
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
                    "name": "get_stock_basic",
                    "description": "获取股票基本信息（名称、上市日期、类型等）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_stock",
                    "description": "按股票名称或代码模糊搜索。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string", "description": "名称或代码关键词"},
                        },
                        "required": ["keyword"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_market_overview",
                    "description": "获取大盘主要股指最新行情。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            # 财报工具
            {
                "type": "function",
                "function": {
                    "name": "get_balance_sheet",
                    "description": "获取资产负债表数据，包括资产总计、负债合计、所有者权益、流动比率等。用于分析公司财务结构和偿债能力。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码，如 600000"},
                            "year": {"type": "integer", "description": "报告年份，默认最近一年"},
                            "quarter": {"type": "integer", "description": "报告季度1-4，默认4（年报）"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_income_statement",
                    "description": "获取利润表数据，包括营业收入、净利润、毛利率、净利率等。用于分析公司盈利能力。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "year": {"type": "integer", "description": "报告年份，默认最近一年"},
                            "quarter": {"type": "integer", "description": "报告季度1-4，默认4（年报）"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_cash_flow_statement",
                    "description": "获取现金流量表数据，包括经营/投资/筹资活动现金流。用于分析公司现金流质量和造血能力。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "year": {"type": "integer", "description": "报告年份，默认最近一年"},
                            "quarter": {"type": "integer", "description": "报告季度1-4，默认4（年报）"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_profitability_indicators",
                    "description": "获取盈利能力指标，包括ROE、ROA、毛利率、净利率等。评估公司盈利能力和股东回报。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "year": {"type": "integer", "description": "报告年份，默认最近一年"},
                            "quarter": {"type": "integer", "description": "报告季度1-4，默认4（年报）"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_solvency_indicators",
                    "description": "获取偿债能力指标，包括流动比率、速动比率、资产负债率等。评估公司财务风险和偿债能力。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "year": {"type": "integer", "description": "报告年份，默认最近一年"},
                            "quarter": {"type": "integer", "description": "报告季度1-4，默认4（年报）"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_growth_indicators",
                    "description": "获取成长能力指标，包括营收增长率、净利润增长率、总资产增长率等。评估公司成长性。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "year": {"type": "integer", "description": "报告年份，默认最近一年"},
                            "quarter": {"type": "integer", "description": "报告季度1-4，默认4（年报）"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_dupont_analysis",
                    "description": "杜邦分析：将ROE分解为净利率×资产周转率×权益乘数，帮助理解ROE驱动因素。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "year": {"type": "integer", "description": "报告年份，默认最近一年"},
                            "quarter": {"type": "integer", "description": "报告季度1-4，默认4（年报）"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_financial_report_history",
                    "description": "获取多期财报数据对比（默认近3年），查看财务指标历史趋势。支持输出趋势图表。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "years": {"type": "integer", "description": "对比年数，默认3年"},
                            "with_plots": {"type": "boolean", "description": "是否生成趋势图表", "default": False},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_financial_report",
                    "description": "综合财报分析：整合资产负债表、利润表、现金流量表，计算盈利/偿债/成长能力，进行杜邦分析，给出综合评分和投资建议。分析公司财报时优先调用此工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "year": {"type": "integer", "description": "报告年份，默认最近一年"},
                            "with_plots": {"type": "boolean", "description": "是否生成图表", "default": False},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _estimate_tokens(self, messages) -> int:
        """估算消息 token 数。"""
        total_chars = 0
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            total_chars += len(str(role))
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
                    else:
                        total_chars += len(str(block))
            else:
                total_chars += len(str(content))
        return total_chars // 4 + 1

    def _build_messages_with_context(self, user_input: str):
        """构建带上下文的消息列表。"""
        messages = list(self.system_messages)
        if self.summary:
            messages.append({"role": "system", "content": f"对话摘要: {self.summary}"})

        recent_history = self.conversation_history[-(self.max_recent_turns * 2):]
        context_messages = recent_history + [{"role": "user", "content": user_input}]

        while len(context_messages) > 1 and self._estimate_tokens(messages + context_messages) > self.max_context_tokens:
            context_messages.pop(0)

        return messages + context_messages

    def _serialize_tool_calls(self, tool_calls) -> list:
        """序列化工具调用。"""
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]

    def _execute_tool_call(self, tool_call) -> dict:
        """执行工具调用。"""
        function_name = tool_call.function.name
        tool_func = self.tool_registry.get(function_name)

        if tool_func is None:
            result = {"error": f"未知工具: {function_name}"}
        else:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            try:
                raw = tool_func(**arguments)
                result = raw if isinstance(raw, dict) else {"result": raw}
            except Exception as exc:
                result = {"error": f"工具执行失败: {str(exc)}"}

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": json.dumps(result, ensure_ascii=False),
        }

    def generate_response(self, user_input: str, max_tool_rounds: int = 5) -> str:
        """生成响应，支持多轮工具调用。"""
        messages = self._build_messages_with_context(user_input)
        assistant_text = ""

        for round_idx in range(max_tool_rounds):
            print(f"\n--- Round {round_idx + 1} ---")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=False,
            )

            if not getattr(response, "choices", None):
                assistant_text = "API返回为空，请稍后重试。"
                break

            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls

            # 打印思考过程（如果有）
            try:
                print(">>> Thinking...")
                if hasattr(assistant_message, "reasoning_content") and assistant_message.reasoning_content:
                    print(assistant_message.reasoning_content[:500] + "..." if len(assistant_message.reasoning_content) > 500 else assistant_message.reasoning_content)
            except Exception:
                pass

            if not tool_calls:
                assistant_text = assistant_message.content or ""
                break

            # 追加 assistant 消息
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": self._serialize_tool_calls(tool_calls),
            })

            # 执行工具调用
            for tool_call in tool_calls:
                print(f">>> Tool: {tool_call.function.name}")
                tool_message = self._execute_tool_call(tool_call)
                messages.append(tool_message)
        else:
            assistant_text = "工具调用次数达到上限，请重试或缩小问题范围。"

        # 保存对话历史
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_text})

        return assistant_text

    def chat(self, user_input: str) -> str:
        """交互式对话入口。"""
        return self.generate_response(user_input)


def main():
    """主函数：启动财报分析 Agent REPL。"""
    from dotenv import load_dotenv
    load_dotenv()

    config = FinancialAgentConfig(
        token=os.getenv("LLM_API_KEY") or os.getenv("GITHUB_TOKEN"),
        endpoint=os.getenv("LLM_ENDPOINT", "https://models.github.ai/inference"),
        model=os.getenv("LLM_MODEL", "GLM-5"),
    )

    try:
        agent = FinancialAgent(config)
    except Exception as exc:
        print(f"初始化财报分析 Agent 失败: {exc}")
        raise

    print("=" * 60)
    print("财报分析 Agent 已启动")
    print("功能：资产负债表、利润表、现金流量表、杜邦分析、综合财报分析")
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
