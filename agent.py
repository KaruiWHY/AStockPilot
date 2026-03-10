# -*- coding: utf-8 -*-
"""
股票分析 Agent：支持行情查询、历史 K 线、股票搜索、技术指标与量化分析，
可基于数据给出投资建议。仅供参考，不构成任何投资决策依据。
"""
import os
import json
import base64
import mimetypes

import openai

from tools import stock_tools


class config:
    def __init__(
        self,
        token=None,
        endpoint="https://models.github.ai/inference",
        model="openai/gpt-5",
        max_context_tokens=6000,
        max_recent_turns=8,
    ):
        self.token = token if token is not None else os.getenv("GITHUB_TOKEN")
        self.endpoint = endpoint
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_recent_turns = max_recent_turns


class MyAgent:
    """
    A simple agent that uses the OpenAI API to generate responses.
    """
    def __init__(self, config):
        self.api_key = config.token
        if not self.api_key:
            raise ValueError("Please set the GITHUB_TOKEN environment variable.")
        self.endpoint = config.endpoint
        self.model = config.model
        self.max_context_tokens = config.max_context_tokens
        self.max_recent_turns = config.max_recent_turns

        # openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.endpoint)

        self.system_messages = [
            {
                "role": "system",
                "content": (
                    "今天是{today_date}".format(today_date=self.get_today_date()) +
                    "。你是一名专业的股票量化交易与投资者，专注于 A 股市场，投资倾向为激进型，维持长期收益的条件下，寻求适合的策略。"
                    "你可以通过工具获取：实时行情、历史 K 线、股票搜索、基本信息、技术指标（MA、MACD、RSI、布林带、波动率等）。"
                    "当用户希望验证策略可行性时，可调用回测工具评估收益、回撤、夏普和胜率。"
                    "当用户希望进行选股时，可调用因子选股工具筛选候选股票并给出打分依据。"
                    "当用户希望得到可执行候选（含仓位与风控）时，可调用交易员选股决策工具。"
                    "请根据用户问题主动调用相应工具，结合量化数据给出分析结论，并可以给出具体的投资建议（如买入/卖出/观望、建议仓位、关注价位等）。"
                    "投资建议需基于技术指标与数据支撑，并简要说明理由。"
                    "分析具体个股时，必须同时调用 get_market_overview 获取大盘（上证、深证、创业板）行情，"
                    "结合大盘走势与个股表现给出分析，例如大盘强势/弱势时对个股的影响、个股与大盘的强弱对比等。"
                    "股票代码可输入 6 位数字（如 600000、000001），工具会自动区分沪/深。"
                ),
            }
        ]

        self.conversation_history = []
        self.summary = ""

        self.tools = self._define_stock_tools()
        self.tool_registry = {
            "get_stock_quote": stock_tools.get_stock_quote,
            "get_stock_history": stock_tools.get_stock_history,
            "search_stock": stock_tools.search_stock,
            "get_stock_basic": stock_tools.get_stock_basic,
            "get_technical_analysis": stock_tools.get_technical_analysis,
            "get_market_overview": stock_tools.get_market_overview,
            "backtest_moving_average_strategy": stock_tools.backtest_moving_average_strategy,
            "backtest_ma_grid_search": stock_tools.backtest_ma_grid_search,
            "screen_stocks_by_factors": stock_tools.screen_stocks_by_factors,
            "select_stocks_for_trader": stock_tools.select_stocks_for_trader,
        }
    
    def get_today_date(self):
        """获取当前日期与星期。"""
        from datetime import date

        today = date.today()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return f"{today.strftime('%Y-%m-%d')} {weekdays[today.weekday()]}"

    def _define_stock_tools(self):
        """定义股票分析相关工具的 schema，供 LLM 与 API 使用。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_quote",
                    "description": "获取单只股票最新行情（最近一个交易日的开盘、最高、最低、收盘、成交量、涨跌幅等）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "股票代码，如 600000、000001、sh.600000。",
                            },
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_history",
                    "description": "获取股票历史 K 线数据，用于分析走势。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码，如 600000、000001。"},
                            "start_date": {"type": "string", "description": "开始日期，格式 yyyy-mm-dd。"},
                            "end_date": {"type": "string", "description": "结束日期，格式 yyyy-mm-dd。"},
                            "frequency": {
                                "type": "string",
                                "enum": ["d", "w", "m"],
                                "description": "K 线周期：d=日线，w=周线，m=月线。",
                                "default": "d",
                            },
                        },
                        "required": ["symbol", "start_date", "end_date"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_stock",
                    "description": "按股票名称或代码模糊搜索，得到匹配的股票列表（code、code_name）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string", "description": "名称或代码关键词，如 浦发、6000、茅台。"},
                        },
                        "required": ["keyword"],
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
                            "symbol": {"type": "string", "description": "股票代码，如 600000、000001。"},
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
                    "description": "获取大盘主要股指（上证指数、深证成指、创业板指）最新行情及近 5 日涨跌幅。分析具体个股时，必须同时调用此工具，结合大盘环境给出分析。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_technical_analysis",
                    "description": "基于历史 K 线计算技术指标与量化指标，用于量化交易分析与投资建议。返回 MA5/10/20/60、MACD、RSI、布林带、年化波动率、近 5/20 日涨跌幅等。支持 with_plots=true 输出技术分析图表 HTML 路径。进行买卖建议前应优先调用此工具获取量化依据。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码，如 600000、000001。"},
                            "start_date": {"type": "string", "description": "开始日期，格式 yyyy-mm-dd。建议至少覆盖 60 个交易日。"},
                            "end_date": {"type": "string", "description": "结束日期，格式 yyyy-mm-dd。"},
                            "with_plots": {"type": "boolean", "description": "是否输出技术分析曲线图（HTML 文件路径）。默认 false。", "default": False},
                        },
                        "required": ["symbol", "start_date", "end_date"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "backtest_moving_average_strategy",
                    "description": "对单只股票执行双均线策略回测。采用收盘信号、次日开盘成交，包含手续费、滑点、印花税，考虑 A 股 100 股一手与 T+1 约束。返回收益、回撤、夏普、胜率、交易明细与净值曲线。支持 with_plots=true 输出回测曲线图（HTML 文件路径）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码，如 600000、000001。"},
                            "start_date": {"type": "string", "description": "开始日期，格式 yyyy-mm-dd。"},
                            "end_date": {"type": "string", "description": "结束日期，格式 yyyy-mm-dd。"},
                            "fast_period": {"type": "integer", "description": "快线周期，默认 5。", "default": 5},
                            "slow_period": {"type": "integer", "description": "慢线周期，默认 20。", "default": 20},
                            "initial_capital": {"type": "number", "description": "初始资金，默认 100000。", "default": 100000},
                            "fee_rate": {"type": "number", "description": "双边手续费率，默认 0.0003。", "default": 0.0003},
                            "slippage_rate": {"type": "number", "description": "双边滑点率，默认 0.0002。", "default": 0.0002},
                            "tax_rate": {"type": "number", "description": "卖出印花税率，默认 0.001。", "default": 0.001},
                            "with_plots": {"type": "boolean", "description": "是否输出回测曲线图（HTML 文件路径）。默认 false。", "default": False},
                        },
                        "required": ["symbol", "start_date", "end_date"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "backtest_ma_grid_search",
                    "description": "对双均线策略执行参数网格搜索，一次性评估多组快慢线组合，返回最优参数与 Top 结果，便于做策略寻优与稳健性比较。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码，如 600000、000001。"},
                            "start_date": {"type": "string", "description": "开始日期，格式 yyyy-mm-dd。"},
                            "end_date": {"type": "string", "description": "结束日期，格式 yyyy-mm-dd。"},
                            "fast_candidates": {"type": "string", "description": "快线候选，逗号分隔整数，如 5,8,10,12。", "default": "5,8,10,12"},
                            "slow_candidates": {"type": "string", "description": "慢线候选，逗号分隔整数，如 20,30,40,60。", "default": "20,30,40,60"},
                            "initial_capital": {"type": "number", "description": "初始资金，默认 100000。", "default": 100000},
                            "fee_rate": {"type": "number", "description": "双边手续费率，默认 0.0003。", "default": 0.0003},
                            "slippage_rate": {"type": "number", "description": "双边滑点率，默认 0.0002。", "default": 0.0002},
                            "tax_rate": {"type": "number", "description": "卖出印花税率，默认 0.001。", "default": 0.001},
                            "top_n": {"type": "integer", "description": "返回前 N 名组合，默认 5。", "default": 5},
                        },
                        "required": ["symbol", "start_date", "end_date"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "screen_stocks_by_factors",
                    "description": "因子选股工具。可对指定股票池或全市场候选进行评分，综合动量、趋势、波动、回撤与流动性，返回 TopN 候选及打分明细。支持 with_plots=true 输出横向对比图（HTML 文件路径）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "description": "开始日期，格式 yyyy-mm-dd。建议覆盖至少 80 个交易日。"},
                            "end_date": {"type": "string", "description": "结束日期，格式 yyyy-mm-dd。"},
                            "universe_codes": {"type": "string", "description": "候选股票池，逗号分隔代码，可为空。示例：600000,000001,300750", "default": ""},
                            "universe_limit": {"type": "integer", "description": "未指定股票池时，从全市场截取前 N 只候选。默认 80。", "default": 80},
                            "top_n": {"type": "integer", "description": "返回前 N 只候选。默认 10。", "default": 10},
                            "min_avg_amount": {"type": "number", "description": "近 20 日平均成交额下限（元），低于该值将过滤。默认 0。", "default": 0},
                            "with_plots": {"type": "boolean", "description": "是否输出横向对比图（HTML 文件路径）。默认 false。", "default": False},
                        },
                        "required": ["start_date", "end_date"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "select_stocks_for_trader",
                    "description": "交易员级选股决策工具：先做因子筛选，再对候选进行回测验证，结合市场状态输出 Top 候选及建议仓位、止损止盈和入选理由。支持 with_plots=true 输出横向对比图与风控参数图（HTML 文件路径）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "description": "开始日期，格式 yyyy-mm-dd。"},
                            "end_date": {"type": "string", "description": "结束日期，格式 yyyy-mm-dd。"},
                            "universe_codes": {"type": "string", "description": "候选股票池，逗号分隔。为空则自动构建。", "default": ""},
                            "universe_limit": {"type": "integer", "description": "自动构建股票池时的规模上限。默认 120。", "default": 120},
                            "candidate_n": {"type": "integer", "description": "初筛候选数量。默认 20。", "default": 20},
                            "pick_n": {"type": "integer", "description": "最终输出数量。默认 5。", "default": 5},
                            "min_avg_amount": {"type": "number", "description": "近20日平均成交额门槛（元）。默认 2e8。", "default": 200000000},
                            "fast_period": {"type": "integer", "description": "验证回测快线周期。默认 5。", "default": 5},
                            "slow_period": {"type": "integer", "description": "验证回测慢线周期。默认 20。", "default": 20},
                            "initial_capital": {"type": "number", "description": "验证回测初始资金。默认 100000。", "default": 100000},
                            "with_plots": {"type": "boolean", "description": "是否输出横向对比图与风控参数图（HTML 文件路径）。默认 false。", "default": False},
                        },
                        "required": ["start_date", "end_date"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _estimate_tokens(self, messages):
        total_chars = 0
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            total_chars += len(str(role))

            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        total_chars += len(block.get("text", ""))
                    elif block_type == "image_url":
                        # Do not count raw base64 length; treat image as bounded token cost.
                        total_chars += 1200
                    else:
                        total_chars += len(str(block))
            else:
                total_chars += len(str(content))

        return total_chars // 4 + 1

    def _build_messages_with_context(self, user_input, user_content_override=None):
        messages = list(self.system_messages)
        if self.summary:
            messages.append({"role": "system", "content": f"Conversation summary: {self.summary}"})

        recent_history = self.conversation_history[-(self.max_recent_turns * 2):]
        user_content = user_input if user_content_override is None else user_content_override
        context_messages = recent_history + [{"role": "user", "content": user_content}]

        # Always keep the latest user message; only prune older history.
        while len(context_messages) > 1 and self._estimate_tokens(messages + context_messages) > self.max_context_tokens:
            context_messages.pop(0) # Remove the oldest message until we fit within the token limit

        return messages + context_messages

    def _maybe_update_summary(self):
        # TODO: Add summary trigger strategy.
        # Example triggers:
        # 1) Every N rounds, summarize early history into self.summary
        # 2) When estimated context tokens exceed a threshold, summarize and shrink history
        pass

    def _serialize_tool_calls(self, tool_calls):
        """Serialize tool calls into a format that can be sent to the API."""
        return [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]

    def _execute_tool_call(self, tool_call):
        """Execute a tool call and return the tool response."""
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

    def run(self, user_input, picture_path=None):
        # 支持流式输出的接口
        print("Streaming response by {}".format(self.model))

        user_content = self._build_user_content(user_input, picture_path)
        messages = self._build_messages_with_context(user_input, user_content_override=user_content)
        print('=' * 20)
        print(messages)
        print('=' * 20)
        try:
            with self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=True,
            ) as stream:
                collected_response = []
                for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue

                    delta = chunk.choices[0].delta
                    content = delta.content
                    tool_calls = delta.tool_calls or []

                    if content:
                        print(content, end="", flush=True)
                        collected_response.append(content)

                    for tool_call in tool_calls:
                        print(f"\n[Tool call: {tool_call['function']['name']} with arguments {tool_call['function']['arguments']}]\n", flush=True)

            print()  # Print a newline after the streaming response is complete
            return "".join(collected_response)

        except Exception as e:
            print("Error:", e)
            return "API调用出错: {}".format(str(e))

    def generate_response(self, user_input, max_tool_rounds=3, picture_path=None):
        user_content = self._build_user_content(user_input, picture_path)
        messages = self._build_messages_with_context(user_input, user_content_override=user_content)
        print('=' * 20)
        print(messages)
        print('=' * 20)
        assistant_text = ""
        # messages.append({"role": "assistant", "content": '用户您好，很高兴为您服务，根据我的分析，'})

        for i in range(max_tool_rounds):
            print("Round {}".format(i + 1))

            print('=' * 20)
            print(messages)
            print('=' * 20)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=False
                # max_tokens=1024,
                # temperature=0.7,
            )

            if not getattr(response, "choices", None):
                assistant_text = "API返回为空，请稍后重试。"
                break

            assistant_message = response.choices[0].message # 输出
            tool_calls = assistant_message.tool_calls
            # print("Thinking...")
            try:
                print("___________________________________THINKING___________________________________")
                print(assistant_message.reasoning_content)
                print("___________________________________ANSWERING___________________________________")
                print(assistant_message.content)
            # print("Tool calls:")
            # print(tool_calls)
            # print('=' * 20)
            except Exception as exc:
                print(f"Error accessing reasoning content: {exc}")
            if not tool_calls:
                assistant_text = assistant_message.content or ""
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": self._serialize_tool_calls(tool_calls),
                }
            )

            for tool_call in tool_calls:
                tool_message = self._execute_tool_call(tool_call)
                messages.append(tool_message)
                # print(f"Executed {i + 1} tool call: {tool_message}")
        else:
            assistant_text = "工具调用次数达到上限，请重试或缩小问题范围。"

        user_history_text = user_input
        if picture_path:
            user_history_text = f"{user_input}\n[image: {os.path.basename(picture_path)}]"

        self.conversation_history.append({"role": "user", "content": user_history_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_text})
        self._maybe_update_summary()

        return assistant_text
    
    def string_to_json(self, string):
        try:
            return json.loads(string)
        except json.JSONDecodeError:
            return None
        
    def _picture_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

    def _build_user_content(self, user_input, picture_path=None):
        if not picture_path:
            return user_input

        if not os.path.exists(picture_path):
            raise FileNotFoundError(f"Image not found: {picture_path}")

        if os.path.getsize(picture_path) == 0:
            raise ValueError(f"Image file is empty: {picture_path}")

        mime_type, _ = mimetypes.guess_type(picture_path)
        if mime_type is None:
            mime_type = "image/jpeg"

        image_base64 = self._picture_to_base64(picture_path)
        image_url = f"data:{mime_type};base64,{image_base64}"

        # OpenAI multimodal content format: text + image_url blocks.
        return [
            {"type": "text", "text": user_input},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    My_config = config(
        token=os.getenv("LLM_API_KEY") or os.getenv("GITHUB_TOKEN"),
        endpoint=os.getenv("LLM_ENDPOINT", "https://llmapi.paratera.com"),
        model=os.getenv("LLM_MODEL", "MiniMax-M2.5"),
        max_context_tokens=6000,
        max_recent_turns=8,
    )
    try:
        agent = MyAgent(My_config)
    except Exception as exc:
        print(f"初始化 Agent 失败: {exc}")
        raise

    print("股票分析 Agent 已启动。输入 'exit' 或 'quit' 退出。")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        picture_path = input("Image path (optional, press Enter to skip): ").strip()
        if picture_path == "":
            picture_path = None

        try:
            # response = agent.generate_response(user_input, picture_path=picture_path)
            response = agent.run(user_input, picture_path=picture_path)
            print(f"Assistant: {response}")
        except Exception as exc:
            print(f"Request failed: {exc}")