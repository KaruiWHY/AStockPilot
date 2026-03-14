# -*- coding: utf-8 -*-
"""
股票分析 Agent：支持行情查询、历史 K 线、股票搜索、技术指标与量化分析，
可基于数据给出投资建议。仅供参考，不构成任何投资决策依据。
"""
import os
import base64
import mimetypes

from base_agent import BaseAgent, BaseAgentConfig
from tools import stock_tools
from tools import financial_tools


class StockAgentConfig(BaseAgentConfig):
    """股票分析 Agent 配置。"""
    def __init__(
        self,
        token=None,
        endpoint="https://models.github.ai/inference",
        model="openai/gpt-5",
        max_context_tokens=6000,
        max_recent_turns=8,
    ):
        super().__init__(
            token=token,
            endpoint=endpoint,
            model=model,
            max_context_tokens=max_context_tokens,
            max_recent_turns=max_recent_turns,
        )


class StockAgent(BaseAgent):
    """
    股票技术分析 Agent：专注于行情查询、技术指标、选股筛选与策略回测。
    特有功能：图片输入、流式输出。
    """

    def _build_system_message(self) -> str:
        """构建系统消息。"""
        return (
            f"今天是{self._get_today_date()}。"
            "你是一名专业的股票量化交易与投资者，专注于 A 股市场，投资倾向为激进型，维持长期收益的条件下，寻求适合的策略。"
            "你可以通过工具获取：实时行情、历史 K 线、股票搜索、基本信息、技术指标（MA、MACD、RSI、布林带、波动率等）。"
            "你也可以获取财务数据：资产负债表、利润表、现金流量表、盈利能力指标、偿债能力指标、成长能力指标、杜邦分析等。"
            "当用户希望验证策略可行性时，可调用回测工具评估收益、回撤、夏普和胜率。"
            "当用户希望进行选股时，可调用因子选股工具筛选候选股票并给出打分依据。"
            "当用户希望得到可执行候选（含仓位与风控）时，可调用交易员选股决策工具。"
            "当用户询问公司基本面或财务状况时，可调用财报分析工具（analyze_financial_report、get_balance_sheet、get_income_statement等）。"
            "请根据用户问题主动调用相应工具，结合量化数据给出分析结论，并可以给出具体的投资建议（如买入/卖出/观望、建议仓位、关注价位等）。"
            "投资建议需基于技术指标与数据支撑，并简要说明理由。"
            "分析具体个股时，必须同时调用 get_market_overview 获取大盘（上证、深证、创业板）行情，"
            "结合大盘走势与个股表现给出分析，例如大盘强势/弱势时对个股的影响、个股与大盘的强弱对比等。"
            "股票代码可输入 6 位数字（如 600000、000001），工具会自动区分沪/深。"
        )

    def _define_tools(self) -> list:
        """定义工具 schema。"""
        return [
            # 行情工具
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
            # 回测工具
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
            # 选股工具
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
            # 财报分析工具
            {
                "type": "function",
                "function": {
                    "name": "get_balance_sheet",
                    "description": "获取资产负债表数据，包括资产总计、负债合计、所有者权益、流动比率、资产负债率等。用于分析公司财务结构和偿债能力。",
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
                    "description": "获取利润表数据，包括营业收入、净利润、毛利率、净利率、每股收益等。用于分析公司盈利能力。",
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
                    "description": "获取盈利能力指标，包括ROE、ROA、毛利率、净利率、营业利润率等。评估公司盈利能力和股东回报。ROE≥15%为优秀。",
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
                    "description": "获取偿债能力指标，包括流动比率、速动比率、资产负债率等。评估公司财务风险和偿债能力。流动比率≥2、资产负债率≤50%为良好。",
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
                    "description": "获取成长能力指标，包括营收增长率、净利润增长率、总资产增长率等。评估公司成长性。增速>20%为高速成长。",
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
                    "description": "杜邦分析：将ROE分解为净利率×资产周转率×权益乘数，帮助理解ROE驱动因素。分析盈利质量、运营效率和财务杠杆。",
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
                    "description": "获取多期财报数据对比（默认近3年），查看财务指标历史趋势。支持输出趋势图表HTML文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "股票代码"},
                            "years": {"type": "integer", "description": "对比年数，默认3年", "default": 3},
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
                    "description": "综合财报分析：整合资产负债表、利润表、现金流量表，计算盈利/偿债/成长能力，进行杜邦分析，给出综合评分（0-100分）和投资建议。分析公司财报时优先调用此工具。",
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

    def _build_tool_registry(self) -> dict:
        """构建工具注册表。"""
        return {
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
            # 财报分析工具
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

    # ===== StockAgent 特有方法：图片输入、流式输出 =====

    def _picture_to_base64(self, image_path: str) -> str:
        """将图片转换为 base64 编码。"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _build_user_content(self, user_input: str, picture_path: str = None):
        """构建用户内容，支持图片输入。"""
        if not picture_path:
            return user_input

        if not os.path.exists(picture_path):
            raise FileNotFoundError(f"图片不存在: {picture_path}")

        if os.path.getsize(picture_path) == 0:
            raise ValueError(f"图片文件为空: {picture_path}")

        mime_type, _ = mimetypes.guess_type(picture_path)
        if mime_type is None:
            mime_type = "image/jpeg"

        image_base64 = self._picture_to_base64(picture_path)
        image_url = f"data:{mime_type};base64,{image_base64}"

        return [
            {"type": "text", "text": user_input},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    def generate_response(self, user_input: str, max_tool_rounds: int = 3, picture_path: str = None) -> str:
        """
        生成响应，支持多轮工具调用和图片输入。
        :param user_input: 用户输入
        :param max_tool_rounds: 最大工具调用轮数
        :param picture_path: 图片路径（可选）
        :return: 助手响应文本
        """
        user_content = self._build_user_content(user_input, picture_path)
        messages = self._build_messages_with_context(user_input, user_content_override=user_content)
        assistant_text = ""

        for i in range(max_tool_rounds):
            print(f"Round {i + 1}")

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

            try:
                print("___________________________________THINKING___________________________________")
                print(assistant_message.reasoning_content)
                print("___________________________________ANSWERING___________________________________")
                print(assistant_message.content)
            except Exception as exc:
                print(f"Error accessing reasoning content: {exc}")

            if not tool_calls:
                assistant_text = assistant_message.content or ""
                break

            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": self._serialize_tool_calls(tool_calls),
            })

            for tool_call in tool_calls:
                tool_message = self._execute_tool_call(tool_call)
                messages.append(tool_message)
        else:
            assistant_text = "工具调用次数达到上限，请重试或缩小问题范围。"

        user_history_text = user_input
        if picture_path:
            user_history_text = f"{user_input}\n[image: {os.path.basename(picture_path)}]"

        self.conversation_history.append({"role": "user", "content": user_history_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_text})

        return assistant_text

    def run(self, user_input: str, picture_path: str = None) -> str:
        """
        流式输出响应。
        :param user_input: 用户输入
        :param picture_path: 图片路径（可选）
        :return: 助手响应文本
        """
        print(f"Streaming response by {self.model}")

        user_content = self._build_user_content(user_input, picture_path)
        messages = self._build_messages_with_context(user_input, user_content_override=user_content)

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

            print()
            return "".join(collected_response)

        except Exception as e:
            print("Error:", e)
            return f"API调用出错: {str(e)}"


# 兼容旧名称
MyAgent = StockAgent
config = StockAgentConfig


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    My_config = StockAgentConfig(
        token=os.getenv("LLM_API_KEY") or os.getenv("GITHUB_TOKEN"),
        endpoint=os.getenv("LLM_ENDPOINT", "https://llmapi.paratera.com"),
        model=os.getenv("LLM_MODEL", "MiniMax-M2.5"),
    )

    try:
        agent = StockAgent(My_config)
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
            response = agent.run(user_input, picture_path=picture_path)
            print(f"Assistant: {response}")
        except Exception as exc:
            print(f"Request failed: {exc}")
