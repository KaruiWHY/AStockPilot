# -*- coding: utf-8 -*-
"""
Agent 基类：提供通用的 LLM 对话、工具调用、上下文管理能力。
子类只需实现 _build_system_message、_define_tools、_build_tool_registry 三个抽象方法。
"""
import os
import json
from abc import ABC, abstractmethod
from datetime import date

import openai


class BaseAgentConfig:
    """通用 Agent 配置基类"""
    def __init__(
        self,
        token=None,
        endpoint="https://models.github.ai/inference",
        model="openai/gpt-5",
        max_context_tokens=8000,
        max_recent_turns=10,
    ):
        self.token = token or os.getenv("GITHUB_TOKEN") or os.getenv("LLM_API_KEY")
        self.endpoint = endpoint
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_recent_turns = max_recent_turns


class BaseAgent(ABC):
    """
    通用 Agent 基类。
    子类必须实现：
    - _build_system_message(): 返回系统消息字符串
    - _define_tools(): 返回工具 schema 列表
    - _build_tool_registry(): 返回工具名到函数的映射字典
    """

    def __init__(self, config: BaseAgentConfig):
        self.api_key = config.token
        if not self.api_key:
            raise ValueError("请设置 GITHUB_TOKEN 或 LLM_API_KEY 环境变量。")

        self.endpoint = config.endpoint
        self.model = config.model
        self.max_context_tokens = config.max_context_tokens
        self.max_recent_turns = config.max_recent_turns

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.endpoint)

        # 子类通过 _build_system_message 定义系统消息
        self.system_messages = [
            {"role": "system", "content": self._build_system_message()}
        ]

        self.conversation_history = []
        self.summary = ""

        # 子类通过 _define_tools 和 _build_tool_registry 定义工具
        self.tools = self._define_tools()
        self.tool_registry = self._build_tool_registry()

    # ===== 抽象方法（子类必须实现）=====

    @abstractmethod
    def _build_system_message(self) -> str:
        """构建系统消息，定义 Agent 的角色和能力。"""
        pass

    @abstractmethod
    def _define_tools(self) -> list:
        """定义工具 schema 列表，供 LLM 调用。"""
        pass

    @abstractmethod
    def _build_tool_registry(self) -> dict:
        """构建工具注册表，映射工具名到函数。"""
        pass

    # ===== 通用方法（子类直接继承）=====

    def _get_today_date(self) -> str:
        """获取当前日期与星期。"""
        today = date.today()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return f"{today.strftime('%Y-%m-%d')} {weekdays[today.weekday()]}"

    def reset(self):
        """重置对话历史和摘要。"""
        self.conversation_history = []
        self.summary = ""

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
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            total_chars += len(block.get("text", ""))
                        elif block_type == "image_url":
                            # 图片按固定 token 数估算
                            total_chars += 1200
                        else:
                            total_chars += len(str(block))
                    else:
                        total_chars += len(str(block))
            else:
                total_chars += len(str(content))

        return total_chars // 4 + 1

    def _build_messages_with_context(self, user_input: str, user_content_override=None) -> list:
        """构建带上下文的消息列表。"""
        messages = list(self.system_messages)
        if self.summary:
            messages.append({"role": "system", "content": f"对话摘要: {self.summary}"})

        recent_history = self.conversation_history[-(self.max_recent_turns * 2):]
        user_content = user_input if user_content_override is None else user_content_override
        context_messages = recent_history + [{"role": "user", "content": user_content}]

        # 保留最新用户消息，只裁剪较早的历史
        while len(context_messages) > 1 and self._estimate_tokens(messages + context_messages) > self.max_context_tokens:
            context_messages.pop(0)

        return messages + context_messages

    def _serialize_tool_calls(self, tool_calls) -> list:
        """序列化工具调用为 API 格式。"""
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
        """执行工具调用并返回工具响应消息。"""
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
        """
        生成响应，支持多轮工具调用。
        :param user_input: 用户输入
        :param max_tool_rounds: 最大工具调用轮数
        :return: 助手响应文本
        """
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
                    content = assistant_message.reasoning_content
                    # print(content[:500] + "..." if len(content) > 500 else content)
            except Exception:
                pass

            if not tool_calls:
                assistant_text = assistant_message.content or ""
                break

            # 追加助手消息
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


if __name__ == "__main__":
    # 测试基类是否能正常导入
    print("BaseAgent 基类定义完成")
    print(f"BaseAgentConfig 默认配置: endpoint=BaseAgentConfig().endpoint")
