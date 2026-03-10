# -*- coding: utf-8 -*-
"""
股票分析 Agent 入口：使用 agent.MyAgent 与 config，支持多轮对话与工具调用。
从项目根目录的 .env 或系统环境变量读取：LLM_API_KEY 或 GITHUB_TOKEN，可选 LLM_ENDPOINT、LLM_MODEL。
"""
import os

from dotenv import load_dotenv

from agent import MyAgent, config

# 加载 .env 到环境变量（在读取 getenv 之前执行）
load_dotenv()

MODEL_DICT = {
        1: "GLM-5",
        2: "MiniMax-M2.5",
        3: "Kimi-K2"
    }
def main():
    model = MODEL_DICT[int(input("请选择模型（1-GLM-5，2-MiniMax-M2.5，3-Kimi-K2，默认1）: ")) or 1]

    cfg = config(
        token=os.getenv("LLM_API_KEY") or os.getenv("GITHUB_TOKEN"),
        endpoint=os.getenv("LLM_ENDPOINT", "https://llmapi.paratera.com"),
        # model=os.getenv("LLM_MODEL", "MiniMax-M2.5"),
        model=os.getenv("LLM_MODEL", model),
        max_context_tokens=10000,
        max_recent_turns=20,
    )
    


    try:
        agent = MyAgent(cfg)
    except ValueError as e:
        print("请设置环境变量 GITHUB_TOKEN 或 LLM_API_KEY 后再运行。")
        raise SystemExit(1) from e

    color_logo = "\033[1;34m"
    reset_color = "\033[0m"
    logo = """
    ██████╗ ██╗   ███╗   ███╗███████╗
    ██╔══██╗██║   ████╗ ████║██╔════╝
    ██████╔╝██║██║██╔████╔██║█████╗  
    ██╔══██╗██║══╝██║╚██╔╝██║██╔══╝  
    ██████╔╝██║   ██║ ╚═╝ ██║███████╗
    ╚═════╝ ╚═    ╚═╝     ╚═╝╚══════╝
    """
    print(color_logo + logo + reset_color)
    print("股票分析 Agent：支持行情、历史 K 线、技术指标、量化分析与投资建议。输入 'exit' 或 'quit' 退出。\n")

    while True:
        user_input = input("您: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        if not user_input:
            continue
        try:
            response = agent.generate_response(user_input, max_tool_rounds=10)
            print(f"\n助手: {response}\n")
        except Exception as exc:
            print(f"请求失败: {exc}\n")


if __name__ == "__main__":
    main()
