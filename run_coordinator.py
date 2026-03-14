# -*- coding: utf-8 -*-
"""
协作模式入口：启动 Coordinator REPL
支持智能路由、顺序管道、预定义工作流。
"""
import os
from dotenv import load_dotenv

from coordinator import Coordinator, CoordinatorConfig


def print_help():
    """打印帮助信息"""
    print("=" * 60)
    print("Agent 协作模式（支持上下文传递）")
    print("=" * 60)
    print("\n【协作工作流】")
    print("  /analyze <代码>    - 完整分析：技术分析→财务验证→交易规划")
    print("                       （后续Agent可获取前序分析结果）")
    print("  /quick <资金>      - 快速选股：因子选股→组合交易计划")
    print("  /check <代码>      - 财务验证：财务分析→交易计划")
    print("\n【智能路由】")
    print("  /route <问题>      - 智能路由到对应 Agent")
    print("  直接输入问题也会自动路由")
    print("\n【直接调用】")
    print("  /stock <问题>      - 直接调用 StockAgent（技术分析）")
    print("  /financial <问题>  - 直接调用 FinancialAgent（财务分析）")
    print("  /trade <问题>      - 直接调用 TradeAgent（交易规划）")
    print("\n【状态管理】")
    print("  /summary           - 查看协作状态摘要")
    print("  /context           - 查看完整共享上下文")
    print("  /log               - 查看执行日志")
    print("  /reset             - 重置所有 Agent")
    print("  /help              - 显示帮助")
    print("  exit/quit          - 退出")
    print("=" * 60)
    print("\n【协作机制说明】")
    print("- 技术分析结果会传递给财务分析Agent")
    print("- 技术和财务分析结果会传递给交易Agent")
    print("- 交易计划基于综合分析制定")
    print("=" * 60)


def main():
    load_dotenv()

    # 从环境变量读取配置
    token = os.getenv("LLM_API_KEY") or os.getenv("GITHUB_TOKEN")
    endpoint = os.getenv("LLM_ENDPOINT", "https://models.github.ai/inference")
    model = os.getenv("LLM_MODEL", "GLM-5")

    config = CoordinatorConfig(
        token=token,
        endpoint=endpoint,
        model=model,
        default_capital=1000000,
        max_position_pct=0.3,
    )

    try:
        coordinator = Coordinator(config)
    except Exception as exc:
        print(f"初始化协调器失败: {exc}")
        raise

    print_help()
    print(f"\n当前配置: model={model}, endpoint={endpoint}")
    print("输入问题自动路由，或使用 /命令 进行特定操作。\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("\n再见!")
            break

        # 命令解析
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            try:
                if cmd == "/route":
                    if not args:
                        print("用法: /route <问题>")
                        continue
                    intent, response = coordinator.route(args)
                    print(f"\n[路由到: {intent}Agent]")
                    print(f"Assistant: {response}")

                elif cmd == "/analyze":
                    symbol = args.strip() or "600000"
                    print(f"\n开始完整分析流程: {symbol}")
                    print("-" * 40)
                    results = coordinator.analyze_and_trade(symbol)
                    print("\n" + "=" * 40)
                    print("完整分析结果")
                    print("=" * 40)
                    for key, value in results.items():
                        print(f"\n【{key}】")
                        print(value)

                elif cmd == "/quick":
                    try:
                        capital = float(args) if args.strip() else None
                    except ValueError:
                        capital = None
                    print("\n开始快速选股流程...")
                    print("-" * 40)
                    results = coordinator.quick_screen(capital)
                    print("\n" + "=" * 40)
                    print("快速选股结果")
                    print("=" * 40)
                    for key, value in results.items():
                        print(f"\n【{key}】")
                        print(value)

                elif cmd == "/check":
                    symbol = args.strip() or "600000"
                    print(f"\n开始财务验证流程: {symbol}")
                    print("-" * 40)
                    results = coordinator.financial_check(symbol)
                    print("\n" + "=" * 40)
                    print("财务验证结果")
                    print("=" * 40)
                    for key, value in results.items():
                        print(f"\n【{key}】")
                        print(value)

                elif cmd == "/stock":
                    if not args:
                        print("用法: /stock <问题>")
                        continue
                    print("\n[StockAgent 技术分析]")
                    response = coordinator.call_stock(args)
                    print(f"Assistant: {response}")

                elif cmd == "/financial":
                    if not args:
                        print("用法: /financial <问题>")
                        continue
                    print("\n[FinancialAgent 财务分析]")
                    response = coordinator.call_financial(args)
                    print(f"Assistant: {response}")

                elif cmd == "/trade":
                    if not args:
                        print("用法: /trade <问题>")
                        continue
                    print("\n[TradeAgent 交易规划]")
                    response = coordinator.call_trade(args)
                    print(f"Assistant: {response}")

                elif cmd == "/context":
                    context = coordinator.get_shared_context()
                    if context:
                        print("\n共享上下文:")
                        for key, value in context.items():
                            if value is None:
                                continue
                            if key == "stock_analysis" and isinstance(value, dict):
                                print(f"  {key}:")
                                print(f"    推荐: {value.get('recommendation', 'N/A')}")
                                print(f"    原始响应: {value.get('raw_response', '')[:100]}...")
                            elif key == "financial_analysis" and isinstance(value, dict):
                                print(f"  {key}:")
                                print(f"    推荐: {value.get('recommendation', 'N/A')}")
                                print(f"    原始响应: {value.get('raw_response', '')[:100]}...")
                            elif isinstance(value, str) and len(value) > 100:
                                print(f"  {key}: {value[:100]}...")
                            else:
                                print(f"  {key}: {value}")
                    else:
                        print("\n共享上下文为空")

                elif cmd == "/summary":
                    print(coordinator.get_summary())

                elif cmd == "/log":
                    log = coordinator.get_execution_log()
                    if log:
                        print("\n执行日志:")
                        for entry in log:
                            print(f"  [{entry['timestamp']}] {entry['agent']}: {entry['action']}")
                            print(f"    输入: {entry['input'][:50]}...")
                            print(f"    输出: {entry['output'][:50]}...")
                    else:
                        print("\n执行日志为空")

                elif cmd == "/reset":
                    coordinator.reset_all()
                    print("\n所有 Agent 已重置。")

                elif cmd == "/help":
                    print_help()

                else:
                    print(f"\n未知命令: {cmd}")
                    print("输入 /help 查看可用命令")

            except Exception as exc:
                print(f"\n执行失败: {exc}")

        else:
            # 默认使用智能路由
            try:
                intent, response = coordinator.route(user_input)
                agent_names = {
                    "stock": "StockAgent (技术分析)",
                    "financial": "FinancialAgent (财务分析)",
                    "trade": "TradeAgent (交易规划)"
                }
                print(f"\n[路由到: {agent_names.get(intent, intent)}]")
                print(f"Assistant: {response}")
            except Exception as exc:
                print(f"\n请求失败: {exc}")


if __name__ == "__main__":
    main()
