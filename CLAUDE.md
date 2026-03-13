# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI agent
python run.py

# Run Streamlit dashboard
streamlit run dashboard.py
```

## Environment Setup

Create a `.env` file in the project root with:
- `LLM_API_KEY` or `GITHUB_TOKEN`: Required for LLM API access
- `LLM_ENDPOINT`: Optional, defaults to `https://llmapi.paratera.com`
- `LLM_MODEL`: Optional, can select at runtime (GLM-5, MiniMax-M2.5, Kimi-K2)

## Architecture

**Core Components:**
- `agent.py`: `MyAgent` class using OpenAI-compatible API with tool calling. Contains tool schemas, conversation history management, context token estimation, and multi-round tool execution loop.
- `run.py`: CLI entry point with model selection and REPL loop.
- `dashboard.py`: Streamlit visualization with tabs for market overview, K-line charts, technical analysis, backtesting, and stock screening.
- `tools/stock_tools.py`: All data tools using `baostock` library for A-share market data.

**Data Flow:**
1. User input → `MyAgent.generate_response()`
2. LLM decides tool calls → `_execute_tool_call()` dispatches to `stock_tools` functions
3. Tool results appended to messages → LLM generates final response
4. Conversation history maintained with token-based pruning

**Stock Code Format:**
- Input: 6-digit codes (e.g., `600000`, `000001`)
- Internal: Normalized to baostock format (`sh.600000`, `sz.000001`)
- 6-prefix → Shanghai (sh), 0/3-prefix → Shenzhen (sz)

**Tool Registry (agent.py:71-82):**
Maps tool names to `stock_tools` functions. When adding new tools, update both `_define_stock_tools()` schema and `tool_registry` dict.

**Plot Output:**
All tools with `with_plots=true` save HTML files to `outputs/plots/` directory.
