# AStockPilot 股票分析助手

基于 `OpenAI-compatible API`、`baostock`、`Streamlit` 的 A 股分析项目，支持：

- 多轮对话式个股分析（行情、历史 K 线、技术指标）
- 大盘概览联动分析（上证/深证/创业板）
- 双均线策略回测与参数网格搜索
- 因子选股与交易员风格候选筛选
- 财报分析与杜邦分析
- 组合优化与风险管理
- 可视化面板（`dashboard.py`）

只是一个小demo，不构成任何投资建议。

## 1. 环境要求

- Python 3.10+

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# 二选一即可
LLM_API_KEY=your_api_key
# GITHUB_TOKEN=your_github_token

# 可选
LLM_ENDPOINT=https://llmapi.paratera.com
LLM_MODEL=GLM-5
```

说明：

- `run.py` 会优先读取 `LLM_API_KEY`，若不存在则读取 `GITHUB_TOKEN`
- `LLM_ENDPOINT` 默认值为 `https://llmapi.paratera.com`
- `LLM_MODEL` 不设置时，可在启动时通过菜单选择模型

## 4. 运行方式

### 4.1 命令行 Agent

```bash
# 通用股票分析 Agent
python run.py

# 财报分析 Agent
python FinancialAgent.py

# 交易 Agent（组合优化、风险管理、执行规划）
python TradeAgent/TradeAgent.py

# Agent 协作模式（推荐）
python run_coordinator.py
```

启动后进入对话，输入 `exit` 或 `quit` 退出，输入 `reset` 重置对话。

### 4.2 Streamlit 可视化面板

```bash
streamlit run dashboard.py
```

默认会展示市场概览、K 线与策略分析相关模块。

## 5. 项目结构

```text
MyAgent/
├─ base_agent.py             # BaseAgent：通用 Agent 基类
├─ agent.py                  # StockAgent：股票技术分析 Agent（继承 BaseAgent）
├─ FinancialAgent.py         # FinancialAgent：财报分析 Agent（继承 BaseAgent）
├─ TradeAgent/
│  ├─ __init__.py
│  └─ TradeAgent.py          # TradeAgent：交易决策 Agent（继承 BaseAgent）
├─ coordinator.py            # Coordinator：Agent 协调器
├─ run.py                    # 命令行入口
├─ run_coordinator.py        # 协作模式入口
├─ dashboard.py              # Streamlit 仪表盘入口
├─ requirements.txt          # Python 依赖
├─ tools/
│  ├─ __init__.py
│  ├─ stock_tools.py         # 股票数据、技术指标、回测、选股工具
│  ├─ financial_tools.py     # 财报分析工具（资产负债表、利润表、杜邦分析等）
│  └─ trade_tools.py         # 交易工具（组合优化、风险计算、执行规划）
└─ outputs/
   ├─ plots/                 # 运行后生成的 HTML 图表
   └─ portfolios/            # 持仓数据存储
```

## 6. 多 Agent 架构

### 6.1 架构设计

所有 Agent 继承自 `BaseAgent` 基类，共享核心能力：

```
BaseAgentConfig                    # 通用配置基类
    ├── StockAgentConfig           # 股票分析配置
    ├── FinancialAgentConfig       # 财报分析配置
    └── TradeAgentConfig           # 交易配置（额外参数：default_capital, max_position_pct 等）

BaseAgent                          # 通用 Agent 基类
    ├── StockAgent                 # 股票技术分析（+ 图片输入、流式输出）
    ├── FinancialAgent             # 财报分析
    └── TradeAgent                 # 交易决策
```

### 6.2 BaseAgent 基类

```python
class BaseAgent(ABC):
    """通用 Agent 基类"""

    # 子类必须实现的抽象方法
    @abstractmethod
    def _build_system_message(self) -> str: ...

    @abstractmethod
    def _define_tools(self) -> list: ...

    @abstractmethod
    def _build_tool_registry(self) -> dict: ...

    # 通用方法（子类直接继承）
    def _estimate_tokens(self, messages) -> int: ...
    def _build_messages_with_context(self, user_input: str) -> list: ...
    def _execute_tool_call(self, tool_call) -> dict: ...
    def generate_response(self, user_input: str, max_tool_rounds: int = 5) -> str: ...
    def reset(self): ...
    def chat(self, user_input: str) -> str: ...
```

### 6.3 Agent 职责分工

| Agent | 职责 | 核心工具 | 特有功能 |
|-------|------|---------|----------|
| **StockAgent** | 股票技术分析：行情、技术指标、回测、选股 | get_stock_quote, get_technical_analysis, backtest_moving_average_strategy | 图片输入、流式输出 |
| **FinancialAgent** | 财报分析：三大报表、财务指标、杜邦分析 | analyze_financial_report, get_balance_sheet, get_dupont_analysis | - |
| **TradeAgent** | 交易决策：组合优化、风险管理、执行规划 | optimize_portfolio, calculate_portfolio_risk, plan_trade_execution | 交易配置参数 |

### 6.4 Agent 差异化定位

| 维度 | MyAgent | FinancialAgent | TradeAgent |
|------|---------|----------------|------------|
| 分析视角 | 技术面+量化 | 基本面+财务 | 组合+风控 |
| 输出类型 | 技术指标、回测结果 | 财务评分、投资建议 | 仓位权重、止损止盈 |
| 适用场景 | 个股技术分析 | 公司基本面研究 | 组合构建与交易执行 |

---

## 7. Agent 协作架构

### 7.1 协作模式

启动协作模式：

```bash
python run_coordinator.py
```

支持三种协作模式：

**模式一：智能路由（默认）**
```
用户输入 → 关键词识别 → 自动路由到对应 Agent
```

**模式二：顺序管道（Pipeline）**
```
StockAgent（选股） → FinancialAgent（财务验证） → TradeAgent（交易规划）
```

**模式三：直接调用**
```
/stock <问题>      # 直接调用 StockAgent
/financial <问题>  # 直接调用 FinancialAgent
/trade <问题>      # 直接调用 TradeAgent
```

### 7.2 可用命令

| 命令 | 说明 |
|------|------|
| `/route <问题>` | 智能路由到对应 Agent |
| `/analyze <代码>` | 完整分析流程：技术分析 → 财务验证 → 交易规划 |
| `/quick <资金>` | 快速选股流程：技术选股 → 交易规划 |
| `/check <代码>` | 财务验证流程：财务分析 → 交易规划 |
| `/stock <问题>` | 直接调用 StockAgent |
| `/financial <问题>` | 直接调用 FinancialAgent |
| `/trade <问题>` | 直接调用 TradeAgent |
| `/context` | 查看共享上下文 |
| `/log` | 查看执行日志 |
| `/reset` | 重置所有 Agent |
| `/help` | 显示帮助 |

### 7.3 典型工作流

**完整分析流程（推荐）**
```
/analyze 600000

流程：
1. StockAgent：技术分析，给出买入/卖出/观望建议
2. FinancialAgent：财务验证，评估投资价值
3. TradeAgent：交易规划，制定仓位与止损止盈
```

**快速选股**
```
/quick 1000000

流程：
1. StockAgent：因子选股，输出候选标的
2. TradeAgent：组合构建，分配仓位权重
```

**财务验证**
```
/check 600000

流程：
1. FinancialAgent：全面财务分析
2. TradeAgent：基于财务结果制定交易计划
```

### 7.4 协作架构设计

```python
class Coordinator:
    """Agent 协调器：管理多 Agent 协作"""

    ROUTER_KEYWORDS = {
        "stock": ["技术", "行情", "K线", "指标", "MA", "MACD", "RSI", "选股", "回测"],
        "financial": ["财务", "财报", "利润", "资产", "负债", "ROE", "杜邦", "现金流"],
        "trade": ["交易", "仓位", "止损", "止盈", "组合", "风险", "VaR", "建仓"],
    }

    def route(self, user_input: str) -> tuple:
        """智能路由：根据意图选择 Agent"""

    def analyze_and_trade(self, symbol: str) -> dict:
        """完整分析流程：技术分析 → 财务验证 → 交易规划"""

    def quick_screen(self, capital: float) -> dict:
        """快速选股流程"""

    def financial_check(self, symbol: str) -> dict:
        """财务验证流程"""
```

### 7.5 共享上下文

协作过程中维护共享上下文，支持跨 Agent 信息传递：

```python
shared_context = {
    "symbol": "600000",           # 当前分析的股票代码
    "capital": 1000000,           # 资金规模
    "results": {                  # 各 Agent 分析结果
        "stock_analysis": "...",
        "financial_analysis": "...",
        "trade_plan": "..."
    },
    "timestamp": "2024-03-14",    # 时间戳
}
```

---

## 8. TradeAgent 设计框架

### 7.1 核心定位

TradeAgent 专注于 **交易执行层面** 的决策支持，与现有 Agent 形成能力互补：

```
MyAgent (分析) → FinancialAgent (基本面) → TradeAgent (执行)
     ↓                  ↓                      ↓
  选股候选           财务评分              组合构建+风控
```

### 7.2 与 select_stocks_for_trader 的差异

| 维度 | select_stocks_for_trader | TradeAgent |
|------|-------------------------|------------|
| 组合构建 | 单标的仓位建议 | 多标的组合优化、相关性约束 |
| 风险管理 | 止损止盈价位 | 组合VaR、CVaR、风险预算 |
| 执行规划 | 无 | 分批建仓、入场时机判断 |
| 持仓管理 | 无 | 持仓快照、盈亏追踪 |

### 7.3 核心功能

#### 7.3.1 组合优化 (`optimize_portfolio`)

基于候选股票列表，计算最优权重分配：

- **风险平价 (Risk Parity)**：各资产风险贡献相等
- **均值方差 (Mean-Variance)**：最大化夏普比率
- **等权重 (Equal)**：均匀分配

输出：
- 各标的建议权重
- 资金分配方案
- 组合预期收益/波动率/夏普比率
- 各资产风险贡献占比

#### 7.3.2 风险计算 (`calculate_portfolio_risk`)

计算组合风险指标：

- **VaR (Value at Risk)**：给定置信水平下的最大损失
- **CVaR (Conditional VaR)**：极端情况下的平均损失
- **最大回撤**：历史最大亏损幅度
- **持仓期调整**：根据持仓天数调整风险估算

输出：
- 日度/持仓期 VaR 金额与百分比
- 历史最大回撤
- 风险预算建议（止损金额）

#### 7.3.3 执行规划 (`plan_trade_execution`)

制定分批建仓/减仓计划：

- **市场状态适配**：
  - Bull（牛市）：前重后轻，快速建仓
  - Bear（熊市）：前轻后重，等待更好价格
  - Neutral（中性）：均匀建仓
- **风控参数**：
  - 止损位：2倍年化波动率
  - 止盈位：3倍年化波动率

输出：
- 分日执行计划（股数、金额、累计进度）
- 止损止盈价位
- 执行建议

#### 7.3.4 持仓快照 (`get_portfolio_snapshot`)

获取当前持仓状态：

- 各持仓市值
- 盈亏金额与百分比
- 组合总市值

### 7.4 系统消息设计

```
今天是{today_date}。你是一名专业的股票交易员，专注于A股市场的组合构建、交易执行与风险管理。

【核心职责】
1. 组合构建：基于候选标的，进行组合优化与风险预算分配
2. 执行规划：制定分批建仓策略、入场时机判断
3. 持仓管理：监控持仓风险、触发风控信号、建议调仓
4. 交易日志：记录交易决策、追踪盈亏

【决策原则】
- 风险优先：先算风险，再看收益
- 纪律执行：止损止盈不犹豫
- 顺势而为：根据市场状态调整执行节奏
- 分散配置：控制集中度风险

【输出规范】
每次决策必须包含：明确建议、量化依据、执行方案、风险提示
```

### 7.5 配置参数

```python
TradeAgentConfig(
    default_capital=1000000,      # 默认资金：100万
    max_position_pct=0.3,         # 单标的最大仓位：30%
    max_sector_pct=0.5,           # 同行业最大仓位：50%
    risk_free_rate=0.03,          # 无风险利率：3%
)
```

---

## 8. 工具一览

### 8.1 行情工具 (`stock_tools.py`)

| 工具 | 功能 |
|------|------|
| `get_stock_quote` | 获取最新行情 |
| `get_stock_history` | 获取历史K线 |
| `get_market_overview` | 获取大盘概况 |
| `get_technical_analysis` | 技术指标分析（MA/MACD/RSI/布林带） |
| `search_stock` | 股票搜索 |
| `get_stock_basic` | 股票基本信息 |

### 8.2 选股与回测工具 (`stock_tools.py`)

| 工具 | 功能 |
|------|------|
| `screen_stocks_by_factors` | 因子选股（动量/趋势/波动/流动性） |
| `select_stocks_for_trader` | 交易员选股（含仓位与风控建议） |
| `backtest_moving_average_strategy` | 双均线策略回测 |
| `backtest_ma_grid_search` | 双均线参数网格搜索 |

### 8.3 财报工具 (`financial_tools.py`)

| 工具 | 功能 |
|------|------|
| `analyze_financial_report` | 综合财报分析（评分+建议） |
| `get_balance_sheet` | 资产负债表 |
| `get_income_statement` | 利润表 |
| `get_cash_flow_statement` | 现金流量表 |
| `get_profitability_indicators` | 盈利能力指标（ROE/毛利率/净利率） |
| `get_solvency_indicators` | 偿债能力指标（流动比率/资产负债率） |
| `get_growth_indicators` | 成长能力指标（营收/利润增长率） |
| `get_dupont_analysis` | 杜邦分析 |
| `get_financial_report_history` | 多期财报对比 |

### 8.4 交易工具 (`trade_tools.py`)

| 工具 | 功能 |
|------|------|
| `optimize_portfolio` | 组合优化（风险平价/均值方差） |
| `calculate_portfolio_risk` | 风险计算（VaR/CVaR/最大回撤） |
| `plan_trade_execution` | 执行规划（分批建仓） |
| `get_portfolio_snapshot` | 持仓快照（市值/盈亏） |

---

## 9. TODO / Roadmap

### 已完成

- [x] MyAgent：通用股票分析 Agent
- [x] FinancialAgent：财报分析 Agent
- [x] TradeAgent：交易决策 Agent
  - [x] 组合优化（风险平价/均值方差）
  - [x] 风险计算（VaR/CVaR/最大回撤）
  - [x] 执行规划（分批建仓）
  - [x] 持仓快照
- [x] Agent 协作架构
  - [x] Coordinator 协调器
  - [x] 智能路由
  - [x] 预定义工作流（完整分析/快速选股/财务验证）
  - [x] 共享上下文

### 待开发

- [ ] **持仓管理器** (`TradeAgent/portfolio_manager.py`)
  - [ ] 模拟持仓记录
  - [ ] 交易日志
  - [ ] 盈亏追踪与可视化

- [ ] **市场情绪模块**
  - [ ] 构建情绪指标：涨跌家数比、涨停跌停比、成交额变化
  - [ ] 输出情绪分数（0-100）与状态标签（Risk-On / Neutral / Risk-Off）
  - [ ] 在 `dashboard.py` 增加情绪仪表盘

- [ ] **策略增强**
  - [ ] 行业/主题维度对比与强弱排序
  - [ ] 多因子权重可配置
  - [ ] 滚动窗口回测与样本外验证
  - [ ] 交易成本敏感性分析

- [ ] **工程化**
  - [ ] 单元测试 (`tests/`)
  - [ ] 日志与错误追踪
  - [ ] 配置文件 (`config.yaml`)
  - [ ] CI（lint + test）

---

## 10. 常见问题

- 启动时报 `Please set the GITHUB_TOKEN environment variable.`
  - 原因：未配置 `LLM_API_KEY` 或 `GITHUB_TOKEN`
  - 处理：检查 `.env` 文件或系统环境变量

- 查询不到股票数据
  - 原因：代码格式或交易日范围问题
  - 处理：优先使用 6 位代码（如 `600000`、`000001`），并确认日期区间

## 11. 合规与风险提示

- 不保证收益，不承担任何投资损失责任
- 实盘前请自行进行更严格的数据验证与风险控制
