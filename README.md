# AStockPilot 股票分析助手

基于 `OpenAI-compatible API`、`baostock`、`Streamlit` 的 A 股分析项目，支持：

- 多轮对话式个股分析（行情、历史 K 线、技术指标）
- 大盘概览联动分析（上证/深证/创业板）
- 双均线策略回测与参数网格搜索
- 因子选股与交易员风格候选筛选
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
python run.py
```

启动后可选择模型并进入对话，输入 `exit` 或 `quit` 退出。

### 4.2 Streamlit 可视化面板

```bash
streamlit run dashboard.py
```

默认会展示市场概览、K 线与策略分析相关模块。

## 5. 项目结构

```text
MyAgent/
├─ agent.py                # LLM Agent 核心：工具定义、对话管理、函数调用
├─ run.py                  # 命令行入口
├─ dashboard.py            # Streamlit 仪表盘入口
├─ requirements.txt        # Python 依赖
├─ tools/
│  ├─ stock_tools.py       # 股票数据、技术指标、回测、选股工具
│  └─ ...
└─ outputs/
   └─ plots/               # 运行后生成的 HTML 图表
```

## 6. TODO / Roadmap

- [ ] 多 Agent 架构（Planner + Specialist）
- [ ] 增加 `MarketAgent`：负责大盘状态识别与风格轮动判断
- [ ] 增加 `StockAgent`：负责个股技术面/量价结构分析
- [ ] 增加 `RiskAgent`：负责仓位、止损止盈、回撤约束建议
- [ ] 增加 `BacktestAgent`：负责策略回测与参数稳健性评估

- [ ] 市场情绪分析模块（Market Sentiment）
- [ ] 构建情绪指标：涨跌家数比、涨停跌停比、成交额变化、指数波动率
- [ ] 输出情绪分数（0-100）与状态标签（Risk-On / Neutral / Risk-Off）
- [ ] 在 `dashboard.py` 增加情绪仪表盘与历史情绪曲线
- [ ] 将情绪状态接入选股与仓位决策（情绪弱时降低建议仓位）

- [ ] 市场信息与财报分析
- [ ] 增加 `InfoAgent`：统一检索财报、公告、新闻、研报等外部信息
- [ ] 接入财报结构化解析：营收、归母净利、毛利率、现金流、资产负债率、ROE
- [ ] 增加财报同比/环比与预期偏差分析（Beat/Miss）
- [ ] 增加管理层表述变化跟踪（年报/季报 MD&A 关键句对比）
- [ ] 增加公告事件识别：回购、减持、再融资、诉讼、监管问询、分红
- [ ] 增加新闻情感与事件冲击评分，并与技术面信号融合
- [ ] 增加信息可信度分层与多源交叉验证（避免单一信源误导）
- [ ] 在 `dashboard.py` 增加“市场信息与财报解读”页面

- [ ] 策略与选股增强
- [ ] 增加行业/主题维度对比与强弱排序
- [ ] 增加多因子权重可配置（动量、趋势、波动、回撤、流动性）
- [ ] 增加滚动窗口回测与样本外验证（避免过拟合）
- [ ] 增加交易成本敏感性分析（手续费/滑点参数扫描）

- [ ] 工程化与可维护性
- [ ] 增加 `tests/` 与核心工具函数单元测试
- [ ] 增加日志与错误追踪（工具调用耗时、失败原因）
- [ ] 增加配置文件（如 `config.yaml`）统一管理策略参数
- [ ] 增加 CI（lint + test）与版本发布说明

## 7. 常见问题

- 启动时报 `Please set the GITHUB_TOKEN environment variable.`
  - 原因：未配置 `LLM_API_KEY` 或 `GITHUB_TOKEN`
  - 处理：检查 `.env` 文件或系统环境变量

- 查询不到股票数据
  - 原因：代码格式或交易日范围问题
  - 处理：优先使用 6 位代码（如 `600000`、`000001`），并确认日期区间

## 8. 合规与风险提示

- 不保证收益，不承担任何投资损失责任
- 实盘前请自行进行更严格的数据验证与风险控制
