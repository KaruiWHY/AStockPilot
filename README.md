# MyAgent 股票分析助手

基于 `OpenAI-compatible API`、`baostock`、`Streamlit` 的 A 股分析项目，支持：

- 多轮对话式个股分析（行情、历史 K 线、技术指标）
- 大盘概览联动分析（上证/深证/创业板）
- 双均线策略回测与参数网格搜索
- 因子选股与交易员风格候选筛选
- 可视化面板（`dashboard.py`）

仅供研究与学习使用，不构成任何投资建议。

## 1. 环境要求

- Python 3.10+
- Windows / macOS / Linux（当前仓库主要在 Windows 环境使用）

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 配置环境变量

在项目根目录创建 `.env` 文件（不要提交到 Git）：

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

## 6. 常见问题

- 启动时报 `Please set the GITHUB_TOKEN environment variable.`
  - 原因：未配置 `LLM_API_KEY` 或 `GITHUB_TOKEN`
  - 处理：检查 `.env` 文件或系统环境变量

- 查询不到股票数据
  - 原因：代码格式或交易日范围问题
  - 处理：优先使用 6 位代码（如 `600000`、`000001`），并确认日期区间

## 7. 合规与风险提示

- 本项目输出内容仅用于研究与教学演示
- 不保证收益，不承担任何投资损失责任
- 实盘前请自行进行更严格的数据验证与风险控制
