# CSI 300 Return Prediction & Portfolio Construction

沪深300 成分股的**中期收益预测与组合构建**项目：用梯度提升树 + 浅层神经网络做横截面收益预测，
再叠加质量因子、波动率惩罚与仓位约束，输出一个最多 5 只股票的投资组合。

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
<img alt="models" src="https://img.shields.io/badge/models-XGBoost%20%2B%20MLP-orange">
<img alt="license" src="https://img.shields.io/badge/license-all%20rights%20reserved-lightgrey">
</p>

本项目不是简单的「模型打分 → 排序 → 全仓买入」，而是把**预测**与**组合构建**分成两层：
模型只负责给出收益预期，仓位由风险调整后的 alpha 与置信度共同决定，允许保留现金。

---

## Highlights

- **标签严格对齐持仓收益定义** —— `(T+5 开盘价 − T+1 开盘价) / T+1 开盘价`，训练与实盘口径一致。
- **双模型融合** —— XGBoost 捕捉树结构非线性，MLP 捕捉连续特征组合；两者预测值加权融合。
- **样本加权** —— 对绝对收益更大的样本赋予更高权重，提升对高收益机会的识别能力。
- **55 维滚动特征** —— 多窗口动量、量价、波动率、市场相对强弱、横截面分位五类。
- **策略层与预测层解耦** —— 模型分数不当权重用；先筛候选，再按风险调整后 alpha 分配，单票上限 35%。
- **允许空仓** —— 置信度不足或全市场预期为负时保留现金，不做机械满仓。
- **完全离线可复现** —— 固定随机种子，不联网、不依赖外部预训练权重。

---

## Pipeline

```
data/train.csv  (历史日频行情)
      │
      ├─► 字段标准化 / 类型清洗
      │
      ├─► 滚动特征工程 (55 维)
      │      动量 · 量价 · 波动率 · 市场相对 · 横截面分位
      │
      ├─► 监督标签  y = (open[T+5] − open[T+1]) / open[T+1]
      │
      ├─► 双模型训练
      │      XGBoost Regressor  ┐
      │      MLPRegressor      ┴─► 融合分数
      │                              │
      └──────────────────────────────┴─► 组合构建
                                          候选筛选 (≤5 只)
                                             ↓
                                         质量因子 × 波动率惩罚
                                             ↓
                                         总仓位 / 单票上限 35%
                                             ↓
                                     output/result.csv
```

---

## Project Structure

```
code/src/
  featurework.py   数据读取、特征工程、组合构建、结果落盘
  train.py         双模型训练与模型保存
  test.py          推理、策略分配、结果输出
data/
  train.csv        历史日频行情（训练与推理的唯一数据源）
  test.csv         占位文件，训练与推理均不读取
model/
  xgb_return_model.joblib   训练产物（已附带一份，可直接推理；也可用 train.sh 重训）
output/           组合输出目录（运行时生成 result.csv）
temp/             中间明细（运行时生成，不入库）
init.sh            创建必要目录
train.sh           训练入口
test.sh            推理入口
Dockerfile         容器化运行环境
```

代码通过 `Path(__file__).resolve().parents[2]` 定位项目根，因此在任何工作目录下运行都能正确找到
`data/`、`model/`、`output/`、`temp/`。

---

## Quick Start

### 本地运行

```bash
pip install -r requirements.txt

sh init.sh      # 创建 model/ output/ temp/ 目录
sh train.sh     # 训练并保存模型（若已有模型可跳过）
sh test.sh      # 推理并生成 output/result.csv
```

也可以直接调用 Python：

```bash
python code/src/train.py
python code/src/test.py
```

### Docker 运行

```bash
docker build -t csi300-portfolio .
docker run --rm -v "$PWD/output:/app/output" csi300-portfolio
```

---

## Data Format

`data/train.csv` 为长表格式，每行是「一只股票 × 一个交易日的横截面行情」：

| 文件列名 | 内部字段名 | 说明 |
|---|---|---|
| `股票代码` | `stock_id` | 股票代码，内部统一补零为 6 位 |
| `日期` | `trade_date` | 交易日期 |
| `开盘` / `收盘` / `最高` / `最低` | `open` / `close` / `high` / `low` | 开高低收 |
| `成交量` / `成交额` | `volume` / `amount` | 成交量、成交额 |
| `振幅` | `amplitude` | 振幅 |
| `涨跌额` | `change_amount` | 涨跌额 |
| `换手率` | `turnover_rate` | 换手率 |
| `涨跌幅` | `pct_change` | 涨跌幅 |

列名通过 `COLUMN_ALIASES` 映射，也兼容英文列名，因此换成同义表头的数据可直接复用。

**当前数据快照**

| 项 | 取值 |
|---|---|
| 样本行数 | 166,490 |
| 股票数 | 300 |
| 交易日数 | 557 |
| 日期范围 | 2024-01-02 ~ 2026-04-23 |

数据来源为公开可获取的沪深300 历史行情，不含任何收费或私有数据。

> `data/test.csv` 仅作为目录结构占位，**训练与推理都不会读取它**。

---

## Model & Strategy

### 特征工程（55 维）

| 类别 | 内容 |
|---|---|
| 动量 | `1/2/3/5/10/20` 日收益与开盘收益变化 |
| 量价 | 成交量变化、成交额/成交量、换手率均值 |
| 波动率 | 多窗口涨跌幅标准差、振幅、日内价差 |
| 市场相对 | 相对市场平均涨跌幅、相对市场换手率 |
| 横截面 | 当日成交量、涨跌幅、换手率的分位 |

### 模型

| 组件 | 配置 |
|---|---|
| 主模型 | `XGBRegressor`，目标 `reg:squarederror` |
| 子模型 | `MLPRegressor(hidden_layer_sizes=(128, 64, 32))` |
| 融合 | 预测值加权组合，再叠加质量因子与波动率惩罚 |
| 权重 | 对绝对收益更大的样本追加训练权重 |

### 组合构建

1. 按融合分数筛选出最多 5 只候选；
2. 用风险调整后的 alpha 重新分配权重，降低高波动股票占比；
3. 依据组合整体置信度决定总仓位，允许低于 100%；
4. 单票权重上限 **35%**；
5. 若所有候选预期收益不为正，则完全空仓。

未分配部分即现金，对应输出中权重和可小于 1。

---

## Output Format

运行 `test.sh` 后生成 `output/result.csv`（UTF-8），格式如下：

```csv
stock_id,weight
000001,0.3325
000002,0.3209
```

约束：

- `stock_id` 为 6 位股票代码，不重复
- 股票数量 ≤ 5
- 单票权重 ≤ 0.35
- 权重总和 ≤ 1，未分配部分视为现金

---

## Training Report

每次训练会在 `temp/training_report.json` 写入验证指标，样例：

| 指标 | 取值 |
|---|---|
| 训练样本 / 验证样本 | 124,524 / 23,843 |
| 特征维数 | 55 |
| 验证 RMSE | 0.0472 |
| XGBoost RMSE | 0.0480 |
| MLP RMSE | 0.0485 |

---

## Reproducibility

- 固定随机种子，训练过程可复现；
- 不联网，不下载任何外部权重；
- 若 `model/xgb_return_model.joblib` 不存在，`test.py` 会自动触发训练；
- 建议在 Docker 中使用固定依赖版本，避免浮点与版本差异。

---

## Notes & Limitations

- 项目面向**横截面排序**任务，绝对收益预测误差（RMSE ≈ 0.047）远大于收益本身，
  因此策略层依赖分位与风险惩罚而非绝对数值；
- 单一数据源（日频行情），未纳入基本面、新闻情绪、行业轮动等信号；
- 未做交易成本、涨跌停、停牌等实盘约束建模；
- 回测与实盘存在滑点与流动性差异，结果仅供参考，**不构成投资建议**。

---

## License

本项目未附许可证文件，默认保留所有权利。
行情数据版权归其原始提供方所有。
